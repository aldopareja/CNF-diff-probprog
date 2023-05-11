from itertools import starmap
from pathlib import Path
import time

import jax
from jax.random import PRNGKey, split
from jax import vmap
from jax import numpy as jnp

import equinox as eqx

from tensorflow_probability.substrates import jax as tfp; tfd = tfp.distributions

from real_nvp import RealNVP_Flow
from src.utils import AttrDict, initialize_optim, ks_test, standardize, unstandardize

'''
computed from 10000 samples from the model
'''
((tau_mu, tau_std),
 (mu_mu,mu_std),
 (theta_mu, theta_std),
 (sigma_mu, sigma_std),
 (y_mu, y_std)) = [(27.653667449951172, 256.46160888671875),
 (0.052995868027210236, 4.991836071014404),
 (-1.0754557847976685, 251.8790740966797),
 (0.5045779943466187, 0.2857249081134796),
 (-1.0753568410873413, 251.87942504882812)]
 
tau_max = 13500.0
mu_max, mu_min = (20.0,-20.0)


def eight_school(k: PRNGKey):
  '''
  implements the 8 school model described in [1]
  
  [1] A. Gelman, J. B. Carlin, H. S. Stern, D. B. Dunson, A. Vehtari, and D. B. Rubin, Bayesian data analysis. CRC press, 2013.
  '''
  ks = split(k, 10)
  
  tau = tfd.HalfCauchy(0,5).sample(seed=ks[0])
  mu = tfd.Normal(0,5).sample(seed=ks[1])
  
  def sample_theta_and_y(k_: PRNGKey):
    ks_ = split(k_,3)
    theta_i = tfd.Normal(mu,tau).sample(seed=ks_[0])
    sigma_i = tfd.Uniform(low=0.01,high=1.0).sample(seed=ks_[1])
    y_i = tfd.Normal(theta_i,sigma_i).sample(seed=ks_[2])
    return y_i, theta_i, sigma_i
  
  y, theta, sigma = vmap(sample_theta_and_y)(split(ks[2],8))
  
  # tau = jnp.repeat(tau, 8)
  # mu = jnp.repeat(mu, 8)
  
  return tau, mu, theta, sigma ,y

class InferenceEightSchool(eqx.Module):
  flow: RealNVP_Flow
  
  def __init__(self,*,key):
    self.flow = RealNVP_Flow(
      num_blocks=10,
      num_layers_per_block=1,
      block_hidden_size=256,
      num_augments=256-18,
      num_latents=18, #tau, mu, 8xsigma_i, 8xtheta_i
      num_conds = 8, #y_i
      key = key,
    )
    
  def log_p(self,tau, mu, theta, sigma, y, key):
    assert tau.ndim == 0 and mu.ndim==0 and theta.shape == (8,) and sigma.shape== (8,) and y.shape == (8,)
    
    tau, mu, theta, sigma, y = starmap(standardize,
                                       [(tau, tau_mu, tau_std),
                                        (mu, mu_mu, mu_std),
                                        (theta, theta_mu, theta_std),
                                        (sigma, sigma_mu, sigma_std),
                                        (y, y_mu, y_std)]
                                       )
    
    tau, mu = tau[None],mu[None]
    z = jnp.concatenate([tau,mu,theta,sigma])
    
    log_p = self.flow.log_p(
      z=z, cond_vars=y, key=key
    )
    
    return log_p
  
  def rsample(self, y, key, unstandardize_sample=True):
    y = standardize(y, y_mu, y_std)
    z = self.flow.rsample(key, y)
    tau, mu = z[:2]
    theta = z[2:2+8]
    sigma = z[10:]
    if unstandardize_sample:
      tau, mu, theta, sigma = starmap(unstandardize,
                                        [(tau, tau_mu, tau_std),
                                          (mu, mu_mu, mu_std),
                                          (theta, theta_mu, theta_std),
                                          (sigma, sigma_mu, sigma_std),
                                          ]
                                        )
    
    return tau, mu, theta, sigma

def initialize_model(
    load_idx: int,
    chkpt_folder: str,
    key: PRNGKey
):
    m = InferenceEightSchool(key=key)
    if load_idx is not None:
        p = Path(chkpt_folder) / f"{load_idx}.eqx"
        m = eqx.tree_deserialise_leaves(p, m)

    return m
  
@eqx.filter_value_and_grad
def loss(model: InferenceEightSchool, sample_batch, key):
  tau, mu, theta, sigma, y = sample_batch
  batch_size = tau.shape[0]
  ks = split(key, batch_size)
  log_p = vmap(model.log_p)(tau, mu, theta, sigma, y, ks[:batch_size])
  
  return -log_p.mean()
  
@eqx.filter_jit
def make_step(model, opt_state, key, virtual_batch_size, num_virtual_batches):
    grads = jax.tree_util.tree_map(
            lambda leaf: jnp.zeros_like(leaf) if eqx.is_inexact_array(leaf) else None,
            model,
        )
    l = 0
    
    #accumulate gradients to simulate larger batches with less memory requirement
    def virtual_step(_, state):
        key, grads, l = state
        ks = split(key, virtual_batch_size + 2)
        s_batch = vmap(eight_school)(ks[: virtual_batch_size])
        l_, grads_ = loss(model, s_batch, ks[-2])
        grads = jax.tree_util.tree_map(lambda a,b: a+b, grads_, grads)
        l += l_
        return ks[-1], grads, l
    
    key, grads, l = jax.lax.fori_loop(0,num_virtual_batches, virtual_step, (key, grads, l))
    
    grads = jax.tree_util.tree_map(lambda a: a/num_virtual_batches, grads)
    l = l/num_virtual_batches
        
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    norm = jnp.linalg.norm(
        jax.flatten_util.ravel_pytree(eqx.filter(model, eqx.is_inexact_array))[0]
    )
    return l, model, opt_state, key, norm
  
@eqx.filter_jit
def evaluate(model: InferenceEightSchool, key, eval_size):
  ks = split(key, eval_size * 2)
  tau, mu, theta, sigma, y = vmap(eight_school)(ks[:eval_size])
  
  tau_hat, mu_hat, theta_hat, sigma_hat = vmap(model.rsample)(y, ks[eval_size:])
  
  tau_fit, mu_fit = starmap(ks_test, 
                            [(tau, tau_hat, 1000, tau_max), 
                             (mu, mu_hat, 1000, mu_max, mu_min)])
  theta_fit = jnp.abs(theta - theta_hat)
  sigma_fit = jnp.abs(sigma - sigma_hat)
  def mean_non_nans(s):
    sum = jnp.where(jnp.isnan(s),0.0,s)
    non_nans = (1 - jnp.isnan(s)).sum()
    return (sum/non_nans+0.0000001).sum()
  
  return dict(tau_fit=tau_fit, 
              mu_fit=mu_fit, 
              theta_fit=mean_non_nans(theta_fit), 
              sigma_fit=mean_non_nans(sigma_fit))
  
    
if __name__ == "__main__":
  import wandb
  
  wandb.login()
  wandb.init(project="eight_school")
  
  c = AttrDict()
  
  c.log_chk = AttrDict(
        save_params=50,
        print_every=50,
        chkpt_folder="eight_school_chkpts/",
        load_idx=None,
        evaluate_iters=100,
    )
  save_idx = c.log_chk.load_idx + 1 if c.log_chk.load_idx is not None else 0
  
  # optimization cfg
  c.virtual_batch_size = 10000
  c.num_virtual_batches = 2
  c.eval_size = 20000
  c.opt_c = AttrDict(
      max_lr=0.0007,
      num_steps=int(500000),
      pct_start=0.0001,
      div_factor=1e0,
      final_div_factor=2e0,
      weight_decay=0.0005,
      gradient_clipping=5.0,
  )
  
  wandb.config.update(c)
  
  m = initialize_model(
        load_idx=c.log_chk.load_idx,
        chkpt_folder=c.log_chk.chkpt_folder,
        key=PRNGKey(123)
    )
  
  optim, opt_state = initialize_optim(c.opt_c, m)
  
  k = PRNGKey(65843)
  
  for i in range(c.opt_c.num_steps):
    start = time.time()
    l, m, opt_state, k, norm = make_step(m, opt_state, k, c.virtual_batch_size, c.num_virtual_batches)
    end = time.time()
    
    log = AttrDict(loss=l.item(), norm=norm.item(), batch=i, time_it=end - start)
    wandb.log(log)
    
    if i % c.log_chk.print_every == 0 or i == 1:
      print(log)
    
    if i == 0 or i == 1:
      print("iter_time", end - start)

    if i % c.log_chk.save_params == 0:
      p = Path(c.log_chk.chkpt_folder) / f"{save_idx}.eqx"
      eqx.tree_serialise_leaves(p, m)
      
    if i % c.log_chk.evaluate_iters == 0:
      start = time.time()
      k, sk = split(k)
      log = evaluate(m, sk, c.eval_size)
      log = jax.tree_util.tree_map(lambda x: x.item(), log)
      end = time.time()
      log["batch"] = i
      log["time_eval"] = end - start
      wandb.log(log)
      print(log)
        
    