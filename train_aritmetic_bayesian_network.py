"""
test if the inference network produces samples that are similar to the joint.
To do this we will compare the inference network and samples from the joint 
according to:

.. math:: q(z|x)p(x) \\sim p(z,x)

This was taken from section 4.1 of [1]

[1] Weilbach, C., Beronov, B., Wood, F.D., & Harvey, W. (2020). Structured Conditional Continuous Normalizing Flows for Efficient Amortized Inference in Graphical Models. AISTATS.
"""
from pathlib import Path
import time

import jax
from jax import vmap
from jax import numpy as jnp
from jax.random import split, PRNGKey
import optax
import equinox as eqx
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from src.utils import AttrDict, ks_test
from aritmetic_bayesian_network import bayesian_network, InferenceForBayesianNetwork

def initialize_model(
  model_cfg:AttrDict,
  load_idx: int,
  chkpt_folder: str,
):
  m = InferenceForBayesianNetwork(**model_cfg)
  if load_idx is not None:
    p = Path(chkpt_folder) / f'{load_idx}.eqx'
    m = eqx.tree_deserialise_leaves(p,m)
  
  return m

def initialize_optim(optim_cfg, model):
  c = optim_cfg
  optim = optax.chain(
    optax.clip_by_global_norm(c.gradient_clipping),
    optax.adamw(
      learning_rate=optax.cosine_onecycle_schedule(
        c.num_steps,
        c.max_lr,
        c.pct_start,
        c.div_factor,
        c.final_div_factor,
      ),
      weight_decay=c.weight_decay
    )
  )
  
  opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
  
  return optim, opt_state

@eqx.filter_value_and_grad
def loss(model: InferenceForBayesianNetwork, s_batch, key):
  z0_b, z1_b, z2_b, z3_b, z4_b, z5_b, x0_b, x1_b = s_batch
  ks = split(key,z0_b.shape[0])
  log_p = vmap(model.log_p)(z0_b, z1_b, z2_b, z3_b, z4_b, z5_b, x0_b, x1_b, ks)
  return -log_p.mean()
  
@eqx.filter_jit
def make_step(model, opt_state, key):
  ks = split(key, c.batch_size+2)
  s_batch = vmap(bayesian_network)(ks[:c.batch_size])
  l, grads = loss(model,s_batch, ks[-2])
  updates, opt_state = optim.update(grads, opt_state, model)
  model = eqx.apply_updates(model, updates)
  norm = jnp.linalg.norm(jax.flatten_util.ravel_pytree(model)[0])
  return l, model, opt_state, ks[-1], norm

@eqx.filter_jit
def evaluate(model: InferenceForBayesianNetwork,key):
  ks = split(key, c.eval_size*2)
  
  *all_z_batches, x0_b, x1_b = vmap(bayesian_network)(ks[:c.eval_size])
  all_z_batches_hat = vmap(model.rsample)(x0_b,x1_b,ks[c.eval_size:])
  
  kolmogorov_smirnovs = []
  for z,z_ in zip(all_z_batches,all_z_batches_hat):
    assert z.shape == z_.shape and z.ndim == 2 and z.shape[1] == 1
    kolmogorov_smirnovs.append(ks_test(z.reshape(-1),z_.reshape(-1)))
    
  return kolmogorov_smirnovs
  
  
if __name__ == "__main__":
  import wandb
  
  wandb.login()
  wandb.init(project="artimetic_bayesian_network")
  
  c = AttrDict()
  c.key = PRNGKey(574)
  k,subkey = split(c.key, 2)
  
  #Inference network
  c.m_cfg = AttrDict(
    key = subkey,
    num_augments = 9,
    width_size = 128,
    num_blocks = 1,
    depth = 3,
  )
  
  #logging and checkpointing
  c.log_chk = AttrDict(
    save_params = 10,
    print_every = 10,
    chkpt_folder = "aritmetic_bayesian_chkpts/",
    load_idx = None,
    evaluate_iters = 10,
  )
  save_idx = c.log_chk.load_idx + 1 if c.log_chk.load_idx is not None else 0
  
  #optimization cfg
  c.batch_size = 1000
  c.eval_size = 10000
  c.opt_c = AttrDict(
    max_lr=0.025,
    num_steps = int(10000),
    pct_start = 0.01,
    div_factor=1e1,
    final_div_factor=1e1,
    weight_decay=0.0005,
    gradient_clipping=5.0,
  )
  
  wandb.config.update(c)
  
  m = initialize_model(model_cfg=c.m_cfg, 
                       load_idx = c.log_chk.load_idx, 
                       chkpt_folder=c.log_chk.chkpt_folder)
  
  optim, opt_state = initialize_optim(c.opt_c, m)
  
  for i in range(c.opt_c.num_steps):
    start = time.time() if i == 0 or i == 1 else None
    l, m, opt_state, k, norm = make_step(m, opt_state, k)
    end = time.time() if i == 0 or i == 1 else None
    
    log = AttrDict(
      loss = l.item(),
      norm = norm.item(),
      batch = i,
      )
    wandb.log(log)
    if i % c.log_chk.print_every == 0 or i == 1:
      print(log)
      
    if i == 0 or i == 1:
      print("iter_time", end-start)
      
    if i % c.log_chk.save_params == 0:
      p = Path(c.log_chk.chkpt_folder) / f'{save_idx}.eqx'
      eqx.tree_serialise_leaves(p, m)
      
    if i % c.log_chk.evaluate_iters == 0:
      k, sk = split(k)
      ks_stats = evaluate(m,sk)
      log = {f'z{i}_ks':ks.item() for i,ks in enumerate(ks_stats)}
      log["batch"] = i
      wandb.log(log)
      print(log)