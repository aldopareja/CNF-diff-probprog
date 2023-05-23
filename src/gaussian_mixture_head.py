from dataclasses import dataclass

import jax
from jax.random import split, PRNGKey
from jax import numpy as jnp

from tensorflow_probability.substrates import jax as tfp_j
tfd_j = tfp_j.distributions
tfb_j = tfp_j.bijectors

import equinox as eqx

from src.Normalizer import Normalizer

import logging
logger = logging.getLogger(__name__)

@dataclass
class GaussianMixtureCfg:
  mlp_width: int
  d_model: int
  mlp_depth: int
  num_mixtures: int

class GaussianMixture(eqx.Module):
  '''
  only supporting single variable latents
  '''
  mu_sigma_mlp: eqx.nn.MLP
  mixture_mlp: eqx.nn.MLP
  normalizer: Normalizer
  
  def __init__(self, *, c: GaussianMixtureCfg, key):
    ks = split(key, 3)
    self.mu_sigma_mlp = eqx.nn.MLP(
      in_size=c.d_model,
      out_size= c.num_mixtures * 2, # mean and variance for each mixture
      width_size=c.mlp_width,
      depth=c.mlp_depth,
      key=ks[0]
    )
    self.mixture_mlp = eqx.nn.MLP(
      in_size=c.d_model,
      out_size=c.num_mixtures,
      width_size=c.mlp_width,
      depth=c.mlp_depth,
      key=ks[1]
    )
    self.normalizer = Normalizer(
      num_latents=1,
      num_conds=c.d_model,
      hidden_size=c.mlp_width,
      key=ks[2]
    )
    
  def eval_log_p(self, z, cond_vars, key, init_logp=0.0):
    z, inv_log_det_jac_normalizer = self.normalizer.reverse(z, cond_vars)
    
    log_prob = init_logp + inv_log_det_jac_normalizer
    
    mu, sigma = jnp.split(self.mu_sigma_mlp(cond_vars), 2)
    sigma = jax.nn.softplus(sigma)
    
    log_prob_gaussians = tfd_j.Normal(loc=mu, scale=sigma).log_prob(z)
    log_p_categories = jax.nn.log_softmax(self.mixture_mlp(cond_vars))
    #log_p(z) = log_sum_exp(log_p(z|μ_i, σ_i) + log_p(mix_i))
    log_prob_gmmm = jax.nn.logsumexp(log_p_categories + log_prob_gaussians)
    
    return log_prob + log_prob_gmmm
  
  def log_p(self, z, cond_vars, key, init_logp=0.0):
    assert z.ndim == 1 and z.shape[0] == 1
    assert cond_vars.ndim == 1
    normalizer_log_p = self.normalizer.gaussian_log_p(z, cond_vars)
    return normalizer_log_p + self.eval_log_p(z, cond_vars, key, init_logp)
  
  def rsample(self, key, cond_vars):
    logger.warning('GaussianMixture.rsample is not a reparameterized sample')
    ks = split(key, 2)
    mu, sigma = jnp.split(self.mu_sigma_mlp(cond_vars), 2)
    sigma = jax.nn.softplus(sigma)
    dist = tfd_j.Normal(loc=mu, scale=sigma)
    z = dist.sample(seed=ks[0])
    
    logits = jax.nn.log_softmax(self.mixture_mlp(cond_vars))
    sampled_mixture = tfd_j.Categorical(logits=logits).sample(seed=ks[1])
    
    z = z[sampled_mixture][None]
    z, _ = self.normalizer.forward(z, cond_vars)
    return z

