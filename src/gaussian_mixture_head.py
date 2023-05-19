from dataclasses import dataclass

import jax
from jax.random import split, PRNGKey
from jax import numpy as jnp

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

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
  mlp: eqx.nn.MLP
  normalizer: Normalizer
  
  def __init__(self, *, c: GaussianMixtureCfg, key):
    ks = split(key, 2)
    self.mlp = eqx.nn.MLP(
      in_size=c.d_model,
      out_size= c.num_mixtures * 2, # mean and variance for each mixture
      width_size=c.mlp_width,
      depth=c.mlp_depth,
      key=ks[0]
    )
    self.normalizer = Normalizer(
      num_latents=1,
      num_conds=c.d_model,
      hidden_size=c.mlp_width,
      key=ks[1]
    )
    
  def eval_log_p(self, z, cond_vars, key, init_logp=0.0):
    z, inv_log_det_jac_normalizer = self.normalizer.reverse(z, cond_vars)
    
    log_prob = init_logp + inv_log_det_jac_normalizer
    
    mu, sigma = jnp.split(self.mlp(cond_vars), 2)
    sigma = jax.nn.softplus(sigma)
    
    log_prob += tfd.Normal(loc=mu, scale=sigma).log_prob(z)
    
    return log_prob.sum()
  
  def log_p(self, z, cond_vars, key, init_logp=0.0):
    assert z.ndim == 1 and z.shape[0] == 1
    assert cond_vars.ndim == 1
    normalizer_log_p = self.normalizer.gaussian_log_p(z, cond_vars)
    return normalizer_log_p + self.eval_log_p(z, cond_vars, key, init_logp)
  
  def rsample(self, key, cond_vars):
    logger.warning('GaussianMixture.rsample is not a reparameterized sample')
    
    mu, sigma = jnp.split(self.mlp(cond_vars), 2)
    sigma = jax.nn.softplus(sigma)
    dist = tfd.Normal(loc=mu, scale=sigma)
    z = dist.sample(seed=key)
    return self.normalizer.forward(z, cond_vars)

