from dataclasses import dataclass
from typing import List

import jax
from jax.random import split, PRNGKey
from jax import numpy as jnp

from tensorflow_probability.substrates import jax as tfp_j

from src.ResnetMLP import ResnetMLP
tfd_j = tfp_j.distributions
tfb_j = tfp_j.bijectors

import equinox as eqx

from src.Normalizer import Normalizer

import logging
logger = logging.getLogger(__name__)

@dataclass
class GaussianMixtureCfg:
  resnet_mlp_width: int
  d_model: int
  num_mixtures: int
  num_mlp_blocks: int
  dropout_rate: float

class GaussianMixture(eqx.Module):
  '''
  only supporting single variable latents
  '''
  mu_sigma_mlps: List[ResnetMLP]
  # mixture_mlps: List[ResnetMLP]
  normalizer: Normalizer
  num_mixtures: int = eqx.static_field()
  
  def __init__(self, *, c: GaussianMixtureCfg, key):
    ks = split(key, 3)
    # self.mu_sigma_mlp = eqx.nn.MLP(
    #   in_size=c.d_model,
    #   out_size= c.num_mixtures * 2, # mean and variance for each mixture
    #   width_size=c.mlp_width,
    #   depth=c.mlp_depth,
    #   key=ks[0]
    # )
    # self.mixture_mlp = eqx.nn.MLP(
    #   in_size=c.d_model,
    #   out_size=c.num_mixtures,
    #   width_size=c.mlp_width,
    #   depth=c.mlp_depth,
    #   key=ks[1]
    # )
    key, *ks = split(key, c.num_mlp_blocks+1)
    self.mu_sigma_mlps = [ResnetMLP(
      width_size=c.resnet_mlp_width,
      in_size=c.d_model,
      out_size=c.d_model,
      dropout_rate=c.dropout_rate,
      key=ks[i],)
      for i in range(c.num_mlp_blocks-1)]
    self.mu_sigma_mlps.append(eqx.nn.Linear(
      in_features=c.d_model,
      out_features=c.num_mixtures * 2 + c.num_mixtures,
      key=ks[-1]
    ))
    self.num_mixtures = c.num_mixtures
    
    # key, *ks = split(key, c.num_mlp_blocks+1)
    # self.mixture_mlps = [ResnetMLP(
    #   width_size=c.resnet_mlp_width,
    #   in_size=c.d_model,
    #   out_size=c.d_model,
    #   dropout_rate=c.dropout_rate,
    #   key=ks[i],)
    #   for i in range(c.num_mlp_blocks-1)]
    # self.mixture_mlps.append(eqx.nn.Linear(
    #   in_features=c.d_model,
    #   out_features=c.num_mixtures,
    #   key=ks[-1]
    # ))
    
    self.normalizer = Normalizer(
      num_latents=1,
      num_conds=c.d_model,
      hidden_size=c.resnet_mlp_width,
      key=ks[2]
    )
    
  def eval_log_p(self, z, cond_vars, key, init_logp=0.0):
    z, inv_log_det_jac_normalizer = self.normalizer.reverse(z, cond_vars)
    # inv_log_det_jac_normalizer = 0.0
    
    log_prob = init_logp + inv_log_det_jac_normalizer
    
    
    mu_sigma_hidden = cond_vars
    for mu_sigma_mlp in self.mu_sigma_mlps:
      key, *ks = split(key, 3)
      mu_sigma_hidden = mu_sigma_mlp(mu_sigma_hidden, key=ks[0])
      # mixture_hidden = mixture_mlp(mixture_hidden, key=ks[1])
      
    mixture_logits = mu_sigma_hidden[-self.num_mixtures:]
    mu, sigma = jnp.split(mu_sigma_hidden[:-self.num_mixtures], 2)
    sigma = jax.nn.softplus(sigma)
    
    log_prob_gaussians = tfd_j.Normal(loc=mu, scale=sigma).log_prob(z)
    log_p_categories = jax.nn.log_softmax(mixture_logits)
    #log_p(z) = log_sum_exp(log_p(z|μ_i, σ_i) + log_p(mix_i))
    log_prob_gmmm = jax.nn.logsumexp(log_p_categories + log_prob_gaussians)
    
    return log_prob + log_prob_gmmm
  
  def log_p(self, z, cond_vars, key, init_logp=0.0):
    assert z.ndim == 1 and z.shape[0] == 1
    assert cond_vars.ndim == 1
    normalizer_log_p = self.normalizer.gaussian_log_p(z, cond_vars)
    # normalizer_log_p = 0.0
    return normalizer_log_p + self.eval_log_p(z, cond_vars, key, init_logp)
  
  def rsample(self, key, cond_vars):
    logger.warning('GaussianMixture.rsample is not a reparameterized sample')
    
    mu_sigma_hidden = cond_vars
    for mu_sigma_mlp in self.mu_sigma_mlps:
      key, *ks = split(key, 3)
      mu_sigma_hidden = mu_sigma_mlp(mu_sigma_hidden, key=ks[0])
    
    
    mixture_logits = mu_sigma_hidden[-self.num_mixtures:]
    mu, sigma = jnp.split(mu_sigma_hidden[:-self.num_mixtures], 2)
    sigma = jax.nn.softplus(sigma)
    dist = tfd_j.Normal(loc=mu, scale=sigma)
    z = dist.sample(seed=ks[0])
    
    logits = jax.nn.log_softmax(mixture_logits)
    sampled_mixture = tfd_j.Categorical(logits=logits).sample(seed=ks[1])
    
    z = z[sampled_mixture][None]
    z, _ = self.normalizer.forward(z, cond_vars)
    return z

