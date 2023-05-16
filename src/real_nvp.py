from typing import List

import jax
from jax import jit, numpy as jnp
from jax.random import PRNGKey, split
from jaxtyping import Array

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import equinox as eqx

from src.utils.miscellaneous import augment_sample


def concat_elu(x):
  assert x.ndim == 1
  return jnp.concatenate([jax.nn.elu(x), jax.nn.elu(-x)])

class GatedDenseLayer(eqx.Module):
  l1: eqx.nn.Linear
  l2: eqx.nn.Linear
  
  def __init__(self, *, hidden_size, key:PRNGKey):
    super().__init__()
    k1, k2 = split(key)
    self.l1 = eqx.nn.Linear(hidden_size*2,hidden_size,key=k1)
    self.l2 = eqx.nn.Linear(hidden_size*2, hidden_size*2,key=k2)
    
  def __call__(self, x):
    assert x.ndim == 1 and x.shape[0] % 2 == 0
    out = concat_elu(x)
    out = self.l1(out)
    out = concat_elu(out)
    out = self.l2(out)
    val, gate = out.split(2)
    return x + val * jax.nn.sigmoid(gate)
  
class GatedDenseNet(eqx.Module):
  l_in: eqx.nn.Linear
  layers: List[GatedDenseLayer]
  l_out: eqx.nn.Linear
  in_size: int
  
  def __init__(self, *, num_layers, in_size, out_size ,hidden_size, key):
    ks = split(key,num_layers + 2)
    self.l_in = eqx.nn.Linear(in_size, hidden_size, key=ks[-1])
    self.layers = [GatedDenseLayer(hidden_size=hidden_size, key=ks[i]) for i in range(num_layers)]
    self.l_out = eqx.nn.Linear(hidden_size*2, out_size, key=ks[-2])
    self.in_size = in_size
    
  def __call__(self, x):
    assert x.ndim == 1 and x.shape[0] == self.in_size
    x = self.l_in(x)
    for l in self.layers:
      x = l(x)
    x = concat_elu(x)
    x = self.l_out(x)
    
    return x
  
class RealNVPLayer(eqx.Module):
  # mask: Array
  # non_mask: Array
  num_mask: int
  nn: GatedDenseNet
  scaling_factor: Array
  
  def __init__(self,*,num_layers,num_variables, hidden_size, num_conds, key):
    assert num_variables % 2 == 0 #require odd num_variables to properly split
    # mask = jnp.arange(num_variables)
    # self.mask = mask[:num_variables//2] if even else mask[num_variables//2:]
    # self.non_mask = mask[:num_variables//2] if not even else mask[num_variables//2:]
    self.nn = GatedDenseNet(num_layers=num_layers,
                            in_size=num_variables//2 + num_conds,
                            out_size=num_variables,
                            hidden_size=hidden_size,
                            key=key)
    self.scaling_factor = jnp.zeros(num_variables//2)
    self.num_mask = num_variables//2
    
  def forward(self, x, cond_vars):
    '''
    takes the first half to compute the scale and shift and applies it
    to the second half. Returns the same first half but a different second half.
    We permute the order so the next layer transforms the first half.
    '''
    assert x.ndim == 1
    x_0 = x[:self.num_mask]
    x_1 = x[self.num_mask:]
    
    in_x = jnp.concatenate([x_0,cond_vars])
    log_scale, shift = self.nn(in_x).split(2)
    
    #stabilize the flow
    s_fac = jnp.exp(self.scaling_factor)
    log_scale = jax.nn.tanh(log_scale/s_fac) * log_scale
    
    y_1 = x_1 * jnp.exp(log_scale) + shift
    
    log_det_jac = log_scale.sum()
    
    return jnp.concatenate([y_1, x_0]), log_det_jac
  
  def inverse(self, y, cond_vars):
    '''
    takes the second half to compute the same scale and shift computed in the forward
    uses this value to unscale and unshift the first half and returns the same
    values that were fed in the beginning.
    '''
    assert y.ndim == 1
    
    x_0 = y[self.num_mask:]
    y_1 = y[:self.num_mask]
    
    in_x = jnp.concatenate([x_0,cond_vars])
    log_scale, shift = self.nn(in_x).split(2)
    
    #stabilize the flow
    s_fac = jnp.exp(self.scaling_factor)
    log_scale = jax.nn.tanh(log_scale/s_fac) * log_scale
    
    x_1 = (y_1 - shift) * jnp.exp(-log_scale)
    
    log_det_jac = -log_scale.sum()
    
    return jnp.concatenate([x_0, x_1]), log_det_jac
  
class Normalizer(eqx.Module):
  normalizer_nn: eqx.nn.MLP
  
  def __init__(self, *, num_latents, num_conds, hidden_size, key):
    self.normalizer_nn = eqx.nn.MLP(
      in_size=num_conds,
      out_size=num_latents*2,
      width_size=hidden_size,
      depth=1,
      key=key,
    )
  
  def forward(self, x, cond_vars):
    assert x.ndim == 1
    mu, sigma = self.normalizer_nn(cond_vars).split(2)
    sigma = jax.nn.softplus(sigma)
    means_, stds_ = map(jax.lax.stop_gradient,
                        (mu, sigma))
    bijector = tfb.Chain([tfb.Shift(means_), tfb.Scale(stds_)])
    return bijector.forward(x), bijector.forward_log_det_jacobian(x).sum()
  
  def reverse(self, y, cond_vars):
    assert y.ndim == 1
    mu, sigma = self.normalizer_nn(cond_vars).split(2)
    sigma = jax.nn.softplus(sigma)
    means_, stds_ = map(jax.lax.stop_gradient,
                        (mu, sigma))
    bijector = tfb.Chain([tfb.Shift(means_), tfb.Scale(stds_)])
    return bijector.inverse(y), bijector.inverse_log_det_jacobian(y).sum()
  
  def gaussian_log_p(self, x, cond_vars):
    assert x.ndim == 1
    mu, sigma = self.normalizer_nn(cond_vars).split(2)
    sigma = jax.nn.softplus(sigma)
    bijector = tfb.Chain([tfb.Shift(mu), tfb.Scale(sigma)])
    log_p = bijector.inverse_log_det_jacobian(x)
    x_ = bijector.inverse(x)
    log_p += tfd.Normal(loc=jnp.zeros_like(mu),
                        scale=jnp.ones_like(sigma)).log_prob(x_)
    return log_p.sum()
  
  
    
class RealNVP_Flow(eqx.Module):
  blocks: List[RealNVPLayer]
  num_latents: int
  num_augments: int
  normalizer: Normalizer
  
  def __init__(self,*,num_blocks, num_layers_per_block, block_hidden_size, num_augments, num_latents, num_conds,key):
    ks = split(key, num_blocks + 1)
    self.blocks = [RealNVPLayer(num_layers=num_layers_per_block,
                                num_variables=num_latents+num_augments,
                                num_conds=num_conds,
                                hidden_size=block_hidden_size,
                                key=ks[i])
                   for i in range(num_blocks)]
    self.normalizer = Normalizer(num_latents=num_latents,
                                   num_conds=num_conds,
                                   hidden_size=512,
                                   key=ks[-1])
                   
    self.num_latents = num_latents
    self.num_augments = num_augments
    
  # @eqx.filter_jit
  def log_p(self, z, cond_vars, key, init_logp=0.0):
    assert z.ndim == 1 and z.shape[0] == self.num_latents
    normalizer_log_p = self.normalizer.gaussian_log_p(z, cond_vars)
    
    z, inv_log_det_jac_normalizer = self.normalizer.reverse(z, cond_vars)
    
    log_prob = init_logp + inv_log_det_jac_normalizer
      
    z_aug = augment_sample(key, z, self.num_augments)
    for block in reversed(self.blocks):
      z_aug, inv_log_det_jac = block.inverse(z_aug,cond_vars)
      log_prob += inv_log_det_jac
      
    log_prob += tfd.Normal(0, 1).log_prob(z_aug).sum() 
    
    return log_prob + normalizer_log_p
  
  def eval_log_p(self, z, cond_vars, key, init_logp=0.0):
    '''
    compute log_p without the auxiliary normalizer log_p since for evaluation we only
    care about the log_p of the ground truth data
    '''
    z, inv_log_det_jac_normalizer = self.normalizer.reverse(z, cond_vars)
    
    log_prob = init_logp + inv_log_det_jac_normalizer
      
    z_aug = augment_sample(key, z, self.num_augments)
    for block in reversed(self.blocks):
      z_aug, inv_log_det_jac = block.inverse(z_aug,cond_vars)
      log_prob += inv_log_det_jac
      
    log_prob += tfd.Normal(0, 1).log_prob(z_aug).sum() 
    
    return log_prob
  
  @eqx.filter_jit
  def rsample(self, key, cond_vars):
    z = tfd.Normal(0, 1).sample(
      seed = key, sample_shape=(self.num_latents + self.num_augments,)
    )
    
    for block in self.blocks:
      z, _ = block.forward(z, cond_vars)
    
    z = z[:self.num_latents]
    z, _ = self.normalizer.forward(z, cond_vars)
      
    return z