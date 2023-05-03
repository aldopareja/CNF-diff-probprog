from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
import math
from typing import List, Tuple
import jax
import numpyro as npy
from numpyro import distributions as dist
from jax import numpy as jnp
import equinox as eqx
from jax import jit, vmap
from jax import lax
from jax.random import PRNGKey, split
from jaxtyping import Array

from src.encoder import Encoder, EncoderCfg
from src.real_nvp import RealNVP_Flow
#TODO: I should use numpyro
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions


class RationalQuadraticKernel(eqx.Module):
    lenght_scale: float
    scale_mixture: float

    # @eqx.filter_jit
    def __call__(self, x1, x2):
        squared_scaled_distance = jnp.square(x1 - x2) / jnp.square(self.lenght_scale)
        return jnp.power(
            (1 + 0.5 * squared_scaled_distance / self.scale_mixture),
            -self.scale_mixture,
        )


class LinearKernel(eqx.Module):
    bias: float

    # @eqx.filter_jit
    def __call__(self, x1, x2):
        return x1 * x2 + self.bias


def sum_kernels(k1, k2):
    def kernel_fn(x1, x2):
        return k1(x1, x2) + k2(x1, x2)
    return kernel_fn


def multiply_kernels(k1, k2):
    def kernel_fn(x1, x2):
        return k1(x1, x2) * k2(x1, x2)
    return kernel_fn


def sample_kernel(key: PRNGKey, address_prefix=""):
    ks = split(key, 4)
    idx = npy.sample(
        f"{address_prefix}kernel_type",
        npy.distributions.Categorical(probs=jnp.array([0.4, 0.4, 0.1, 0.1])),
        rng_key=ks[0],
    )

    if idx == 0.0:
        bias = npy.sample(
            f"{address_prefix}bias",
            npy.distributions.InverseGamma(2.0, 1.0),
            rng_key=ks[1],
        )
        return LinearKernel(bias=bias.item())
    elif idx == 1.0:
        lenght_scale = npy.sample(
            f"{address_prefix}lenght_scale",
            npy.distributions.InverseGamma(2.0, 1.0),
            rng_key=ks[1],
        )
        scale_mixture = npy.sample(
            f"{address_prefix}scale_mixture",
            npy.distributions.InverseGamma(2.0, 1.0),
            rng_key=ks[2],
        )
        return RationalQuadraticKernel(
            lenght_scale=lenght_scale.item(), scale_mixture=scale_mixture.item()
        )
    elif idx == 2.0:
        return sum_kernels(
            sample_kernel(ks[1], address_prefix=f"{address_prefix}leftchild_"),
            sample_kernel(ks[2], address_prefix=f"{address_prefix}rightchild_"),
        )
    elif idx == 3.0:
        return multiply_kernels(
            sample_kernel(ks[1], address_prefix=f"{address_prefix}leftchild_"),
            sample_kernel(ks[2], address_prefix=f"{address_prefix}rightchild_"),
        )
        
class ObsDistribution(dist.Distribution):
  def __init__(self, kernel_fn, num: int, std: float):
    self.kernel_fn = kernel_fn
    self.num = num
    self.std = std
    super().__init__(event_shape=(num,2))

  def sample(self, key):
    ks = split(key, 3)
    x = dist.Uniform(0, 1).sample(ks[0], sample_shape=(self.num,))

    cov = vmap(vmap(self.kernel_fn, in_axes=(None, 0)), in_axes=(0, None))(x, x)
    cov = self.add_noise_to_diagonal(cov, self.std)

    y = dist.MultivariateNormal(loc=jnp.zeros(self.num), covariance_matrix=cov).sample(ks[1])
    return jnp.stack([x, y], axis=1)

  def log_prob(self, value):
    assert value.shape == (2, self.num)
    x, y = value[0,:], value[1,:]
    
    # Calculate the log probability of x values from a Uniform distribution
    log_prob_x = dist.Uniform(0, 1).log_prob(x).sum()
    
    # Calculate the covariance matrix for y values conditioned on x
    cov = vmap(vmap(self.kernel_fn, in_axes=(None, 0)), in_axes=(0, None))(x, x)
    
    #add noise to the diagonal
    cov = self.add_noise_to_diagonal(cov, self.std)
    
    # Calculate the log probability of y values from a MultivariateNormal distribution
    log_prob_y = dist.MultivariateNormal(loc=jnp.zeros(self.num), covariance_matrix=cov).log_prob(y)

    return log_prob_x + log_prob_y
  
  @staticmethod
  # @jit
  def add_noise_to_diagonal(cov,std):
    cov = cov + jnp.eye(cov.shape[0]) * (1e-6 + std)
    return cov
  
        
def sample_observations(key:PRNGKey, kernel_fn, num:int) -> jnp.ndarray:
  ks = split(key, 2)
  obs_dist = ObsDistribution(kernel_fn, num, std=npy.sample('std', dist.HalfNormal(1), rng_key=ks[0]))
  obs_ = obs_dist.sample(ks[1])
  obs = npy.sample('obs', obs_dist, obs=obs_)
  return obs

def model(key:PRNGKey):
  ks = split(key, 3)
  kernel = sample_kernel(ks[0])
  return sample_observations(ks[3], kernel, 100)

@dataclass
class GPInferenceCfg:
  d_model:int = 256
  dropout_rate:float = 0.1
  discrete_mlp_width:int = 512
  discrete_mlp_depth:int=4
  continuous_flow_blocks:int=8
  continuous_flow_num_layers_per_block:int=2
  continuous_flow_num_augment:int=91
  num_enc_layers:int=4
  max_discrete_choices:int =5
  num_input_variables:Tuple[int] = (1,2)
  num_observations:int =100
  slow_compilation:bool = False

class GPInference(eqx.Module):
  input_encoder: Encoder
  continuous_flow_dist: RealNVP_Flow
  discrete_mlp_dist: eqx.nn.MLP
  obs_to_embed_list: List[eqx.nn.MLP]
  num_input_variables: Tuple[int]
  num_observations: int
  slow_compilation: bool = False
  
  def __init__(
    self,
    *,
    key: PRNGKey,
    c: GPInferenceCfg = GPInferenceCfg(),
  ):
    ks = split(key, 4)
    
    ec = EncoderCfg(num_enc_layers=c.num_enc_layers,
                    d_model=c.d_model,
                    dropout_rate=c.dropout_rate,)
    self.input_encoder = Encoder(
      key=ks[0],
      c=ec,
    )
    
    self.discrete_mlp_dist = eqx.nn.MLP(
      in_size = c.d_model,
      out_size = c.max_discrete_choices,
      width_size= c.discrete_mlp_width,
      depth= c.discrete_mlp_depth,
      key = ks[1]
    )
    
    self.continuous_flow_dist = RealNVP_Flow(
      num_blocks=c.continuous_flow_blocks,
      num_layers_per_block=c.continuous_flow_num_layers_per_block,
      block_hidden_size=c.d_model,
      num_augments=c.continuous_flow_num_augment,
      num_latents = 1,
      num_conds = c.d_model,
      key = ks[2]
    )
    
    self.obs_to_embed_list = [eqx.nn.Linear(in_features=i,
                                          out_features=c.d_model,
                                          key=k) 
                              for i,k in zip(c.num_input_variables,
                                             split(ks[3], len(c.num_input_variables)))]
    
    self.num_input_variables = c.num_input_variables
    self.num_observations = c.num_observations
    self.slow_compilation = c.slow_compilation
    
  def log_p(self, t, key):
    mask, indices, variables = t['attention_mask'], t['indices'], t['trace']
    
    enc_input = []
    outputs = []
    is_discrete = []
    
    for k,v in variables.items():
      v = v['value']
      assert len(v.shape) == 2 and v.shape[1] in self.num_input_variables
      input = vmap(self.obs_to_embed_list[v.shape[1]-1])(jnp.float32(v))
      enc_input.append(input)
      if k != 'obs':
        assert v.shape[1] == 1, 'only single dimensional latents are supported, consider flattening multivariate samples'
        outputs.append(jnp.float32(v))
        is_discrete.append(jax.lax.cond(v.dtype == jnp.int32, lambda: True, lambda: False))
      else:
        outputs.append(jnp.zeros((self.num_observations, 1)))
        is_discrete+=[False]*self.num_observations
    
    #shift inputs and outputs by one so output_i is conditioned on input_i
    enc_input = jnp.concatenate(enc_input, axis=0)[indices]
    enc_input += self.positional_encoding(*enc_input.shape)
    
    outputs = jnp.concatenate(outputs, axis=0)[indices]
    is_discrete = jnp.stack(is_discrete)[indices]
    
    outputs = outputs[self.num_observations:]
    is_discrete = is_discrete[self.num_observations:]
    
    key, sk = split(key)
    
    if self.slow_compilation:
      #remove the last element of the input and mask since the ith element of the output should be conditioned on the ith-1 element of the input
      embs = self.get_causal_embs_slow(enc_input, mask, key)
    else:
      embs = self.get_causal_embs(enc_input, mask, sk)
    
    all_log_p = vmap(self.get_causal_log_p)(embs, is_discrete, outputs, split(key, len(is_discrete)))
    return all_log_p.sum()
    
    raise NotImplementedError()
  
  @staticmethod
  def get_causal_mask(mask, i):
    '''returns a mask that is true for all indices before i but no more than mask.sum() indices
    '''
    return jnp.where(jnp.arange(mask.shape[0]) < i, True, False)*mask
  
  @eqx.filter_jit
  def get_causal_log_p(self, emb, is_discrete_i, output_i, key):
    log_p = jax.lax.cond(
      is_discrete_i,
      #round to avoid floating point errors and cast to int for indexing
      lambda: self.discrete_log_prob(emb, jnp.int32(jnp.round(output_i[0]))),
      lambda: self.continuous_log_prob(emb, jnp.float32(output_i), key),
    )
    return log_p
  
  @eqx.filter_jit
  def get_causal_embs_slow(self, enc_input, mask, key):
    ks = split(key, enc_input.shape[0]-self.num_observations)
    embs = []
    for i in range(self.num_observations, enc_input.shape[0]):
      mask_ = mask[:i]
      enc_input_ = enc_input[:i]
      emb = self.input_encoder(enc_input_, mask=mask_, key=ks[i])[-1]
      embs.append(emb)
    return jnp.stack(embs)
  
  @eqx.filter_jit
  def get_causal_embs(self, enc_input, mask, key):
    ks = split(key, enc_input.shape[0])
    masks = vmap(self.get_causal_mask, in_axes=(None,0))(mask, jnp.arange(self.num_observations,enc_input.shape[0]))
    enc_input_ = jnp.broadcast_to(enc_input, (enc_input.shape[0]-self.num_observations, *enc_input.shape))
    get_emb = lambda enc_input_, mask_, k, i: self.input_encoder(enc_input_, mask=mask_, key=k)[i-1]
    embs = vmap(get_emb)(enc_input_, masks, ks[self.num_observations:], i=jnp.arange(self.num_observations, enc_input.shape[0]))
    return embs
  
      
  def discrete_log_prob(self, emb, value):
    assert len(emb.shape) == 1
    assert value.shape == ()
    logits = self.discrete_mlp_dist(emb)
    assert logits.ndim == 1
    log_p = logits[value] - jax.nn.logsumexp(logits)
    return log_p

  def continuous_log_prob(self, emb, value, key):
    return self.continuous_flow_dist.log_p(z=value, cond_vars=emb, key=key)
  
  @staticmethod
  @eqx.filter_jit
  def positional_encoding(num_tokens, d_model):
    def inner_loop_fn(carry, i, pos, d_model):
      sin_val = jnp.sin(pos / jnp.power(10000, (2 * i) / d_model))
      cos_val = jnp.cos(pos / jnp.power(10000, (2 * (i + 1)) / d_model))
      carry = carry.at[i].set(sin_val)
      carry = carry.at[i + 1].set(cos_val)
      return carry, None
    def outer_loop_fn(carry, pos, d_model):
      init_carry = carry[pos]
      i_vals = jnp.arange(0, d_model, 2)
      final_carry, _ = lax.scan(lambda c, i: inner_loop_fn(c, i, pos, d_model), init_carry, i_vals)
      carry = carry.at[pos].set(final_carry)
      return carry, None
    pos_enc = jnp.zeros((num_tokens, d_model))
    pos_vals = jnp.arange(num_tokens)
    pos_enc, _ = lax.scan(lambda c, pos: outer_loop_fn(c, pos, d_model), pos_enc, pos_vals)
    return pos_enc
  
  def rsample(self, obs, key):
    # assert obs.shape == () #TODO: this only works for unary observations, which is fine, but not necessarily
    key, sk1,sk2 = split(key,3)
    
    enc_input = vmap(self.obs_to_embed_list[obs.shape[1]-1])(jnp.float32(obs))
    latent_emb = self.latent_input_embeddings[enc_input.shape[0]-self.num_observations][None]
    
    encoder_input = jnp.concatenate([enc_input,latent_emb], axis=0)
    
    encoder_input += self.positional_encoding(*encoder_input.shape)
    emb = self.input_encoder(encoder_input, key=sk1, mask=jnp.ones(len(encoder_input),dtype=jnp.bool_))[-1]
    logits = self.discrete_mlp_dist(emb)
    num_samples = tfd.Categorical(logits=logits).sample(seed=sk2).item()
    sample = {'num_samples': num_samples}
    
    num_samples_ = jnp.array(num_samples,dtype=jnp.float32)[None]
    num_samples_ = self.obs_to_embed_list[0](num_samples_)[None]
    enc_input = jnp.concatenate([enc_input, num_samples_], axis=0)
    
    for i in range(num_samples+1):
      key, sk1, sk2 = split(key,3)
      latent_emb = self.latent_input_embeddings[enc_input.shape[0]-self.num_observations][None]
      encoder_input = jnp.concatenate([enc_input,latent_emb], axis=0)
      encoder_input += self.positional_encoding(*encoder_input.shape)
      emb = self.input_encoder(encoder_input, key=sk1, mask=jnp.ones(len(encoder_input),dtype=jnp.bool_))[-1]
      
      val = self.continuous_flow_dist.rsample(key=sk2, cond_vars=emb)
      sample[f'sample_{i}'] = val.item()
      
      val_ = self.obs_to_embed_list[0](jnp.float32(val))
      enc_input = jnp.concatenate([encoder_input, val_[None]], axis=0)
      
    return sample

        
    

  
