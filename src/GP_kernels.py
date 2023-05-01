from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import List, Tuple
import jax
import numpyro as npy
from numpyro import distributions as dist
from jax import numpy as jnp
import equinox as eqx
from jax import jit, vmap
from jax.random import PRNGKey, split

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
  discrete_mlp_width:int = 128
  discrete_mlp_depth:int=2
  continuous_flow_blocks:int=8
  continuous_flow_num_layers_per_block:int=2
  continuous_flow_num_augment:int=91
  num_enc_layers:int=4
  max_discrete_choices:int =5
  num_input_variables:Tuple[int] = (1,2)
  num_observations:int =100
  max_num_latents:int = 100

class GPInference(eqx.Module):
  input_encoder: Encoder
  continuous_flow_dist: RealNVP_Flow
  discrete_mlp_dist: eqx.nn.MLP
  obs_to_embed_list: List[eqx.nn.MLP]
  num_input_variables: Tuple[int]
  num_observations: int
  
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
    
    lim = 1 / math.sqrt(c.d_model)
    self.latent_input_embeddings = jax.random.uniform(ks[1], (c.max_num_latents, c.d_model), minval=-lim, maxval=lim)
    
    self.num_input_variables = c.num_input_variables
    
                                         
    
  def log_p(self, t, key):
    mask, indices, variables = t['attention_mask'], t['indices'], t['trace']
    
    enc_input = []
    outputs = []
    is_discrete = []
    for k,v in variables.items():
      assert len(v.shape) == 2 and v.shape[1] in self.num_input_variables
      input = vmap(self.obs_to_embed_list[v.shape[1]])(jnp.float32(v))
      enc_input.append(input)
      if k != 'obs':
        assert v.shape[1] == 1, 'only single dimensional latents are supported, consider flattening multivariate samples'
        outputs.append(jnp.float32(v))
        is_discrete.append(jax.lax.cond(v.dtype == jnp.int32, lambda: True, lambda: False, None))
        
    enc_input = jnp.concatenate(enc_input, axis=0)[indices]
    outputs = jnp.concatenate(outputs, axis=0)[indices]
    is_discrete = jnp.stack(is_discrete)[indices]
    
    ks = split(key, len(t['values']))
    
    masks = self.make_masks(mask, self.num_observations)
    
    #start from second value since the first is the observation
    trs = [(ks[i], outputs[i], is_discrete[i], masks[i], enc_input[:i+self.num_observations]) 
           for i in range(len(outputs))]
    all_log_p = []
    for i in range(len(trs)):
      log_p = self.process_trace_element(trs[i])
      all_log_p.append(log_p)
    
    return jnp.sum(jnp.stack(all_log_p))
  
  def rsample(self, obs, key):
    assert obs.shape == () #TODO: this only works for unary observations, which is fine, but not necessarily
    key, sk1,sk2 = split(key,3)
    
    sample = OrderedDict()
    sample['obs'] = obs
    input = jnp.broadcast_to(obs, (1, self.num_input_variables))
    
    emb = self.input_encoder(input, key=sk2, mask = jnp.ones((2,), dtype=bool))[-1]
    
    logits = self.discrete_mlp_dist(emb)
    num_steps = tfd.Categorical(logits=logits).sample(seed=sk1).item()
    print(f"num_steps: {num_steps}")
    sample['num_steps'] = num_steps
    input = jnp.concatenate([input, jnp.full((1,1), jnp.float32(num_steps))], axis=0)
    
    for i in range(num_steps+1):
      key, sk1, sk2 = split(key,3)
      emb = self.input_encoder(input, key=sk1, mask = jnp.ones((3+i,), dtype=bool))[-1]
      val = self.continuous_flow_dist.rsample(key=sk2, cond_vars=emb)
      print(f"val_{i}: {val}")
      sample[f"val_{i}"] = val
      input = jnp.concatenate([input, jnp.full((1,1), val)], axis=0)
      
    return sample
    
    
  
  @staticmethod
  def make_masks(mask, num_observations):
    '''
    make masks for each step in the trace, attending at the observation and up to the current variable
    ex:
    [Array([ True], dtype=bool),
    Array([ True,  True], dtype=bool),
    Array([ True,  True,  True], dtype=bool),
    Array([ True,  True,  True,  True], dtype=bool),
    Array([ True,  True,  True,  True,  True], dtype=bool),
    Array([ True,  True,  True,  True,  True, False], dtype=bool),
    Array([ True,  True,  True,  True,  True, False, False], dtype=bool)]
    '''
    masks = []
    for i in range(num_observations, len(mask)):
      mask_i = jnp.where(jnp.arange(i+1)<mask.sum(), True, False)
      masks.append(mask_i)
    return masks
  
  # @eqx.filter_jit
  def process_trace_element(self, t_i):
    key, variable_value, discrete, mask, so_far_input = t_i
    # add latent embedding to input
    latent_emb = self.latent_input_embeddings[so_far_input.shape[0]-self.num_observations][None]
    encoder_input = jnp.concatenate([so_far_input,latent_emb], axis=0)
    
    encoder_input += self.positional_encoding(*encoder_input.shape)
    
    ks_ = split(key, 2)
    emb = self.input_encoder(encoder_input, key=ks_[0], mask=mask)[-1]
    
    log_p = jax.lax.cond(
      discrete,
      lambda: self.discrete_log_prob(emb, jnp.int32(variable_value)),
      lambda: self.continuous_log_prob(emb, jnp.float32(variable_value), ks_[1]),
    )
    log_p = jax.lax.cond(mask[-1]==True, lambda: log_p, lambda: 0.)
    return log_p
        
  def discrete_log_prob(self, emb, value):
    logits = self.discrete_mlp_dist(emb)
    assert logits.ndim == 1
    log_p = logits[value] - jax.nn.logsumexp(logits)
    return log_p

  def continuous_log_prob(self, emb, value, key):
    return self.continuous_flow_dist.log_p(z=value[None], cond_vars=emb, key=key)
  
  @staticmethod
  @eqx.filter_jit
  def positional_encoding(num_tokens, d_model):
    # Initialize a matrix with shape (num_tokens, d_model)
    pos_enc = jnp.zeros((num_tokens, d_model))

    # Calculate positional encoding for each token
    for pos in range(num_tokens):
        for i in range(0, d_model, 2):
            pos_enc = pos_enc.at[pos, i].set(jnp.sin(pos / jnp.power(10000, (2 * i) / d_model)))
            pos_enc = pos_enc.at[pos, i + 1].set(jnp.cos(pos / jnp.power(10000, (2 * (i + 1)) / d_model)))

    return pos_enc
        
    

  
