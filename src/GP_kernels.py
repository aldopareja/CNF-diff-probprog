from dataclasses import dataclass
import jax
import numpyro as npy
from numpyro import distributions as dist
from jax import numpy as jnp
import equinox as eqx
from jax import jit, vmap
from jax.random import PRNGKey, split

from src.encoder import Encoder, EncoderCfg
from src.real_nvp import RealNVP_Flow


class RationalQuadraticKernel(eqx.Module):
    lenght_scale: float
    scale_mixture: float

    @eqx.filter_jit
    def __call__(self, x1, x2):
        squared_scaled_distance = jnp.square(x1 - x2) / jnp.square(self.lenght_scale)
        return jnp.power(
            (1 + 0.5 * squared_scaled_distance / self.scale_mixture),
            -self.scale_mixture,
        )


class LinearKernel(eqx.Module):
    bias: float

    @eqx.filter_jit
    def __call__(self, x1, x2):
        return x1 * x2 + self.bias


def sum_kernels(k1, k2):
    return lambda x1, x2: k1(x1, x2) + k2(x1, x2)


def multiply_kernels(k1, k2):
    return lambda x1, x2: k1(x1, x2) * k2(x1, x2)


def sample_kernel(key: PRNGKey, address_prefix=""):
    ks = split(key, 4)
    idx = npy.sample(
        f"{address_prefix}idx",
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
  @jit
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
  kernel = jit(sample_kernel(ks[0]))
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
  max_discrete_choices:int =4
  num_input_variables:int =2
  num_observations:int =100

class GPInference(eqx.Module):
  input_encoder: Encoder
  continuous_flow_dist: RealNVP_Flow
  discrete_mlp_dist: eqx.nn.MLP
  num_input_variables: int
  
  def __init__(
    self,
    *,
    key: PRNGKey,
    c: GPInferenceCfg = GPInferenceCfg(),
  ):
    ks = split(key, 3)
    
    ec = EncoderCfg(num_enc_layers=c.num_enc_layers,
                    d_model=c.d_model,
                    dropout_rate=c.dropout_rate,
                    num_input_variables=c.num_input_variables,
                    num_observations=c.num_observations)
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
    
    self.num_input_variables = c.num_input_variables
    
  def log_p(self, t, key):
    values, _, is_discrete, mask = t['values'], t['is_obs'], t['is_discrete'], t['attention_mask']
    assert len(values) == len(is_discrete) == len(mask)
    assert mask.ndim == 1
    
    obs = values[0]
    input = jnp.broadcast_to(obs, (1, self.num_input_variables))
    
    ks = split(key, len(t['values']))
    
    masks = self.make_masks(mask)
    
    #start from second value since the first is the observation
    trs = [(ks[i], values[i], is_discrete[i], masks[i]) for i in range(1,len(values))]
    all_log_p = []
    for i in range(len(trs)):
      input, log_p = self.process_trace_element(input, trs[i])
      all_log_p.append(log_p)
    
    return jnp.sum(jnp.stack(all_log_p))
  
  @staticmethod
  def make_masks(mask):
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
    for i in range(len(mask)):
      mask_i = jnp.where(jnp.arange(i+1)<mask.sum(), True, False)
      masks.append(mask_i)
    return masks
  
  # @eqx.filter_jit
  def process_trace_element(self, input, t_i):
    key, variable_value, discrete, mask = t_i
    ks_ = split(key, 2)
    emb = self.input_encoder(input, key=ks_[0], mask=mask)[-1]
    log_p = jax.lax.cond(
      discrete,
      lambda: self.discrete_log_prob(emb, jnp.int32(variable_value)),
      lambda: self.continuous_log_prob(emb, jnp.float32(variable_value), ks_[1]),
    )
    new_input = jnp.concatenate([input, jnp.full((1,input.shape[1]), variable_value)], axis=0)
    return new_input, log_p
        
  def discrete_log_prob(self, emb, value):
    logits = self.discrete_mlp_dist(emb)
    log_p = logits[value] - jax.nn.logsumexp(logits)
    return log_p

  def continuous_log_prob(self, emb, value, key):
    return self.continuous_flow_dist.log_p(z=value[None], cond_vars=emb, key=key)
        
    

  
