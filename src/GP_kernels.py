from collections import OrderedDict, namedtuple
from functools import partial
import math
import numpyro as npy
from numpyro import distributions as dist
from jax import numpy as jnp
import equinox as eqx
from jax import vmap
from jax.random import PRNGKey, split

#TODO: I should use numpyro
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


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
    assert value.shape == (self.num,2)
    x, y = value[:,0], value[:,1]
    
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
  return obs, obs_dist

def model(key:PRNGKey, return_dist=False):
  ks = split(key, 3)
  kernel = sample_kernel(ks[0])
  obs, obs_dist =  sample_observations(ks[3], kernel, 100)
  if return_dist:
    return obs, obs_dist
  else:
    return obs

  


        
    

  


