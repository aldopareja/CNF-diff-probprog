from jax.random import PRNGKey
from jax import numpy as jnp
from jax import vmap
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import equinox as eqx


def augment_sample(k: PRNGKey, s, num_augment):
    new_s = jnp.concatenate(
        [s, tfd.Normal(0, 1).sample(seed=k, sample_shape=(num_augment,))]
    )
    return new_s
  
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

@eqx.filter_jit
def ks_test(s1,s2, resolution=1000, r_max=10, r_min=-10):
  """
  computes a kolmogorov-smirnov test between two samples
  
  assumes values between -10.0 and 10.0
  
  References
  ----------
  [1] https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Two-sample_Kolmogorov%E2%80%93Smirnov_test
  """
  assert s1.shape == s2.shape and s1.ndim == 1
  cdf1 = tfd.Empirical(s1)
  cdf2 = tfd.Empirical(s2)
  
  s = jnp.arange(resolution)/resolution * (r_max - r_min) + r_min
  
  cdf1 = vmap(cdf1.cdf)(s)
  cdf2 = vmap(cdf2.cdf)(s)
  
  ks_test = jnp.abs(cdf1 - cdf2).max()
  return ks_test
  
