from collections import namedtuple
from jax.random import PRNGKey
from jax import numpy as jnp
from jax import vmap
from tensorflow_probability.substrates import jax as tfp_j
tfd_j = tfp_j.distributions

import optax

import equinox as eqx

def dict_to_namedtuple(d: dict, name: str = 'NamedTuple'):
    return namedtuple(name, d.keys())(*[dict_to_namedtuple(d[k], k) if isinstance(d[k], dict) else d[k] for k in d])

def standardize(a,mu,std):
  assert jnp.array(mu).ndim == 0 and jnp.array(std).ndim == 0
  return (a - mu)/std

def unstandardize(a,mu,std):
  assert jnp.array(mu).ndim == 0 and jnp.array(std).ndim == 0
  return a*std + mu

def augment_sample(k: PRNGKey, s, num_augment):
    new_s = jnp.concatenate(
        [s, tfd_j.Normal(0, 1).sample(seed=k, sample_shape=(num_augment,))]
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
  cdf1 = tfd_j.Empirical(s1)
  cdf2 = tfd_j.Empirical(s2)
  
  s = jnp.arange(resolution)/resolution * (r_max - r_min) + r_min
  
  cdf1 = vmap(cdf1.cdf)(s)
  cdf2 = vmap(cdf2.cdf)(s)
  
  ks_test = jnp.abs(cdf1 - cdf2).max()
  return ks_test
  
def compare_discrete_samples(y1,y2):
  """computes how many times two variables are the same
  """
  assert y1.shape == y2.shape and y1.ndim == 1
  return jnp.where(
    y1==y2, 
    jnp.ones((y1.shape[0],)), 
    jnp.zeros((y1.shape[0]))
  ).sum()/y1.shape[0]
  
def initialize_optim(optim_cfg, model):
    c = optim_cfg
    schedule = optax.cosine_onecycle_schedule(
                c.num_steps,
                c.max_lr,
                c.pct_start,
                c.div_factor,
                c.final_div_factor,
            )
      
    optim = optax.chain(
        optax.clip_by_global_norm(c.gradient_clipping),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=c.weight_decay,
        ),
    )

    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    return optim, opt_state, schedule