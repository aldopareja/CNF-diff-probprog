from jax.random import PRNGKey
from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

def augment_sample(k:PRNGKey, s ,num_augment):
  new_s = jnp.concatenate([s,
                           tfd.Normal(0,1).sample(seed=k,sample_shape=(num_augment,))
                           ]
  )
  return new_s