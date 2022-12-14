from jax.random import split, PRNGKey
from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

def bayesian_network(k:PRNGKey):
  sks = split(k,10)
  z0 = tfd.Laplace(5,1).sample(seed=sks[0])
  z1 = tfd.Laplace(-2,1).sample(seed=sks[1])
  z2 = tfd.Normal(jnp.tanh(z0 + z1 - 2.8),0.1).sample(seed=sks[2])
  z3 = tfd.Normal(z0 * z1,0.1).sample(seed=sks[3])
  z4 = tfd.Normal(7,2).sample(seed=sks[4])
  z5 = tfd.Normal(jnp.tanh(z3 + z4),0.1).sample(seed=sks[5])
  x0 = tfd.Normal(z3,0.1).sample(seed=sks[6])
  x1 = tfd.Normal(z5,0.1).sample(seed=sks[7])
  return [v[None] for v in [z0, z1, z2, z3, z4, z5, x0, x1]]

def augmented_sample(k:PRNGKey, s ,num_augment):
  new_s = jnp.concatenate([s,
                           tfd.Normal(0,1).sample(seed=k,sample_shape=(num_augment,))
                           ]
  )
  return new_s