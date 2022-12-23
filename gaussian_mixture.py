
from jax import numpy as jnp
from jax.random import PRNGKey, split
from jax import vmap

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions


def sample_cov_matrices(dims,max_num_mixtures,k,eps=1e-5):
  """ 
  if you get a lower triangular matrix L then you can ensure a positive 
  definite covariance matrix by computing:
  
  cov = LL' + eps*I
  
  see https://stackoverflow.com/questions/40573478/ensuring-positive-definite-covariance-matrix
  """
  t = tfd.Uniform(low=0.01,high=1.0).sample(seed=k,sample_shape=(max_num_mixtures,int(dims * (dims + 1) / 2)))
  t = jnp.concatenate([t, t[:, dims:][:, ::-1]], axis=-1)
  t = t.reshape((max_num_mixtures,dims,dims))
  t = jnp.triu(t)
  
  eps_I = jnp.eye(dims)[None] * eps
  cov_matrices = jnp.matmul(t, t.swapaxes(-2, -1)) + eps_I
  return cov_matrices

def sample_observations(means, cov_matrices, class_label, k):
  s = tfd.MultivariateNormalFullCovariance(
    loc=means[class_label],
    covariance_matrix=cov_matrices[class_label]
    ).sample(seed=k)
  
  return s

def gaussian_mixture(k:PRNGKey, *, max_num_mixtures=6, dims=2, num_obs=100):
  ks = split(k,5)
  num_mixtures = tfd.Categorical(
    probs=jnp.ones((max_num_mixtures,))/max_num_mixtures
    ).sample(seed=ks[0])
  
  means = tfd.Uniform(low=-1.0,high=1.0).sample(seed=ks[1],sample_shape=(max_num_mixtures,dims))
  cov_matrices = sample_cov_matrices(dims, max_num_mixtures, ks[2])
  
  class_labels_probs = jnp.ones((max_num_mixtures,))
  class_labels_probs = class_labels_probs.at[num_mixtures:].set(0)
  class_labels = tfd.Categorical(
    probs=class_labels_probs
    ).sample(seed=ks[3],sample_shape=(num_obs,))
  
  obs = vmap(sample_observations,in_axes=(None,None,0,0))(
    means,
    cov_matrices,
    class_labels,
    split(ks[4],num_obs)
    )
  
  return num_mixtures, means, cov_matrices, class_labels, obs
  
  