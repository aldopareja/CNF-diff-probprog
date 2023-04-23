from src import GP_kernels as gpk
from jax.random import PRNGKey, split
from jax import numpy as jnp
from jax import vmap


def test_sample_kernel():
  key = PRNGKey(0)
  kernel = gpk.sample_kernel(key)
  #this key samples a quadratic kernel with lenght_scale=0.4261 and scale_mixture=0.5156
  
  samples = jnp.arange(100, dtype=jnp.float32)/10 #range from 0 to 10
  
  manual_kernel = gpk.RationalQuadraticKernel(lenght_scale=0.4261, scale_mixture=0.5156)

  cov_kernel = vmap(vmap(kernel, in_axes=(None, 0)), in_axes=(0, None))(samples, samples)
  cov_manual_kernel = vmap(vmap(manual_kernel, in_axes=(None, 0)), in_axes=(0, None))(samples, samples)
  
  assert jnp.allclose(cov_kernel, cov_manual_kernel, atol=1e-4), "sampled kernel is not equal to the manual kernel"
  
  
if __name__ == '__main__':
  test_sample_kernel()