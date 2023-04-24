from src import GP_kernels as gpk
from jax.random import PRNGKey, split
from jax import numpy as jnp
from jax import vmap
from numpyro.handlers import trace


def test_sample_kernel():
  key = PRNGKey(0)
  kernel = gpk.sample_kernel(key)
  #this key samples a quadratic kernel with lenght_scale=0.4261 and scale_mixture=0.5156
  
  samples = jnp.arange(100, dtype=jnp.float32)/10 #range from 0 to 10
  
  manual_kernel = gpk.RationalQuadraticKernel(lenght_scale=0.4261, scale_mixture=0.5156)

  cov_kernel = vmap(vmap(kernel, in_axes=(None, 0)), in_axes=(0, None))(samples, samples)
  cov_manual_kernel = vmap(vmap(manual_kernel, in_axes=(None, 0)), in_axes=(0, None))(samples, samples)
  
  assert jnp.allclose(cov_kernel, cov_manual_kernel, atol=1e-4), "sampled kernel is not equal to the manual kernel"
  
def test_model():
  key = PRNGKey(0)
  t = trace(gpk.model).get_trace(key)
  from ipdb import set_trace; set_trace()
  assert t['idx']['value'].item() == 1.0 #sampled kernel is a quadratic kernel
  #test the closeness of the parameters of the sampled kernel with 0.01 tolerance
  assert jnp.allclose(t['lenght_scale']['value'].item(), 0.4100382, atol=1e-2)
  assert jnp.allclose(t['scale_mixture']['value'].item(), 0.34801906, atol=1e-2)
  
  
if __name__ == '__main__':
  test_model()
  test_sample_kernel()