import numpyro as npy
from numpyro import distributions as dist
from jax import numpy as jnp
import equinox as eqx
from jax import jit, vmap
from jax.random import PRNGKey, split


class RationalQuadraticKernel(eqx.Module):
    lenght_scale: float
    scale_mixture: float

    @jit
    def __call__(self, x1, x2):
        squared_scaled_distance = jnp.square(x1 - x2) / jnp.square(self.lenght_scale)
        return jnp.power(
            (1 + 0.5 * squared_scaled_distance / self.scale_mixture),
            -self.scale_mixture,
        )


class LinearKernel(eqx.Module):
    bias: float

    @jit
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
        
def sample_observations(key:PRNGKey, kernel_fn, num:int) -> jnp.ndarray:
  ks = split(key, 2)
  x_ = dist.Uniform(0, 1).sample(ks[0],sample_shape=(num,))
  x = npy.sample('x', dist.Uniform(0, 1), sample_shape=(num,), obs=x_)
  
  cov = vmap(vmap(kernel_fn, in_axes=(None, 0)), in_axes=(0, None))(x, x)
  
  y = dist.MultivariateNormal(loc=jnp.zeros(num), covariance_matrix=cov).sample(ks[1])
  y = npy.sample('y', dist.MultivariateNormal(loc=jnp.zeros(num), covariance_matrix=cov), obs=y)
  return x,y

def model(key:PRNGKey):
  ks = split(key, 2)
  kernel = jit(sample_kernel(ks[0]))
  return sample_observations(ks[1], kernel, 100)

if __name__ == '__main__':
  x,y = model(PRNGKey(0))
  
  
