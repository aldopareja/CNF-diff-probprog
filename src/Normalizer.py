import equinox as eqx
import jax
from jax import numpy as jnp

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors



class Normalizer(eqx.Module):
  normalizer_nn: eqx.nn.MLP

  def __init__(self, *, num_latents, num_conds, hidden_size, key):
    self.normalizer_nn = eqx.nn.MLP(
      in_size=num_conds,
      out_size=num_latents*2,
      width_size=hidden_size,
      depth=1,
      key=key,
    )

  def forward(self, x, cond_vars):
    assert x.ndim == 1
    mu, sigma = self.normalizer_nn(cond_vars).split(2)
    sigma = jax.nn.softplus(sigma)
    means_, stds_ = map(jax.lax.stop_gradient,
                        (mu, sigma))
    bijector = tfb.Chain([tfb.Shift(means_), tfb.Scale(stds_)])
    return bijector.forward(x), bijector.forward_log_det_jacobian(x).sum()

  def reverse(self, y, cond_vars):
    assert y.ndim == 1
    mu, sigma = self.normalizer_nn(cond_vars).split(2)
    sigma = jax.nn.softplus(sigma)
    means_, stds_ = map(jax.lax.stop_gradient,
                        (mu, sigma))
    bijector = tfb.Chain([tfb.Shift(means_), tfb.Scale(stds_)])
    return bijector.inverse(y), bijector.inverse_log_det_jacobian(y).sum()

  def gaussian_log_p(self, x, cond_vars):
    assert x.ndim == 1
    mu, sigma = self.normalizer_nn(cond_vars).split(2)
    sigma = jax.nn.softplus(sigma)
    bijector = tfb.Chain([tfb.Shift(mu), tfb.Scale(sigma)])
    log_p = bijector.inverse_log_det_jacobian(x)
    x_ = bijector.inverse(x)
    log_p += tfd.Normal(loc=jnp.zeros_like(mu),
                        scale=jnp.ones_like(sigma)).log_prob(x_)
    return log_p.sum()