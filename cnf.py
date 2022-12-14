#Credit, inspired by https://docs.kidger.site/diffrax/examples/continuous_normalising_flow/

from typing import List

import jax
from jax import numpy as jnp
from jax.random import split

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import equinox as eqx
import diffrax

class ConcatSquash(eqx.Module):
  lin1: eqx.nn.Linear
  lin2: eqx.nn.Linear
  lin3: eqx.nn.Linear
  
  def __init__(self, *, in_size, out_size, key, **kwargs):
    super().__init__(*kwargs)
    k1, k2, k3 = split(key, 3)
    self.lin1 = eqx.nn.Linear(in_size, out_size, key=k1)
    self.lin2 = eqx.nn.Linear(1, out_size, key=k2)
    self.lin3 = eqx.nn.Linear(1, out_size, use_bias=False, key=k3)
    
  def __call__(self, t, x):
    return self.lin1(x) * jax.nn.sigmoid(self.lin2(t)) + self.lin3(t)

class Func(eqx.Module):
  layers: List[eqx.nn.Linear]
  
  def __init__(self, *, num_latents, num_conds, width_size, depth, key, **kwargs):
    super().__init__(**kwargs)
    assert depth>=1
    ks = split(key,depth+1)
    layers = []
    layers.append(
      ConcatSquash(in_size=num_latents+num_conds, out_size=width_size, key=ks[0])
    )
    for i in range(depth - 1):
      layers.append(
        ConcatSquash(
          in_size=width_size, out_size=width_size, key=ks[i+1]
        )
      )
    layers.append(
      ConcatSquash(in_size=width_size, out_size=num_latents, key=ks[-1])
    )
    self.layers = layers
  
  def __call__(self, t, z, args):
    cond_vars = args[0]
    assert len(cond_vars.shape) == 1
    t = jnp.asarray(t)[None]
    z = self.layers[0](t,jnp.concatenate([z,cond_vars]))
    z = jax.nn.tanh(z)
    for layer in self.layers[1:-1]:
      z = layer(t,z)
      z = jax.nn.tanh(z)
    dz_dt = self.layers[-1](t,z)
    return dz_dt
  
def exact_logp_wrapper(t, z, args):
    z, _ = z
    *args, func = args
    fn = lambda y: func(t, y, args)
    f, vjp_fn = jax.vjp(fn, z)
    (size,) = z.shape  # this implementation only works for 1D input
    eye = jnp.eye(size)
    (dfdy,) = jax.vmap(vjp_fn)(eye)
    logp = jnp.trace(dfdy)
    return f, logp

class CNF(eqx.Module):
  funcs: List[Func]
  num_latents: int
  t0: float
  t1: float
  dt0: float
  
  def __init__(
    self,
    *,
    num_latents,
    num_conds,
    width_size,
    num_blocks,
    depth,
    key,
    **kwargs,  
  ):
    super().__init__(**kwargs)
    ks = split(key, num_blocks)
    self.funcs = [
      Func(
        num_latents=num_latents,
        num_conds=num_conds,
        width_size=width_size,
        depth=depth,
        key=k,
      )
      for k in ks
    ]
    self.num_latents = num_latents
    self.t0 = 0
    self.t1 = 1
    self.dt0 = 0.1
  
  def log_p(self, z, cond_vars):
    term = diffrax.ODETerm(exact_logp_wrapper)
    solver = diffrax.Tsit5(scan_stages=False)
    delta_log_likelihood = 0.0
    for func in reversed(self.funcs):
      z = (z, delta_log_likelihood)
      sol = diffrax.diffeqsolve(
        term, solver, self.t1, self.t0, -self.dt0, z, (cond_vars, func)
      )
      (z,), (delta_log_likelihood,) = sol.ys
    return delta_log_likelihood + tfd.Normal(0,1).log_prob(z).sum()
  
  def rsample(self, key, cond_vars):
    z = tfd.Normal(0,1).sample(seed=key, sample_shape=(self.num_latents,))
    for func in self.funcs:
      term = diffrax.ODETerm(func)
      solver = diffrax.Tsit5()
      sol = diffrax.diffeqsolve(term, solver, self.t0, self.t1, self.dt0, z, (cond_vars,))
      (z,) = sol.ys
    return z
  
