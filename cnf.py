# Credit, inspired by https://docs.kidger.site/diffrax/examples/continuous_normalising_flow/

from typing import List

import jax
from jax import numpy as jnp
from jax.random import split

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

import equinox as eqx
import diffrax

from functools import partial


from utils import augment_sample


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
        assert depth >= 1
        ks = split(key, depth + 1)
        layers = []
        layers.append(
            ConcatSquash(
                in_size=num_latents + num_conds, out_size=width_size, key=ks[0]
            )
        )
        for i in range(depth - 1):
            layers.append(
                ConcatSquash(in_size=width_size, out_size=width_size, key=ks[i + 1])
            )
        layers.append(
            ConcatSquash(in_size=width_size, out_size=num_latents, key=ks[-1])
        )
        self.layers = layers

    def __call__(self, t, z, args):
        cond_vars = args[0]
        assert len(cond_vars.shape) == 1
        t = jnp.asarray(t)[None]
        z = self.layers[0](t, jnp.concatenate([z, cond_vars]))
        z = jax.nn.tanh(z)
        for layer in self.layers[1:-1]:
            z = layer(t, z)
            z = jax.nn.tanh(z)
        dz_dt = self.layers[-1](t, z)
        return dz_dt

# the function to drive the differential equation (i.e. in dx/dt = g(x,t), this would be g)
# in particular, the ODE we're solving is actually two ODEs stacked, so [dz/dt, d ln(q)/dt] = [f, -\nabla_z \cdot f]
# So this gives 
def exact_logp_wrapper(func, t, odeVar, args):
    # takes:
        # t
        # z
        # (cond_vars, f)
    # returns: [f(z,t), -\nabla_z \cdot f(z,t)]
    z, _ = odeVar
    # *args = args
    fVal, vjp_fn = jax.vjp(lambda y: func(t, y, args), z)
    (size,) = z.shape  # this implementation only works for 1D input
    eye = jnp.eye(size)
    (dfdy,) = jax.vmap(vjp_fn)(eye) # don't quite get why we don't do a matrix multiply here
    div_z_f = jnp.trace(dfdy)
    return fVal, div_z_f


class CNF(eqx.Module):
    funcs: List[Func]
    num_latents: int
    num_augments: int
    t0: float
    t1: float
    dt0: float

    def __init__(
        self,
        *,
        num_latents,
        num_augments=0,
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
                num_latents=num_latents + num_augments,
                num_conds=num_conds,
                width_size=width_size,
                depth=depth,
                key=k,
            )
            for k in ks
        ]
        self.num_latents = num_latents
        self.num_augments = num_augments
        self.t0 = 0
        self.t1 = 1
        self.dt0 = 0.1

    def log_p(self, z, cond_vars, key):
        solver = diffrax.Tsit5(scan_stages=False)

        # initial values
        z_aug = augment_sample(key, z, self.num_augments)
        delta_log_likelihood = 0.0

        for func in reversed(self.funcs): # reversed because we're going backwards
            # solving two 1D diff-eqs at once, formulated as a single 2D diff-eq
            sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(partial(exact_logp_wrapper, func)), # i.e. the "f" in dx/dt = f(x,t)
                solver=solver,
                t0=self.t1, # initial time (we're running it backwards)
                t1=self.t0, # final time
                dt0=-1.0, # step size
                y0=(z_aug, delta_log_likelihood),
                args=(cond_vars,), # additional vars passed to f. passing func is a bit weird, I wonder if I can refactor...
                stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            )
            (z_aug,), (delta_log_likelihood,) = sol.ys
        # the log prob in the observation space, adjusted as necessary by the determinant of the jacobian
        return delta_log_likelihood + tfd.Normal(0, 1).log_prob(z_aug).sum() 

    def rsample(self, key, cond_vars):
        z = tfd.Normal(0, 1).sample(
            seed=key, sample_shape=(self.num_latents + self.num_augments,)
        )
        for func in self.funcs:
            term = diffrax.ODETerm(func)
            solver = diffrax.Tsit5()
            sol = diffrax.diffeqsolve(
                terms=term,
                solver=solver,
                t0=self.t0,
                t1=self.t1,
                dt0=1.0,
                y0=z,
                args=(cond_vars,),
                stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            )
            (z,) = sol.ys
        return z[: self.num_latents] # why would z be larger? should this be an assert check?
