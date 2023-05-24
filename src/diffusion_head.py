
'''
see (this annotation)[https://nn.labml.ai/diffusion/ddpm/index.html] for where I got the implementation with some tweaks
However, I'm predicting X0 directly given it's noisy version, as it was the same in
(this paper)[https://arxiv.org/pdf/2205.09991.pdf#page=11&zoom=100,48,413]

'''


from typing import List
from dataclasses import dataclass

import jax
from jax.random import split, PRNGKey
from jax import numpy as jnp
from jax import lax
import equinox as eqx
import numpy as np

from tensorflow_probability.substrates import jax as tfp_j
tfd_j = tfp_j.distributions

from src.ResnetMLP import ResnetMLP

from src.Normalizer import Normalizer

import logging
logger = logging.getLogger(__name__)

@dataclass
class DiffusionConf:
    num_latents: int = 1
    width_size: int = 128
    depth: int = 3
    num_conds: int =128
    num_steps: int = 100
    noise_scale: float = 0.008
    dropout_rate: float = 0.1
    use_normalizer: bool = True
    
class DiffusionNet(eqx.Module):
    layers: List[ResnetMLP]
    output_layer: eqx.nn.Linear
    
    def __init__(self, *, num_latents, num_conds, width_size, depth, dropout_rate, key):
        assert depth >= 1
        key, sk = split(key, 2)
        self.layers = [
            ResnetMLP(
                width_size=width_size,
                in_size=num_latents + num_conds,
                out_size=num_latents+num_conds,
                dropout_rate=dropout_rate,
                key=k
            )
            for k in split(key, depth)
        ]
        
        self.output_layer = eqx.nn.Linear(
            in_features=num_latents+num_conds, 
            out_features=num_latents,
            key=sk,
        )
        
        
    def __call__(self, z, cond_vars, key):
        assert len(cond_vars.shape) == 1
        z = jnp.concatenate([z, cond_vars])
        for layer in self.layers:
            z = layer(z, key=key)
        z = self.output_layer(z)
        return z

class DiffusionHead(eqx.Module):
    diffusion_net: DiffusionNet
    normalizer: Normalizer
    num_steps: int = eqx.static_field()
    sqrt_one_minus_alphas_cumprod: jnp.array = eqx.static_field()
    sqrt_alphas_cumprod: jnp.array = eqx.static_field()
    time_pos_emb: jnp.array = eqx.static_field()
    mu_1: jnp.array = eqx.static_field()
    mu_2: jnp.array = eqx.static_field()
    posterior_std: jnp.array = eqx.static_field()
    use_normalizer: bool = eqx.static_field()

    def __init__(self, *, c: DiffusionConf, key: PRNGKey):
                 
        num_latents, num_conds, width_size, depth, num_steps, noise_scale, dropout_rate, use_normalizer = \
            map(lambda x: getattr(c, x), ['num_latents', 'num_conds', 'width_size', 'depth', 'num_steps', 'noise_scale', 'dropout_rate', 'use_normalizer'])
            
        ks = split(key, 2)
        self.diffusion_net = DiffusionNet(
            num_latents=num_latents,
            num_conds=num_conds,
            width_size=width_size,
            depth=depth,
            dropout_rate=dropout_rate,
            key=ks[0],
        )
        self.normalizer = Normalizer(
            num_latents=num_latents,
            num_conds=num_conds,
            hidden_size=width_size,
            key=ks[1]
        )
        self.time_pos_emb = jnp.array(positional_encoding(num_steps, num_conds))

        self.num_steps = num_steps
        betas = cosine_beta_schedule(num_steps, noise_scale)

        #for training
        alphas = 1. - betas
        alphas_cumprod = jnp.cumprod(alphas)
        self.sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1. - alphas_cumprod)

        #for sampling --> refer to DDPM https://arxiv.org/pdf/2006.11239v2.pdf equation (7)
        alphas_cumprod_prev = jnp.concatenate([jnp.array([1.]), alphas_cumprod[:-1]])
        self.mu_1 = betas * jnp.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.mu_2 = (1. - alphas_cumprod_prev) * jnp.sqrt(alphas) / (1. - alphas_cumprod)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_std = (jnp.clip(posterior_variance, 1e-20, 1e20)) ** 0.5
        
        self.use_normalizer = use_normalizer
        
    def eval_log_p(self, z, cond_vars, key, init_logp=0.0):
        logger.warning('this adds an MSE and the logprobability of the data under the normalizer, which is not sound, but the higher the better')

        ks = split(key, 3)

        t = jax.random.choice(ks[0], jnp.arange(self.num_steps), shape=(),)
        # t = jax.random.randint(ks[0], (), 0, self.num_steps)

        q_noise = jax.random.normal(ks[1], z.shape)
        q_sample = self.sqrt_alphas_cumprod[t] * z + \
                    self.sqrt_one_minus_alphas_cumprod[t] * q_noise

        p_sample = self.diffusion_net(q_sample, 
                                      cond_vars + self.time_pos_emb[t], 
                                      key=ks[2])

        return -((p_sample - q_sample) ** 2).mean()#-jnp.abs(p_sample - q_sample).mean()#

    def log_p(self, z, cond_vars, key, init_logp=0.0):
        logger.warning('only a loss, discrete diffusion does not directly compute log_likelihoods')
        assert z.ndim == 1 and z.shape[0] == 1
        if self.use_normalizer:
            normalizer_log_p = self.normalizer.gaussian_log_p(z, cond_vars)
            z, inv_log_det_jac_normalizer = self.normalizer.reverse(z, cond_vars)
        else:
            normalizer_log_p = 0.0
            inv_log_det_jac_normalizer = 0.0

        log_prob = init_logp + inv_log_det_jac_normalizer

        recon_loss = self.eval_log_p(z, cond_vars, key, init_logp=init_logp)
        
        return normalizer_log_p + log_prob + recon_loss.mean()

    def rsample_slow(self, cond_vars, key):
        assert cond_vars.ndim == 1

        key, sk = split(key, 2)

        x_t = jax.random.normal(sk, (1,))

        log_p = 0.0

        for t in reversed(range(self.num_steps)):
            key, *ks = split(key, 3)
            x_0_pred = self.diffusion_net(x_t, cond_vars + self.time_pos_emb[t], key=ks[0])
            
            mu = self.mu_1[t] * x_0_pred + self.mu_2[t] * x_t
            sigma = self.posterior_std[t]
            q_posterior = tfd_j.Normal(loc=mu,scale=sigma)

            x_t = q_posterior.sample(seed=ks[1], sample_shape=())
            log_p += q_posterior.log_prob(x_t)
            
        x_0_pred, forward_log_det_jac = self.normalizer.forward(x_0_pred, cond_vars)
        log_p += forward_log_det_jac
        
        return x_0_pred, log_p
    
    def rsample(self, cond_vars, key):
        assert cond_vars.ndim == 1

        key, sk = split(key, 2)

        x_t = jax.random.normal(sk, (1,))

        log_p = jnp.zeros((1,))

        def scan_fn(carry, t):
            key, x_t, x_0_pred, log_p = carry
            key, *ks = split(key, 3)
            x_0_pred = self.diffusion_net(x_t, cond_vars + self.time_pos_emb[t], key=ks[0])

            mu = self.mu_1[t] * x_0_pred + self.mu_2[t] * x_t
            sigma = self.posterior_std[t]
            q_posterior = tfd_j.Normal(loc=mu,scale=sigma)

            x_t = q_posterior.sample(seed=ks[1], sample_shape=())
            log_p += q_posterior.log_prob(x_t)

            return (key, x_t, x_0_pred, log_p), None

        (key, x_t, x_0_pred, log_p), _ = jax.lax.scan(scan_fn, (key, x_t, x_t, log_p), jnp.arange(self.num_steps, -1, -1))

        if self.use_normalizer:
            x_0_pred, forward_log_det_jac = self.normalizer.forward(x_0_pred, cond_vars)
            log_p += forward_log_det_jac

        return x_0_pred, log_p




def cosine_beta_schedule(timesteps, s=0.008,):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return betas_clipped

def positional_encoding(num_tokens, d_model):
    def inner_loop_fn(carry, i, pos, d_model):
        sin_val = jnp.sin(pos / jnp.power(10000, (2 * i) / d_model))
        cos_val = jnp.cos(pos / jnp.power(10000, (2 * (i + 1)) / d_model))
        carry = carry.at[i].set(sin_val)
        carry = carry.at[i + 1].set(cos_val)
        return carry, None
    def outer_loop_fn(carry, pos, d_model):
        init_carry = carry[pos]
        i_vals = jnp.arange(0, d_model, 2)
        final_carry, _ = lax.scan(lambda c, i: inner_loop_fn(c, i, pos, d_model), init_carry, i_vals)
        carry = carry.at[pos].set(final_carry)
        return carry, None
    pos_enc = jnp.zeros((num_tokens, d_model))
    pos_vals = jnp.arange(num_tokens)
    pos_enc, _ = lax.scan(lambda c, pos: outer_loop_fn(c, pos, d_model), pos_enc, pos_vals)
    return pos_enc