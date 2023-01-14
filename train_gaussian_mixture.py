"""
test if the inference network produces samples that are similar to the joint.
To do this we will compare the inference network and samples from the joint 
according to:

.. math:: q(z|x)p(x) \\sim p(z,x)

This was taken from section 4.1 of [1]

[1] Weilbach, C., Beronov, B., Wood, F.D., & Harvey, W. (2020). Structured Conditional Continuous Normalizing Flows for Efficient Amortized Inference in Graphical Models. AISTATS.
"""
from pathlib import Path
import time

import jax
from jax import vmap
from jax import numpy as jnp
from jax.random import split, PRNGKey
import optax
import equinox as eqx
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

from utils import AttrDict, compare_discrete_samples
from gaussian_mixture import gaussian_mixture, InferenceGaussianMixture


def initialize_model(
    model_cfg: AttrDict,
    load_idx: int,
    chkpt_folder: str,
):
    m = InferenceGaussianMixture(**model_cfg)
    if load_idx is not None:
        p = Path(chkpt_folder) / f"{load_idx}.eqx"
        m = eqx.tree_deserialise_leaves(p, m)

    return m


def initialize_optim(optim_cfg, model):
    c = optim_cfg
    optim = optax.chain(
        optax.clip_by_global_norm(c.gradient_clipping),
        optax.adamw(
            learning_rate=optax.cosine_onecycle_schedule(
                c.num_steps,
                c.max_lr,
                c.pct_start,
                c.div_factor,
                c.final_div_factor,
            ),
            weight_decay=c.weight_decay,
        ),
    )

    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    return optim, opt_state


@eqx.filter_value_and_grad
def loss(model: InferenceGaussianMixture, s_batch, key):
    num_mixtures, means, cov_terms, class_labels, obs = s_batch
    assert num_mixtures.ndim==1 and obs.ndim == 3 and obs.shape[2] == 2
    ks = split(key, num_mixtures.shape[0])
    log_p = vmap(model.log_p)(num_mixtures, means, cov_terms, obs, ks)
    return -log_p.mean()


@eqx.filter_jit
def make_step(model, opt_state, key, batch_size):
    ks = split(key, batch_size + 2)
    s_batch = vmap(gaussian_mixture)(ks[: batch_size])
    l, grads = loss(model, s_batch, ks[-2])
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    norm = jnp.linalg.norm(
        jax.flatten_util.ravel_pytree(eqx.filter(model, eqx.is_inexact_array))[0]
    )
    return l, model, opt_state, ks[-1], norm


def trunc_abs_distance(max_num_mixtures, num_mixtures, x, x_hat):
    assert num_mixtures.ndim == 0 and x.shape == x_hat.shape and x.ndim == 2
    abs_dist = jnp.where(
        jnp.arange(max_num_mixtures) <= num_mixtures, jnp.abs(x - x_hat).sum(axis=1), 0.0
    ).sum()
    # normalize by num mixtures and number of variables
    abs_dist /= (num_mixtures + 1) * x_hat.shape[1]
    return abs_dist


@eqx.filter_jit
def evaluate(model: InferenceGaussianMixture, key, eval_size):
    ks = split(key, eval_size * 2)

    num_mixtures, means, cov_terms, class_labels, obs = vmap(gaussian_mixture)(
        ks[: eval_size]
    )
    num_mixtures_hat, means_hat, cov_terms_hat = vmap(model.rsample)(
        obs, ks[eval_size :]
    )
    fit_num_mixtures = compare_discrete_samples(num_mixtures, num_mixtures_hat)
    # return dict(fit_num_mixtures=fit_num_mixtures)
    fit_means = vmap(trunc_abs_distance, in_axes=(None, 0, 0, 0))(
        model.max_num_mixtures, num_mixtures, means, means_hat
    ).mean()
    return dict(fit_num_mixtures=fit_num_mixtures, fit_means=fit_means)
    fit_covs = vmap(trunc_abs_distance, in_axes=(None, 0, 0, 0))(
        model.max_num_mixtures, num_mixtures, cov_terms, cov_terms_hat
    ).mean()

    return dict(
        fit_num_mixtures=fit_num_mixtures, fit_means=fit_means, fit_covs=fit_covs
    )


if __name__ == "__main__":
    import wandb

    wandb.login()
    wandb.init(project="gaussian_mixture_network")

    c = AttrDict()
    c.key = PRNGKey(574)
    k, subkey = split(c.key, 2)

    # Inference network
    c.m_cfg = AttrDict(
        key=subkey,
        max_num_mixtures=6,
        dims=2,
        d_model=256,
        dropout_rate=0.1,
        num_mixtures_mlp_width=100,
        num_mixtures_mlp_depth=1,
        flows_depth=3,
        flows_num_augment=30,
        num_enc_layers=4
    )

    # logging and checkpointing
    c.log_chk = AttrDict(
        save_params=50,
        print_every=50,
        chkpt_folder="gaussian_mixture_chkpts/",
        load_idx=None,
        evaluate_iters=10,
    )
    save_idx = c.log_chk.load_idx + 1 if c.log_chk.load_idx is not None else 0

    # optimization cfg
    c.batch_size = 250
    c.eval_size = 10000
    c.opt_c = AttrDict(
        max_lr=0.003,
        num_steps=int(10000),
        pct_start=0.001,
        div_factor=1e1,
        final_div_factor=2e1,
        weight_decay=0.0005,
        gradient_clipping=5.0,
    )

    wandb.config.update(c)

    m = initialize_model(
        model_cfg=c.m_cfg,
        load_idx=c.log_chk.load_idx,
        chkpt_folder=c.log_chk.chkpt_folder,
    )

    optim, opt_state = initialize_optim(c.opt_c, m)

    for i in range(c.opt_c.num_steps):
        start = time.time()
        l, m, opt_state, k, norm = make_step(m, opt_state, k, c.batch_size)
        end = time.time()

        log = AttrDict(loss=l.item(), norm=norm.item(), batch=i, time_it=end - start)
        wandb.log(log)
        if i % c.log_chk.print_every == 0 or i == 1:
            print(log)

        if i == 0 or i == 1:
            print("iter_time", end - start)

        if i % c.log_chk.save_params == 0:
            p = Path(c.log_chk.chkpt_folder) / f"{save_idx}.eqx"
            eqx.tree_serialise_leaves(p, m)

        if i % c.log_chk.evaluate_iters == 0:
            start = time.time()
            k, sk = split(k)
            m = eqx.tree_inference(m, value=True)
            log = evaluate(m, sk, c.eval_size)
            m = eqx.tree_inference(m, value=False)
            log = jax.tree_util.tree_map(lambda x: x.item(), log)
            end = time.time()
            log["batch"] = i
            log["time_eval"] = end - start
            wandb.log(log)
            print(log)
            
