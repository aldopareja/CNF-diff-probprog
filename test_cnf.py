import optax
import jax
from jax.random import PRNGKey, split, bernoulli, normal
from jax import numpy as jnp
import equinox as eqx
from cnf import CNF
import time


def test_CNF():
    """
    test if the conditional CNF can learn to produce
    a normal with mean 0.5,-0.5 depending on a bernoulli
    """
    model = CNF(
        num_latents=1,
        num_augments=9,
        num_conds=1,
        width_size=128,
        num_blocks=1,
        depth=3,
        key=PRNGKey(0),
    )

    def sample(k):
        """sample a bernoulli and then sample the normal based on the value of the bernoulli.
        Augment the flow with standard normals to make it easier to learn.
        """
        ks = split(k, 3)
        b = bernoulli(ks[0], 0.5)
        s = normal(ks[1])
        s /= 10.0
        s += jax.lax.cond(b, lambda: 0.5, lambda: -0.5)
        # s = jnp.concatenate([s[None], normal(ks[2],shape=(9,))])
        return s[None], b[None]
        # return s[None], b[None]

    num_steps = 250
    optim = optax.chain(
        optax.clip_by_global_norm(5.0),
        optax.adamw(
            learning_rate=optax.cosine_onecycle_schedule(
                num_steps,
                0.025,
                0.01,
                1e0,
                1e2,
            ),
            weight_decay=0.0005,
        ),
    )
    batch_size = 1000
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_value_and_grad
    def loss(model, s_batch, b_batch, key):
        log_p = jax.vmap(model.log_p)(s_batch, b_batch, split(key, b_batch.shape[0]))
        return -jnp.mean(log_p)

    @eqx.filter_jit
    def make_step(model, opt_state, key):
        ks = split(key, batch_size + 2)
        s_batch, b_batch = jax.vmap(sample)(ks[:-2])
        l, grads = loss(model, s_batch, b_batch, ks[-2])
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return l, model, opt_state, ks[-1]

    key = PRNGKey(573)
    for i in range(num_steps):
        start = time.time()
        l, model, opt_state, key = make_step(model, opt_state, key)
        end = time.time()
        if i % 20 == 0 or i == 1:
            print("l", l, "t", end - start)
        if i % 50 == 0:
            pos_sample = jax.vmap(model.rsample)(split(key, 1000), jnp.ones((1000, 1)))
            print(
                "pos_sample_mean",
                pos_sample[:, 0].mean(),
                pos_sample[:, 0].std(),
                pos_sample.shape,
            )

            neg_sample = jax.vmap(model.rsample)(split(key, 1000), jnp.zeros((1000, 1)))
            print("neg_sample_mean", neg_sample[:, 0].mean(), neg_sample[:, 0].std())

    pos_sample = jax.vmap(model.rsample)(split(key, 1000), jnp.ones((1000, 1)))
    print(
        "pos_sample_mean",
        pos_sample[:, 0].mean(),
        pos_sample[:, 0].std(),
        pos_sample.shape,
    )

    neg_sample = jax.vmap(model.rsample)(split(key, 1000), jnp.zeros((1000, 1)))
    print("neg_sample_mean", neg_sample[:, 0].mean(), neg_sample[:, 0].std())

    assert jnp.isclose(pos_sample[:, 0].mean(), 0.5, rtol=0.1)
    assert jnp.isclose(neg_sample[:, 0].mean(), -0.5, rtol=0.1)
    assert jnp.isclose(pos_sample[:, 0].std(), 0.1, rtol=0.1)
    assert jnp.isclose(neg_sample[:, 0].std(), 0.1, rtol=0.1)


if __name__ == "__main__":
    test_CNF()
