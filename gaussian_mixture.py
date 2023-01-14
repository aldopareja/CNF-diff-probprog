from typing import List

import jax
from jax import numpy as jnp
from jax.random import PRNGKey, split
from jax import vmap

import equinox as eqx
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

from encoder import Encoder, EncoderCfg
from cnf import CNF


def build_cov_matrices(t, dims, max_num_mixtures, eps=1e-5):
    """
    if you get a lower triangular matrix L then you can ensure a positive
    definite covariance matrix by computing:

    cov = LL' + eps*I

    see https://stackoverflow.com/questions/40573478/ensuring-positive-definite-covariance-matrix
    """
    # t = jnp.concatenate([t, t[:, dims:][:, ::-1]], axis=-1)
    # t = t.reshape((max_num_mixtures, dims, dims))
    # t = jnp.triu(t)
    t = tfp.math.fill_triangular(t)

    eps_I = jnp.eye(dims)[None] * eps
    cov_matrices = jnp.matmul(t, t.swapaxes(-2, -1)) + eps_I
    return cov_matrices


def sample_observations(means, cov_matrices, class_label, k):
    # s = tfd.MultivariateNormalTriL(
    #     loc=means[class_label], scale_tril=cov_matrices[class_label]
    # ).sample(seed=k)
    s = tfd.MultivariateNormalDiag(
        loc=means[class_label],
        scale_diag=jnp.ones_like(means[class_label])/50.0
        ).sample(seed=k)
    return s


def gaussian_mixture(k: PRNGKey, *, max_num_mixtures=6, dims=2, num_obs=100):
    ks = split(k, 5)
    # num_mixtures = tfd.Categorical(
    #     probs=jnp.ones((max_num_mixtures,)) / max_num_mixtures
    # ).sample(seed=ks[0])
    # num_mixtures = max_num_mixtures - 1
    num_mixtures = 0

    means = tfd.Uniform(low=-1.0, high=1.0).sample(
        seed=ks[1], sample_shape=(max_num_mixtures, dims)
    )

    cov_terms = tfd.Uniform(low=0.01, high=1.0).sample(
        seed=ks[2], sample_shape=(max_num_mixtures, int(dims * (dims + 1) / 2))
    )
    cov_matrices = build_cov_matrices(cov_terms, dims, max_num_mixtures)

    class_labels_probs = jnp.where(
        jnp.arange(max_num_mixtures) <= num_mixtures,
        jnp.ones((max_num_mixtures,)),
        jnp.zeros((max_num_mixtures,)),
    )/(num_mixtures + 1)

    # order to break symmetry
    mean_norms = jnp.linalg.norm(means, axis=-1) * class_labels_probs
    idx = jnp.flip(jnp.argsort(mean_norms))

    means = means[idx]
    cov_matrices = cov_matrices[idx]
    cov_terms = cov_terms[idx]

    class_labels = tfd.Categorical(probs=class_labels_probs).sample(
        seed=ks[3], sample_shape=(num_obs,)
    )

    obs = vmap(sample_observations, in_axes=(None, None, 0, 0))(
        means, cov_matrices, class_labels, split(ks[4], num_obs)
    )

    return num_mixtures, means, cov_terms, class_labels, obs


class InferenceGaussianMixture(eqx.Module):
    obs_encoder: Encoder
    means_flow: CNF
    cov_flow: CNF
    num_mixtures_est: eqx.nn.MLP
    dims: eqx.static_field()
    max_num_mixtures: eqx.static_field()
    d_model: eqx.static_field()

    def __init__(
        self,
        *,
        key,
        max_num_mixtures=6,
        dims=2,
        d_model=128,
        dropout_rate=0.1,
        num_mixtures_mlp_width=128,
        num_mixtures_mlp_depth=2,
        flows_depth=3,
        flows_num_augment=30,
        num_enc_layers=4,
    ):
        ks = split(key, 10)

        self.obs_encoder = Encoder(
            key=ks[0],
            c=EncoderCfg(
                num_heads=4,
                dropout_rate=dropout_rate,
                d_model=d_model,
                num_input_variables=dims,
                num_enc_layers=num_enc_layers,
            ),
        )

        self.num_mixtures_est = eqx.nn.MLP(
            in_size=d_model,
            out_size=max_num_mixtures,
            width_size=num_mixtures_mlp_width,
            depth=num_mixtures_mlp_depth,
            key=ks[1],
        )

        self.means_flow = CNF(
            num_latents=max_num_mixtures * dims,
            num_augments=flows_num_augment,
            num_conds=d_model + int(d_model/2),  # the encoded observations plus K repeated d_model/2 times
            width_size=d_model * 2,
            depth=flows_depth,
            key=ks[2],
            num_blocks=1,
        )

        self.cov_flow = CNF(
            num_latents=int(max_num_mixtures * dims * (dims + 1) / 2),
            num_augments=flows_num_augment,
            num_conds=d_model + int(d_model/2) + max_num_mixtures * dims,
            width_size=d_model * 2,
            depth=flows_depth,
            key=ks[3],
            num_blocks=1,
        )

        self.dims = dims
        self.max_num_mixtures = max_num_mixtures
        self.d_model = d_model

    def log_p(self, num_mixtures, means, cov_terms, obs, key):
        ks = split(key, 3)
        assert obs.ndim == 2 and num_mixtures.dtype == jnp.int32 and num_mixtures.ndim==0

        encoded_obs = self.obs_encoder(obs, key=ks[2])
        assert encoded_obs.ndim == 1

        mlp_ = self.num_mixtures_est(encoded_obs)
        assert mlp_.ndim == 1
    
        num_mixtures_log_p = mlp_[num_mixtures] - jax.nn.logsumexp(mlp_)
        
        # return 30*num_mixtures_log_p
        
        conds = jnp.concatenate(
            [encoded_obs, 
             jnp.repeat(jnp.float32(num_mixtures)/5.0 - 0.5, int(self.d_model/2))]
        )
        means_log_p = self.means_flow.log_p(
            z=means.reshape(-1), cond_vars=conds, key=ks[0]
        )
        
        return 30*num_mixtures_log_p + means_log_p

        conds = jnp.concatenate(
            [
                conds,
                means.reshape(-1),
            ]
        )

        covs_log_p = self.cov_flow.log_p(
            z=cov_terms.reshape(-1), cond_vars=conds, key=ks[1]
        )
        #added a 30 multiplier to make the losses on the same order of magnitude
        return 30*num_mixtures_log_p + means_log_p + covs_log_p

    def rsample(self, obs, key):
        ks = split(key, 4)

        encoded_obs = self.obs_encoder(obs, key=ks[0])

        num_mixtures = tfd.Categorical(
            logits=self.num_mixtures_est(encoded_obs)
        ).sample(seed=ks[1])
        
        # return num_mixtures, None, None

        conds = jnp.concatenate(
            [encoded_obs, 
             jnp.repeat(jnp.float32(num_mixtures)/5.0 - 0.5, int(self.d_model/2))]
        )

        means = self.means_flow.rsample(ks[2], conds)

        # conds = jnp.concatenate([conds, means])

        # cov_terms = self.cov_flow.rsample(ks[3], conds)

        means = means.reshape(self.max_num_mixtures, self.dims)
        # cov_terms = cov_terms.reshape(
        #     self.max_num_mixtures, int(self.dims * (self.dims + 1) / 2)
        # )
        return num_mixtures, means, None
        # return num_mixtures, means, cov_terms


if __name__ == "__main__":
    num_mixtures, means, cov_terms, class_labels, obs = vmap(gaussian_mixture)(
        split(PRNGKey(0), 2)
    )
    m = InferenceGaussianMixture(key=PRNGKey(1))
    vmap(m.log_p)(num_mixtures, means, cov_terms, obs, split(PRNGKey(2), 2))
