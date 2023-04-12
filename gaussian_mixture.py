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
from real_nvp import RealNVP_Flow


def build_cov_matrices(t, dims, eps=1e-5):
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
    # cov_matrices = jnp.matmul(t, t.swapaxes(-2, -1)) + eps_I
    return t + eps_I


def sample_observations(means, cov_matrices, class_label, k):
    s = tfd.MultivariateNormalTriL(
        loc=means[class_label], scale_tril=cov_matrices[class_label]
    ).sample(seed=k)
    # s = tfd.MultivariateNormalDiag(
    #     loc=means[class_label],
    #     scale_diag=jnp.ones_like(means[class_label])/50.0
    #     ).sample(seed=k)
    return s

def gaussian_mixture(k: PRNGKey, *, max_num_mixtures=6, dims=2, num_obs=200):
    ks = split(k, 5)
    num_mixtures = tfd.Categorical(
        probs=jnp.ones((max_num_mixtures,)) / max_num_mixtures
    ).sample(seed=ks[0])
    # num_mixtures = tfd.Categorical(
    #     probs=jnp.ones((3,)) / 3
    # ).sample(seed=ks[0])
    # num_mixtures = 3

    means = tfd.Uniform(low=-1.0, high=1.0).sample(
        seed=ks[1], sample_shape=(max_num_mixtures, dims)
    )

    cov_terms = tfd.Uniform(low=-0.2, high=0.2).sample(
        seed=ks[2], sample_shape=(max_num_mixtures, int(dims * (dims + 1) / 2))
    )
    # cov_terms = jnp.array([[1.0,0.0,1.0],]*6)/10.0
    cov_matrices = build_cov_matrices(cov_terms, dims)

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

def gaussian_mixture_log_p_single_obs(observation, means,cov_terms, num_mixtures, max_num_mixtures=6):
    assert observation.ndim == 1
    assert means.ndim == 2 and means.shape[1] == observation.shape[0] and means.shape[0] == max_num_mixtures
    assert cov_terms.ndim == 2 and cov_terms.shape[0] == max_num_mixtures
    cov_matrices = build_cov_matrices(cov_terms, 2)
    assert cov_matrices.ndim == 3 and cov_matrices.shape[2] == observation.shape[0]
    
    normals_log_p = tfd.MultivariateNormalTriL(
        loc=means, scale_tril=cov_matrices
    ).log_prob(observation)
    
    # num_mix_mask = jnp.where(
    #     jnp.arange(max_num_mixtures) <= num_mixtures,
    #     jnp.zeros((max_num_mixtures,)),
    #     jnp.stack([-jnp.inf]*max_num_mixtures),
    #     # jnp.zeros((max_num_mixtures,)),
    # )
    
    # assert normals_log_p.shape == num_mix_mask.shape
    
    # log_p = jax.nn.logsumexp(normals_log_p + num_mix_mask)
    
    log_p = jnp.where(jnp.arange(max_num_mixtures)<=num_mixtures,
                      normals_log_p,
                      0.0).sum()
    
    return log_p

def gaussian_mixture_log_p(many_obs, means, cov_terms, num_mixtures, max_num_mixtures=6):
    assert many_obs.ndim == 2
    return (vmap(gaussian_mixture_log_p_single_obs, in_axes=(0,None,None,None,None))(
        many_obs, means, cov_terms, num_mixtures, max_num_mixtures)
            ).mean()

    

class InferenceGaussianMixture(eqx.Module):
    obs_encoder: Encoder
    flow: RealNVP_Flow
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
        flows_num_blocks=8,
        flows_num_layers_per_block=1,
        flows_num_augment=90,
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

        self.flow = RealNVP_Flow(
            num_blocks = flows_num_blocks,
            num_layers_per_block= flows_num_layers_per_block,
            block_hidden_size=d_model,
            num_augments=flows_num_augment,
            num_latents = max_num_mixtures * dims + int(max_num_mixtures * dims * (dims + 1) / 2),
            num_conds = d_model + int(d_model/2),
            key = ks[2]
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
        
        conds = jnp.concatenate(
            [encoded_obs, 
             jnp.repeat(jnp.float32(num_mixtures)/5.0 - 0.5, int(self.d_model/2))]
        )
        
        z = jnp.concatenate([means.reshape(-1), cov_terms.reshape(-1)])
        
        gmm_log_p = self.flow.log_p(
            z=z, cond_vars=conds, key=ks[0]
        )
        
        return num_mixtures_log_p #+ gmm_log_p

    def rsample(self, obs, key):
        ks = split(key, 4)

        encoded_obs = self.obs_encoder(obs, key=ks[0])

        num_mixtures = jax.lax.stop_gradient(
            tfd.Categorical(
                logits=self.num_mixtures_est(encoded_obs)
                ).sample(seed=ks[1])
        )

        conds = jnp.concatenate(
            [encoded_obs, 
             jnp.repeat(jnp.float32(num_mixtures)/5.0 - 0.5, int(self.d_model/2))]
        )

        gmm_params = self.flow.rsample(ks[2], conds)
        
        means = gmm_params[:self.max_num_mixtures * self.dims]
        cov_terms = gmm_params[self.max_num_mixtures * self.dims:]

        means = means.reshape(self.max_num_mixtures, self.dims)
        cov_terms = cov_terms.reshape(
            self.max_num_mixtures, int(self.dims * (self.dims + 1) / 2)
        )
        return num_mixtures, means, cov_terms


if __name__ == "__main__":
    num_mixtures, means, cov_terms, class_labels, obs = vmap(gaussian_mixture)(
        split(PRNGKey(0), 2)
    )
    m = InferenceGaussianMixture(key=PRNGKey(1))
    vmap(m.log_p)(num_mixtures, means, cov_terms, obs, split(PRNGKey(2), 2))
