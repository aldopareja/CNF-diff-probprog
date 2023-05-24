import os
from pathlib import Path
from time import time
from jax.random import PRNGKey, split
from jax import tree_map, vmap
from jax import numpy as jnp
import numpy as np

from tensorflow_probability.substrates import jax as tfp

from src.diffusion_head import DiffusionHead, DiffusionConf
from src.InferenceModel import InferenceModel, InferenceModelCfg
from src.gaussian_mixture_head import GaussianMixture, GaussianMixtureCfg
from src.real_nvp import RealNVP_Flow, RealNVPConfig
tfb = tfp.bijectors

import equinox as eqx
import optax
from tqdm import tqdm

from src.utils.common_training_functions import make_step, eval_batch, sample_random_batch
from src.utils.trace_dataset import load_traces
from src.utils.setup_logger import setup_logger
from src.utils.miscellaneous import dict_to_namedtuple

import logging
logger = setup_logger(__name__, level=logging.INFO)


if __name__ == "__main__":
  traces = load_traces("experiments/bayesian_reg/data/train_1MM.pkl")
  # traces = load_traces("experiments/bayesian_reg/traces.pkl")
  eval_traces = load_traces("experiments/bayesian_reg/data/test_1MM.pkl")
  eval_traces_batch = sample_random_batch(eval_traces, len(eval_traces))
  
  
  variable_metadata = load_traces("experiments/bayesian_regression/data/metadata.pkl")
  variable_metadata = tree_map(lambda x: jnp.array(x, dtype=np.float32), variable_metadata)
  variable_metadata = dict_to_namedtuple(variable_metadata)
  
  continuous_distribution = DiffusionHead(
    c=DiffusionConf(
      num_latents=1,
      width_size=512,
      depth=8,
      num_conds=128,
      num_steps=100,
      noise_scale=0.008,
      dropout_rate=0.1,
      use_normalizer=True,
    ),
    key=PRNGKey(13),
  )
  
  # continuous_distribution = RealNVP_Flow(
  #   c = RealNVPConfig(
  #     num_latents=1,
  #     num_blocks=8,
  #     num_layers_per_block=2,
  #     block_hidden_size=256,
  #     num_conds=128,
  #     normalizer_width=512,
  #   ),
  #   key=PRNGKey(13),
  # )
  # gmc = GaussianMixtureCfg(
  #   resnet_mlp_width=512,
  #   d_model=256,
  #   num_mlp_blocks=3,
  #   num_mixtures=3,
  #   dropout_rate=0.0,
  # )
  
  # continuous_distribution = GaussianMixture(c=gmc, key=PRNGKey(13))
  
  inference = InferenceModel(
    key=PRNGKey(0),
    c=InferenceModelCfg(
        variable_metadata=variable_metadata,
        d_model = 128,
        dropout_rate = 0.0,
        discrete_mlp_width = 512,
        discrete_mlp_depth=1,
        num_enc_layers=4,
        max_discrete_choices =6,
        num_input_variables = (1,),
        num_observations =6,
      ),
    continuous_distribution=continuous_distribution
    )
  
  # inference = eqx.tree_deserialise_leaves("tmp/blr_10k_0005_gmm_no_norm.eqx", inference)
  
  ########### DEBUG ###############
  # inference.log_p(tree_map(jnp.array,traces[0]), key=PRNGKey(0))
  # normal = vmap(inference.log_p)(eval_traces_batch, split(PRNGKey(0),1000))
  # eval = eval_batch(inference, eval_traces_batch, PRNGKey(0))
  # from IPython import embed; embed()
  
  num_steps = 10000
  optim = optax.chain(
      optax.clip_by_global_norm(5.0),
      optax.adamw(
          learning_rate=optax.cosine_onecycle_schedule(
              num_steps,
              0.0005,
              0.01,
              1e1,
              1e2,
          ),
          weight_decay=0.0005,
      ),
  )
  
  batch_size = 1024
  
  opt_state = optim.init(eqx.filter(inference, eqx.is_inexact_array))
  
  key = PRNGKey(573)
  out_path = Path("tmp/")
  best_eval = float("inf")
  os.makedirs(out_path, exist_ok=True)
  for i in tqdm(range(num_steps), desc="blr_10k_0005_diff_8layers"):
      start = time()
      batch_traces = sample_random_batch(traces, batch_size)
      l, inference, opt_state, key = make_step(inference, opt_state, key, batch_traces, batch_size, optim)
      end = time()
      if i % 100 == 0 or i == 1:
        logger.info(f"{l.item()} t {end-start}")
        # print("l", l, "t", end - start)
        #save model to dummy file
        p = out_path / f"blr_10k_0005_diff_8layers.eqx"
        eqx.tree_serialise_leaves(p, inference)

        key, sk = split(key)
        start = time()
        eval_log_p = eval_batch(inference, eval_traces_batch, sk)
        end = time()
        if eval_log_p < best_eval:
          logger.info(f"new best {eval_log_p}, took {end-start}")
          best_eval = eval_log_p
          p = out_path / f"blr_10k_0005_diff_8layers_best.eqx"
          eqx.tree_serialise_leaves(p, inference)
  