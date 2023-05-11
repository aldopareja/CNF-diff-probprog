import os
from pathlib import Path
from time import time
from jax.random import PRNGKey, split
from jax.tree_util import tree_map

import equinox as eqx
import numpy as np
import optax
from tqdm import tqdm

from src import GP_kernels as gpk
from src.utils.common_training_functions import make_step, shard_data
from src.utils.trace_dataset import load_traces, sample_random_batch, serialize_traces
from src.utils.setup_logger import setup_logger
from src.utils.miscellaneous import dict_to_namedtuple

import logging
logger = setup_logger(__name__, level=logging.INFO)


if __name__ == "__main__":
  traces = load_traces("tmp/500k_bayes3d.pkl")
  means_and_stds = load_traces("tmp/500k_bayes3d_mean_and_std.pkl")
  means_and_stds = dict_to_namedtuple(means_and_stds)
  
  c = gpk.GPInferenceCfg(
    means_and_stds = means_and_stds,
    d_model = 128,
    dropout_rate = 0.1,
    discrete_mlp_width = 512,
    discrete_mlp_depth=1,
    continuous_flow_blocks=8,
    continuous_flow_num_layers_per_block=2,
    continuous_flow_num_augment=91,
    num_enc_layers=5,
    max_discrete_choices =6,
    num_input_variables = (225,1),
    num_observations =100,
  )
  inference = gpk.GPInference(key=PRNGKey(0),c=c)

  num_steps = 100000
  optim = optax.chain(
      optax.clip_by_global_norm(5.0),
      optax.adamw(
          learning_rate=optax.cosine_onecycle_schedule(
              num_steps,
              0.0001,
              0.01,
              1e1,
              1e2,
          ),
          weight_decay=0.0005,
      ),
  )
  
  batch_size = 32
  
  opt_state = optim.init(eqx.filter(inference, eqx.is_inexact_array))
  
  key = PRNGKey(573)
  out_path = Path("tmp/")
  os.makedirs(out_path, exist_ok=True)
  for i in tqdm(range(num_steps), desc="500k_bayes3d"):
      start = time()
      batch_traces = sample_random_batch(traces, batch_size)
      l, inference, opt_state, key = make_step(inference, opt_state, key, batch_traces, batch_size, optim)
      end = time()
      if i % 100 == 0 or i == 1:
        logger.info(f"{l.item()} t {end-start}")
        # print("l", l, "t", end - start)
        #save model to dummy file
        p = out_path / f"500k_bayes3d.eqx"
        eqx.tree_serialise_leaves(p, inference)
  
  