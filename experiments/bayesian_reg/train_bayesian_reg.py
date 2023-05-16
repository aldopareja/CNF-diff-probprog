import os
from pathlib import Path
from time import time
from jax.random import PRNGKey, split
from jax import vmap

from tensorflow_probability.substrates import jax as tfp

import src.InferenceModel
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
  
  means_and_stds = load_traces("experiments/bayesian_reg/means_and_stds.pkl")
  means_and_stds = dict_to_namedtuple(means_and_stds)
  
  c = src.InferenceModel.InferenceModelCfg(
    d_model = 128,
    dropout_rate = 0.1,
    discrete_mlp_width = 512,
    discrete_mlp_depth=1,
    continuous_flow_blocks=8,
    continuous_flow_num_layers_per_block=2,
    continuous_flow_num_augment=91,
    num_enc_layers=5,
    max_discrete_choices =6,
    num_input_variables = (1,),
    num_observations =6,
    means_and_stds=means_and_stds,
  )
  inference = src.InferenceModel.InferenceModel(key=PRNGKey(0),c=c)
  
  # inference = eqx.tree_deserialise_leaves("tmp/1MM_blr_00005_eval_last_2_norm.eqx", inference)
  
  ########### DEBUG ###############
  
  # normal = vmap(inference.log_p)(eval_traces_batch, split(PRNGKey(0),1000))
  # eval = eval_batch(inference, eval_traces_batch, PRNGKey(0))
  # from IPython import embed; embed()
  
  num_steps = 10000
  optim = optax.chain(
      optax.clip_by_global_norm(5.0),
      optax.adamw(
          learning_rate=optax.cosine_onecycle_schedule(
              num_steps,
              0.0003,
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
  for i in tqdm(range(num_steps), desc="1MM_blr_00005_last_1_norm_mlp_no_aug_div1000"):
      start = time()
      batch_traces = sample_random_batch(traces, batch_size)
      l, inference, opt_state, key = make_step(inference, opt_state, key, batch_traces, batch_size, optim)
      end = time()
      if i % 100 == 0 or i == 1:
        logger.info(f"{l.item()} t {end-start}")
        # print("l", l, "t", end - start)
        #save model to dummy file
        p = out_path / f"1MM_blr_00005_last_1_norm_mlp_no_aug_div1000.eqx"
        eqx.tree_serialise_leaves(p, inference)

        key, sk = split(key)
        start = time()
        eval_log_p = eval_batch(inference, eval_traces_batch, sk)
        end = time()
        if eval_log_p < best_eval:
          logger.info(f"new best {eval_log_p}, took {end-start}")
          best_eval = eval_log_p
          p = out_path / f"1MM_blr_00005_last_1_norm_mlp_no_aug_div1000_best.eqx"
          eqx.tree_serialise_leaves(p, inference)
  