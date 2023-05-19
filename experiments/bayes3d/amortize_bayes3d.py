import os
from pathlib import Path
from time import time
from jax.random import PRNGKey, split
from jax.tree_util import tree_map

import equinox as eqx
import numpy as np
import optax
from tqdm import tqdm

import src.InferenceModel
from src.utils.common_training_functions import eval_batch, evaluate_per_batch, make_step, sample_random_batch, BatchSampler
from src.utils.trace_dataset import load_traces, serialize_traces
from src.utils.setup_logger import setup_logger
from src.utils.miscellaneous import dict_to_namedtuple

import logging
logger = setup_logger(__name__, level=logging.INFO)


if __name__ == "__main__":
  traces = load_traces("tmp/1M_bayes3d.pkl")
  traces, test_traces = traces[:-2000], traces[-2000:]
  
  ######DEBUG########
  # traces = load_traces("tmp/1k_bayes3d.pkl")
  # test_traces = traces
  ########
  
  
  variable_metadata = load_traces("tmp/1M_bayes3d_metadata.pkl")
  variable_metadata = dict_to_namedtuple(variable_metadata)
  
  c = src.InferenceModel.InferenceModelCfg(
    variable_metadata=variable_metadata,
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
  inference = src.InferenceModel.InferenceModel(key=PRNGKey(0),c=c)
  
  #########Debug ##########
  # debug_sampler = BatchSampler(traces, 1000)
  # batch = next(debug_sampler)
  # obs = batch['trace']['obs']['value']
  # bound_and_standardize = lambda val: inference.bound_and_standardize('obs', val)
  # bound_and_standardize(batch['trace']['obs']['value'][0])
  # from jax import vmap
  # value, inv_log_det_jacobian, is_discrete = vmap(bound_and_standardize)(obs)
  
  # inference = eqx.tree_deserialise_leaves("tmp/500k_bayes3d.eqx", inference)

  num_steps = 100000
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
  
  batch_size = 128
  eval_batch_size = 2000
  train_sampler = BatchSampler(traces, batch_size)
  
  ######## DEBUG ########
  # inference.log_p(traces[0], PRNGKey(0))
  
  opt_state = optim.init(eqx.filter(inference, eqx.is_inexact_array))
  
  key = PRNGKey(573)
  out_path = Path("tmp/")
  os.makedirs(out_path, exist_ok=True)
  
  init_time = time()
  best_eval = np.inf
  for i,batch_traces in tqdm(zip(range(num_steps), train_sampler), desc="1M_bayes3d_b128",total= num_steps):
      start = time()
      # batch_traces = sample_random_batch(traces, batch_size)
      l, inference, opt_state, key = make_step(inference, opt_state, key, batch_traces, batch_size, optim)
      end = time()
      if i % 100 == 0 or i == 1:
        logger.info(f"{l.item():.4f} t {end-start:.4f}")
        # print("l", l, "t", end - start)
        #save model to dummy file
        p = out_path / f"1M_bayes3d_b128.eqx"
        eqx.tree_serialise_leaves(p, inference)
        key, sk = split(key)
        start = time()
        eval_log_p = evaluate_per_batch(inference, test_traces, eval_batch_size, sk)
        logger.info(f"{str(i).zfill(6)} eval {eval_log_p:.4f}")
        end = time()
        if eval_log_p < best_eval:
          logger.info(f"{str(i).zfill(6)} new best {eval_log_p:.4f}, took {end-start:.2f}, after {(time()-init_time)/60:.2f} min")
          best_eval = eval_log_p
          p = out_path / f"1M_bayes3d_b128_best.eqx"
          eqx.tree_serialise_leaves(p, inference)
  
  