from collections import defaultdict
from itertools import repeat
from os import makedirs
from jax.random import split, PRNGKey
from numpyro.handlers import trace
from multiprocessing import Pool, cpu_count
from jax import numpy as jnp
from tqdm import tqdm
import equinox as eqx
from pathlib import Path
import pickle
import numpy as np
from jax import tree_util as jtu

def get_trace(args):
  m,k = args
  exec_trace = trace(m).get_trace(k)
  return exec_trace

def get_sample(args):
  #TODO: this doesn't work for multivalued variables, e.g. the obsevation from a mulitvariate gaussian, which is a 2d matrix with num_samples x num_dimensions
  #need to fix this
  *args, total_length = args
  exec_trace = get_trace(args)
  
  result = defaultdict(list)
  for k,v in exec_trace.items():
    value = v['value']
    assert value.ndim == 0, "value is not a scalar"
    if k == 'obs':
      is_discrete = value.dtype.kind in ('i', 'u')
    else:
      is_discrete = v['fn'].is_discrete
    
    result['values'].append(jnp.float32(value))
    result['is_discrete'].append(jnp.array(is_discrete))
    result['is_obs'].append(jnp.array(k == 'obs'))
    
  #move observation to the first position
  obs_idx = result['is_obs'].index(True)
  for k,v in result.items():
    result[k] = [v[obs_idx]] + v[:obs_idx] + v[obs_idx+1:]
  
  result = {k:jnp.stack(v) for k,v in result.items()}
  
  result['attention_mask'] = jnp.bool_(jnp.ones_like(result['values']))
  
  #padd to the same length
  arr_len = len(result['values'])
  if total_length is not None:
    for k,v in result.items():
      result[k] = jnp.pad(v, (0, total_length-arr_len))
  
  return jtu.tree_map(np.array,result)
  

def sample_many_traces(m, key:PRNGKey, num_traces, parallel):
  '''
    samples many traces from a model and process the results into a list of dicts. Use this function to get a dataset to train an amortized inference model.
    better to set CUDA_VISIBLE_DEVICES="" to avoid problems with multiprocessing.
    
    Args:
      m: numpyro model function that takes a PRNGKey as input.
      key: PRNGKey
      num_traces: int
      parallel: bool
    Returns:
      traces: list of dicts each with keys 'values' and 'is_discrete'. Values contain the values of the samples and is_discrete is a boolean array indicating if the variable is discrete or not.
  '''
  ks = split(key, num_traces)
  
  #model, key, total_length
  args = zip(repeat(m),ks, repeat(7))
  
  if parallel:
    with Pool(cpu_count()//4) as p:
      traces = list(p.imap_unordered(get_sample, tqdm(args, total=num_traces),chunksize=200))
  else:
    traces = list(map(get_sample, tqdm(args, total=num_traces)))
  
  return traces

#serialize traces to disk with pickle
def serialize_traces(traces, path):
  p = Path(path)
  makedirs(p, exist_ok=True)
  pickle.dump(traces, open(p / f"dataset.pkl", "wb"))
  
#load traces
def load_traces(path):
  p = Path(path) / f"dataset.pkl"
  traces = pickle.load(open(p, "rb"))
  return traces

def sample_random_batch(traces, num_traces):
  '''
    samples a random batch of traces from a list of traces.
    returns a stacked single pytree
  '''
  idx = np.random.choice(len(traces), num_traces)
  batch = jtu.tree_map(lambda *ts: np.stack(ts), *[traces[i] for i in idx])
  return jtu.tree_map(jnp.array, batch)
  