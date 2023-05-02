from collections import OrderedDict, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from itertools import repeat
from os import makedirs
import jax
from jax.random import split, PRNGKey
from numpyro.handlers import trace
import multiprocessing as mp
import dill
dill.Pickler.dumps, dill.Pickler.loads = dill.dumps, dill.loads
mp.reduction.ForkingPickler = dill.Pickler
mp.reduction.dump = dill.dump
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

def sample_many_traces(m, key:PRNGKey, num_traces, parallel, max_num_variables=14, default_trace=None):
  '''
    samples many traces from a model and process the results into a list of dicts. Use this function to get a dataset to train an amortized inference model.
    better to set CUDA_VISIBLE_DEVICES="" to avoid problems with multiprocessing.
    
    Args:
      m: numpyro model function that takes a PRNGKey as input.
      key: PRNGKey
      num_traces: int
      parallel: bool
      get_fn: either get_trace or get_sample
    Returns:
      traces: list of dicts each with keys 'values' and 'is_discrete'. Values contain the values of the samples and is_discrete is a boolean array indicating if the variable is discrete or not.
  '''
  with jax.default_device(jax.devices('cpu')[0]):
    
    #model, key, total_length
    ks = split(key, num_traces)
    args = zip(repeat(m),ks)
    
    if parallel:
      with mp.Pool(mp.cpu_count()//4) as p:
        traces = list(p.imap_unordered(get_trace, tqdm(args, total=num_traces, desc='sampling traces'),chunksize=100))
    else:
      traces = list(map(get_trace, tqdm(args, total=num_traces)))
      
    traces = list(filter(lambda t: len(t) <= max_num_variables, traces))
    
    if default_trace is None:
      default_trace = build_default_trace(traces)
      
    if parallel:
      with mp.Pool(mp.cpu_count()//4) as p:
        traces = list(p.imap_unordered(get_trace_order_and_values, 
                                      tqdm(zip(traces, repeat(default_trace)),total=len(traces), desc='postprocessing traces'),
                                      chunksize=100))
    else:
      traces = list(map(get_trace_order_and_values, zip(traces, repeat(default_trace))))
    return traces, default_trace

def get_trace_order_and_values(args):
  ''' process a trace to make it compatible with the default trace.
      This is necessary because batching pytrees requires that all pytrees have the same structure and shape in the leafs.
      The observation gets moved to the first position as shown by the indices.
      variables present in the trace are added sequentially also as shown by the indices.
      the values of the default trace are replaced by the values of the trace (only for present variables).
      
      Each address should only be sampled once and and should have the same shape regardless of the model's draw.
    args: (trace, default_trace)
    returns 
  '''
  trace, default_trace = args
  p_trace = deepcopy(default_trace)
  
  #add present variables indices and values starting with the observation
  indices = [p_trace['obs']['indices']]
  
  #add other variables except the observation
  for k,v in trace.items():
    p_trace[k]['value'][:] = v['value']
    if k != 'obs':
      indices.append(p_trace[k]['indices'])
  indices = [np.concatenate(indices)]
  attention_mask = np.ones(len(indices[0]), dtype=bool)
  
  #add not present variables indices
  for k in p_trace.keys():
    if k not in trace:
      indices.append(p_trace[k]['indices'])
    p_trace[k].pop('indices')
    
  indices = np.concatenate(indices)
  attention_mask = np.concatenate([attention_mask, 
                                   np.zeros(len(indices) - len(attention_mask), 
                                            dtype=bool)])
  return dict(trace=p_trace, indices=indices, attention_mask=attention_mask)
    

def get_necessary_variable_info(v):
  shape = v['value'].shape
  shape = (1,1) if shape == () else shape
  assert len(shape) == 2, "a variable shape should be seq_len, num_dims"
  def_value = np.zeros(shape, dtype=v['value'].dtype)
  # is_discrete = v['value'].dtype.kind in ('i', 'u')
  return dict(value=def_value)

def build_default_trace(traces):
  default_trace = OrderedDict()
  for t in traces:
    for k,v in t.items():
      if k not in default_trace:
        default_trace[k] = get_necessary_variable_info(v)
  default_trace = OrderedDict(sorted(default_trace.items(), key=lambda t: t[0]))
  
  current_idx = 0
  for k,v in default_trace.items():
    v['indices'] = np.arange(current_idx, current_idx + v['value'].shape[0])
    current_idx += v['value'].shape[0]
  return default_trace

#serialize traces to disk with pickle
def serialize_traces(traces, path="tmp/dummy_data.pkl"):
  p = Path(path)
  makedirs(p.parent, exist_ok=True)
  pickle.dump(traces, open(p, "wb"))
  
#load traces
def load_traces(path="tmp/dummy_data.pkl"):
  p = Path(path)
  #extract the base path without the file name (only the directory)
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
  