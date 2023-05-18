from collections import OrderedDict, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from itertools import repeat
from os import makedirs
import jax
from jax.random import split, PRNGKey
from numpyro.handlers import trace
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
import pickle
import numpy as np

from src.common_bijectors import make_bounding_bijector

def get_trace(args):
  m,k = args
  exec_trace = trace(m).get_trace(k)
  return exec_trace

def sample_many_traces(m, key:PRNGKey, num_traces, parallel, max_num_variables=14, default_trace=None, traces=None, use_dill=True, max_traces_for_default_trace=1000):
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
    
    if traces is None:
      if use_dill:
        import dill
        dill.Pickler.dumps, dill.Pickler.loads = dill.dumps, dill.loads
        mp.reduction.ForkingPickler = dill.Pickler
        mp.reduction.dump = dill.dump
      #model, key, total_length
      ks = split(key, num_traces)
      args = zip(repeat(m),ks)
      
      if parallel:
        with mp.Pool(mp.cpu_count()//4) as p:
          traces = list(p.imap_unordered(get_trace, tqdm(args, total=num_traces, desc='sampling traces'),chunksize=200))
      else:
        traces = list(map(get_trace, tqdm(args, total=num_traces)))
        
    traces = list(filter(lambda t: len(t) <= max_num_variables, traces))
    
    if default_trace is None:
      default_trace, metadata = build_default_trace(traces, max_traces_for_default_trace)
      
    if parallel:
      with mp.Pool(mp.cpu_count()//4) as p:
        traces = list(p.imap_unordered(get_trace_order_and_values, 
                                      tqdm(zip(traces, repeat(default_trace)),total=len(traces), desc='postprocessing traces'),
                                      chunksize=100))
    else:
      traces = list(map(get_trace_order_and_values, zip(traces, repeat(default_trace))))
    return traces, default_trace, metadata

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
  if len(shape) == 1:
    shape = (1,shape[0])
  shape = (1,1) if shape == () else shape
  assert len(shape) == 2, "a variable shape should be seq_len, num_dims"
  def_value = np.zeros(shape, dtype=v['value'].dtype)
  # is_discrete = v['value'].dtype.kind in ('i', 'u')
  return dict(value=def_value)

def build_default_trace(traces, max_traces_to_use):
  default_trace = OrderedDict()
  metadata = defaultdict(lambda: dict(values=[], bounds=dict(upper_bound=-np.inf, lower_bound=np.inf)))
  
  #accumulate values and bounds
  for i,t in tqdm(enumerate(traces)):
    for k,v in t.items():
      if k not in default_trace:
        default_trace[k] = get_necessary_variable_info(v)
      #reshape as the default trace
      metadata[k]['values'].append(v['value'].reshape(*default_trace[k]['value'].shape))
      fn = v['fn']
      if hasattr(fn.support, 'lower_bound'):
        metadata[k]['bounds']['lower_bound'] = min(metadata[k]['bounds']['lower_bound'], fn.support.lower_bound)
      if hasattr(fn.support, 'upper_bound'):
        metadata[k]['bounds']['upper_bound'] = max(metadata[k]['bounds']['upper_bound'], fn.support.upper_bound)
      
    if i > max_traces_to_use:
      break
  
  #postprocess values and compute mean and std
  #mean and std are taken after bounding, since that's what's the model sees
  for k,v in metadata.items():
    bounds = v['bounds']
    v = np.concatenate(v['values'], axis=0)
    #remove infinite bounds
    bounds = {k_:np.float32(v_) if not np.isinf(v_) else None
              for k_,v_ in bounds.items() }
        
    if default_trace[k]['value'].dtype in (np.float32, np.float64):
      v = make_bounding_bijector(**bounds).inverse(v)
      metadata[k] = {'mean': np.mean(v, axis=0, keepdims=True), 
                     'std': np.std(v, axis=0, keepdims=True),
                     **bounds}
    else:
      metadata[k] = {'mean': np.zeros((1,1), np.float32), 'std': np.ones((1,1), np.float32), 
                     **{'lower_bound': None, 'upper_bound': None}}
  
  #sort the default trace and make default indices
  default_trace = OrderedDict(sorted(default_trace.items(), key=lambda t: t[0]))
  
  current_idx = 0
  for k,v in default_trace.items():
    v['indices'] = np.arange(current_idx, current_idx + v['value'].shape[0])
    current_idx += v['value'].shape[0]
  
  return default_trace, dict(**metadata)

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

  