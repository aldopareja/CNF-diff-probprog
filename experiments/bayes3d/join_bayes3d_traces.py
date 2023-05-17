from pathlib import Path
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures import as_completed
# mp.set_start_method('spawn')
import multiprocessing as mp
from src.utils.trace_dataset import sample_many_traces, serialize_traces
import numpy as np
from jax import numpy as jnp
import jax
from jax.random import PRNGKey, split 
from itertools import chain

def load_trace(p):
  return pickle.load(open(p, "rb"))
  
num_traces = 1000000
traces = []
raw_traces_folder = "tmp/bayes3d_traces_mug"
key = jax.random.PRNGKey(56385)
parallel = True
num_files = int(num_traces//200)

if parallel:
  sks = split(key, num_files)
  all_paths = list(Path(raw_traces_folder).glob("*.pkl"))
  all_paths = all_paths[:num_files]
  sks = sks[:len(all_paths)]
  #use a threading pool to load traces in parallel, not multiprocess since we just need to load the traces
  # with mp.Pool(mp.cpu_count()//4) as p:
  #   traces = list(chain.from_iterable(p.imap_unordered(load_trace, tqdm(all_paths, desc='loading raw traces', total=len(all_paths)), chunksize=30)))
  # with ProcessPoolExecutor(max_workers=mp.cpu_count()//4) as executor:
  with ThreadPoolExecutor(max_workers=30) as executor:
    futures = {executor.submit(load_trace, path) for path in all_paths}
    traces = list(chain.from_iterable([future.result() for future in tqdm(as_completed(futures), desc='loading raw traces', total=len(all_paths))]))
else:
  #go over each .pkl file in raw traces folder and load it
  for i,p in tqdm(enumerate(Path(raw_traces_folder).glob("*.pkl")), desc='loading raw traces'):
    key, sk = jax.random.split(key)
    tr = pickle.load(open(p, "rb"))
    traces+=tr
    if len(traces) >= num_traces:
      break
from IPython import embed; embed(using=False)
traces, default_trace, mean_and_std = sample_many_traces(None, None, num_traces, True, max_num_variables=60, traces=traces, use_dill=False, max_traces_for_default_trace=10000)
#pickle dataset
serialize_traces(traces, "tmp/500k_bayes3d.pkl")
serialize_traces(mean_and_std, "tmp/500k_bayes3d_mean_and_std.pkl")
from IPython import embed; embed(using=False)