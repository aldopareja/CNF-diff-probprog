from itertools import repeat
from experiments.bayes3d.bayes3d import SceneSampler
from pathlib import Path
from jax.random import PRNGKey, split
import bayes3d as j
from numpyro.handlers import trace
from tqdm import tqdm
import sys
import os
import multiprocessing as mp
# import dill
# dill.Pickler.dumps, dill.Pickler.loads = dill.dumps, dill.loads
# mp.reduction.ForkingPickler = dill.Pickler
# mp.reduction.dump = dill.dump
import pickle
from jax.tree_util import tree_map
import numpy as np

def generate_traces(args):
  key_int, num_samples, outpath = args
  model = SceneSampler(mesh_paths=[Path("ycb_video_models/models/025_mug/textured_simple.obj")])
  outpath = Path(outpath)
  key = PRNGKey(key_int)
  traces = []
  for i in tqdm(range(num_samples)):
    key, sk = split(key)
    tr = trace(model).get_trace(sk)
    traces.append(tree_map(np.array,tr))
    
  #pickle the traces
  with open(outpath/f"traces_{key_int}.pkl", "wb") as f:
    pickle.dump(traces, f)

if __name__ == "__main__":
  #read an int as the first argument to this script
  key_int = int(sys.argv[1])
  num_samples_per_batch = int(sys.argv[2])
  num_processes = int(sys.argv[3])
  num_batches = int(sys.argv[4])
  outpath = sys.argv[5]
  parallel = True
  
  os.makedirs(outpath, exist_ok=True)
  
  if parallel:
    #parallelize using spawn to use cuda with multiprocessing
    mp.set_start_method('spawn')
    with mp.Pool(num_processes) as p:
      p.map(generate_traces, zip(range(key_int, key_int+num_batches), repeat(num_samples_per_batch), repeat(outpath)))
  else:
    for i in range(num_batches):
      generate_traces((key_int+i, num_samples_per_batch, outpath))