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
  key, num_samples, outpath, i, key_int = args
  model = SceneSampler(
    mesh_paths=[Path("ycb_video_models/models/025_mug/textured_simple.obj")],
    num_objects=1,
    max_pose_xy_noise=0.3,
    )
  outpath = Path(outpath)
  traces = []
  for _ in range(num_samples):
    key, sk = split(key)
    tr = trace(model).get_trace(sk)
    traces.append(tree_map(np.array,tr))
  
  #pickle the traces
  with open(outpath/f"traces_{key_int}_{str(i).zfill(6)}.pkl", "wb") as f:
    pickle.dump(traces, f)

if __name__ == "__main__":
  #read an int as the first argument to this script
  key_int = int(sys.argv[1])
  key = PRNGKey(key_int)
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
      list(p.imap_unordered(generate_traces, tqdm(zip(split(key, num_batches), 
                                      repeat(num_samples_per_batch), 
                                      repeat(outpath), 
                                      range(num_batches), 
                                      repeat(key_int)),
                                  desc="Generating Traces",
                                  total=num_batches)))
  else:
      [generate_traces(a) for a in zip(split(key, num_batches), repeat(num_samples_per_batch), repeat(outpath), range(num_batches), repeat(key_int))]