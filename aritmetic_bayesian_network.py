from typing import List

from jax.random import split, PRNGKey
from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import equinox as eqx

from cnf import CNF



def bayesian_network(k:PRNGKey):
  # sampled the model 10000 times to find this
  means = [4.998,-1.976,0.091,-9.892,7.012,-0.312, -9.893, -0.311]
  stds = [1.401,1.410,0.764,7.846,1.993,0.891,7.847,0.897]
  sks = split(k,10)
  z0 = tfd.Laplace(5,1).sample(seed=sks[0])
  z1 = tfd.Laplace(-2,1).sample(seed=sks[1])
  z2 = tfd.Normal(jnp.tanh(z0 + z1 - 2.8),0.1).sample(seed=sks[2])
  z3 = tfd.Normal(z0 * z1,0.1).sample(seed=sks[3])
  z4 = tfd.Normal(7,2).sample(seed=sks[4])
  z5 = tfd.Normal(jnp.tanh(z3 + z4),0.1).sample(seed=sks[5])
  x0 = tfd.Normal(z3,0.1).sample(seed=sks[6])
  x1 = tfd.Normal(z5,0.1).sample(seed=sks[7])
  s = [((v-means[i])/stds[i])[None] for i,v in enumerate([z0, z1, z2, z3, z4, z5, x0, x1])]
  return s

class InferenceForBayesianNetwork(eqx.Module):
  flows: List[CNF]
  
  def __init__(self,*,key, **kwargs):
    ks = split(key,6)
    flows = []
    for i in range(6):
      flows.append(
        CNF(
          num_latents=1,
          num_augments=9 if not "num_augments" in kwargs else kwargs["num_augments"],
          num_conds = 2 + i, #cond on x0 and x1 plus previous sampled vars
          width_size=128 if not "width_size" in kwargs else kwargs["width_size"],
          num_blocks=1 if not "num_blocks" in kwargs else kwargs["num_blocks"],
          depth=3 if not "depth" in kwargs else kwargs["depth"],
          key=ks[i]
        )
      )
    self.flows = flows
    
  def log_p(self,z0, z1, z2, z3, z4, z5, x0, x1,key):
    ks = split(key,6)
    
    log_p = 0.0
    so_far_added = []
    for i,v in enumerate([z0,z1,z2,z3,z4,z5]):
      log_p += self.flows[i].log_p(
        v,
        jnp.concatenate([x0,x1]+so_far_added),
        ks[i]
      )
      so_far_added+=[v]
    return log_p
  
  def rsample(self, x0,x1,key):
    ks = split(key, 6)
    s = []
    for i in range(6):
      new_s = self.flows[i].rsample(
        ks[i],
        jnp.concatenate([x0,x1] + s)
      )
      s.append(new_s)
      
    return s

if __name__ == "__main__":
  q_z_x = InferenceForBayesianNetwork(key = PRNGKey(0))
  
  z0, z1, z2, z3, z4, z5, x0, x1 = bayesian_network(PRNGKey(0))
  
  q_z_x.log_p(z0, z1, z2, z3, z4, z5, x0, x1,PRNGKey(0))
  q_z_x.rsample(x0,x1,PRNGKey(0))