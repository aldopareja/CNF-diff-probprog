from functools import partial
from os import mkdir
import os
from pathlib import Path
import time
import optax
from tqdm import tqdm
from src import GP_kernels as gpk
from jax.random import PRNGKey, split
from jax import jit, numpy as jnp
from jax import vmap
from numpyro.handlers import trace
from numpyro import distributions as dist
import numpyro as npy
import equinox as eqx
import jax
from src.utils.trace_dataset import load_traces, sample_random_batch



def test_sample_kernel():
  key = PRNGKey(0)
  kernel = gpk.sample_kernel(key)
  #this key samples a quadratic kernel with lenght_scale=0.4261 and scale_mixture=0.5156
  
  samples = jnp.arange(100, dtype=jnp.float32)/10 #range from 0 to 10
  
  manual_kernel = gpk.RationalQuadraticKernel(lenght_scale=0.4261, scale_mixture=0.5156)

  cov_kernel = vmap(vmap(kernel, in_axes=(None, 0)), in_axes=(0, None))(samples, samples)
  cov_manual_kernel = vmap(vmap(manual_kernel, in_axes=(None, 0)), in_axes=(0, None))(samples, samples)
  
  assert jnp.allclose(cov_kernel, cov_manual_kernel, atol=1e-4), "sampled kernel is not equal to the manual kernel"
  
def test_model():
  key = PRNGKey(0)
  t = trace(gpk.model).get_trace(key)
  assert 'obs' in t, "model does not sample obs"
  
  test = jnp.stack([gpk.model(k) for k in split(key, 100)])
  assert not jnp.any(jnp.isnan(test)), "model returns nan"
  
def test_GP_Inference():
  @partial(jit, static_argnums=(0,))
  def body_fn(num_steps, key):
    def update_fn(curr, key):
      probs_even = jnp.array([0.75, 0.25])
      probs_odd = jnp.array([0.25, 0.75])
      new_curr = jax.lax.cond(
          curr % 2 == 0,
          lambda: dist.Categorical(probs=probs_even).sample(key),
          lambda: dist.Categorical(probs=probs_odd).sample(key),
      )
      return new_curr, new_curr

    ks = split(key, num_steps + 1)
    curr = dist.Categorical(probs=jnp.array([0.5, 0.5])).sample(ks[0])
    _, values = jax.lax.scan(update_fn, curr, ks[1:])
    total = values.sum()
    return total, values
    
  def sampler(key):
    '''
    samples a number of steps, if previous sample was odd then add have higher probabability of adding odd
    else add have higher probability of adding even.
    '''
    ks = split(key, 7)
    n = npy.sample("num", dist.Categorical(probs=jnp.array([0.2, 0.2, 0.2, 0.2, 0.2])), rng_key=ks[0])
      
    _, values = body_fn(n.item()+1, ks[1])
    total = 0
    for i,v in enumerate(values):
      # if jnp.any(jnp.isclose(v, jnp.arange(5), atol=1e-7)):
      #   d = dist.Delta(v)
      # else:
      #   d = dist.Normal(v, 0.001)      
      total+=npy.sample(f"step_{i}", dist.Uniform(v,v+1), rng_key=ks[i+2])
    
    npy.deterministic("obs", total)
    return total

  model = gpk.GPInference(key=PRNGKey(0), c=gpk.GPInferenceCfg(num_input_variables=1,
                                                               num_observations=1,
                                                               max_discrete_choices=5))
  # check if the model has been saved already and load it
  if os.path.exists("tmp/dummy.eqx") and False:
    model = eqx.tree_deserialise_leaves(Path("tmp/dummy.eqx"), model)
  else:
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
    batch_size = 400
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    traces = load_traces("tmp/")

    @eqx.filter_value_and_grad
    def loss(model, trs, ks):
        log_p = vmap(model.log_p)(trs, ks)
        return -jnp.mean(log_p)

    def update_model(grads, opt_state, model):
      updates, opt_state = optim.update(grads, opt_state, model)
      model = eqx.apply_updates(model, updates)
      return model, opt_state
    
    @eqx.filter_jit
    def make_step(model, opt_state, key, trs):
        ks = split(key, batch_size + 1)
        l, grads = loss(model, trs, ks[:batch_size])
        model, opt_state = update_model(grads, opt_state, model)
        return l, model, opt_state, ks[-1]
    
    key = PRNGKey(573)
    out_path = Path("tmp/")
    os.makedirs(out_path, exist_ok=True)
    for i in tqdm(range(num_steps)):
        start = time.time()
        batch_traces = sample_random_batch(traces, batch_size)
        l, model, opt_state, key = make_step(model, opt_state, key, batch_traces)
        end = time.time()
        if i % 100 == 0 or i == 1:
            print("l", l, "t", end - start)
            #save model to dummy file
            p = out_path / f"dummy.eqx"
            eqx.tree_serialise_leaves(p, model)
  
  
  
if __name__ == '__main__':
  test_GP_Inference()
  test_model()
  test_sample_kernel()