import equinox as eqx
from jax import numpy as jnp
from jax import vmap
from jax.random import PRNGKey, split
import jax

@eqx.filter_jit
def eval_batch(model, trs, key):
    # get the shape of the first leaf of trs
    model = eqx.tree_inference(model, value=True)
    num_traces = jax.tree_leaves(trs)[0].shape[0]
    log_p = vmap(model.eval_log_p)(trs, split(key,num_traces)).mean()
    model = eqx.tree_inference(model, value=False)
    return -log_p

@eqx.filter_value_and_grad
def loss(model, trs, ks):
    log_p = vmap(model.log_p)(trs, ks)
    return -jnp.mean(log_p)

def update_model(grads, opt_state, model, optim):
  updates, opt_state = optim.update(grads, opt_state, model)
  model = eqx.apply_updates(model, updates)
  return model, opt_state

@eqx.filter_jit
def make_step(model, opt_state, key, trs, batch_size, optim):
    ks = split(key, batch_size + 1)
    l, grads = loss(model, trs, ks[:batch_size])
    model, opt_state = update_model(grads, opt_state, model, optim)
    return l, model, opt_state, ks[-1]
  
def shard_data(data, num_devices):
    devices = jax.devices()[:num_devices]
    def shard_array(x):
        x = x.reshape((num_devices, -1) + x.shape[1:])
        return jax.pmap(lambda x: x)(x)
    return jax.tree_map(shard_array, data)