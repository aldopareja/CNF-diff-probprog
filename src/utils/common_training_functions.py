import equinox as eqx
from jax import numpy as jnp, tree_util as jtu
from jax import vmap
from jax.random import PRNGKey, split
import jax
from jaxtyping import PyTree
import numpy as np
from tqdm import tqdm

@eqx.filter_jit
def eval_batch(model, trs, key):
    # get the shape of the first leaf of trs
    model = eqx.tree_inference(model, value=True)
    num_traces = jax.tree_leaves(trs)[0].shape[0]
    log_p = vmap(model.eval_log_p)(trs, split(key,num_traces)).mean()
    model = eqx.tree_inference(model, value=False)
    return -log_p

def evaluate_per_batch(model, test_traces, eval_batch_size, key):
    eval_sampler = BatchSampler(test_traces, eval_batch_size, infinite=False)
    model = eqx.tree_inference(model, value=True)
    log_p = 0.0
    ks = split(key, len(eval_sampler))
    for i,trs in enumerate(eval_sampler):
        log_p += eval_batch(model, trs, ks[i])
    return log_p/len(eval_sampler)

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


def sample_random_batch(traces, num_traces):
  '''
    samples a random batch of traces from a list of traces.
    returns a stacked single pytree
  '''
  idx = np.random.choice(len(traces), size=num_traces, replace=False)
  batch = jtu.tree_map(lambda *ts: np.stack(ts), *[traces[j] for j in idx])
  return jtu.tree_map(jnp.array, batch)


class BatchSampler:
    def __init__(self, traces, num_traces, infinite=True):
        self.traces = traces
        self.num_traces = num_traces
        self.infinite = infinite
        self.idx = np.arange(len(traces)) if not infinite else np.random.permutation(len(traces))
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        '''
            samples a random batch of traces from a list of traces.
            returns a stacked single pytree
        '''
        if self.current_index + self.num_traces > len(self.traces):
            if self.infinite:
                self.idx = np.random.permutation(len(self.traces))
                self.current_index = 0
            else:
                raise StopIteration

        batch = jtu.tree_map(lambda *ts: np.stack(ts), *[self.traces[j] for j in self.idx[self.current_index:self.current_index+self.num_traces]])
        self.current_index += self.num_traces

        return jtu.tree_map(jnp.array, batch)
    
    def __len__(self):
        return len(self.traces) // self.num_traces