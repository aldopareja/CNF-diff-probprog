import time

import jax
from jax.random import split, PRNGKey, bernoulli, normal
from jax import numpy as jnp

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import optax

import equinox as eqx

from encoder import Encoder, EncoderCfg

class EncoderClassifier(eqx.Module):
  encode: Encoder
  mlp: eqx.nn.MLP
  
  def __init__(self,key):
    ks = split(key)
    self.encode = Encoder(
    key=PRNGKey(0),
    c=EncoderCfg(
      num_heads=4,
      dropout_rate=0.1,
      d_model=128,
      num_input_variables=1,
      num_enc_layers=5,
      ),
    )
    self.mlp = eqx.nn.MLP(128,6,width_size=128,depth=1,key=ks[1])
    
  def __call__(self,x, key):
    assert x.ndim == 2 and x.shape[1] ==1
    enc_obs = self.encode(x,key=key)
    return self.mlp(enc_obs)

def test_encoder():
  """
  test if an encoder can sumarize noisy data to predict
  one of two possible means (0,1)
  """
  
  m = EncoderClassifier(key=PRNGKey(5543))
  
  obs_size = 150
  def sample(k,*, max_num_mixtures=6, dims=1):
    """sample a bernoulli and then sample the normal based on the value of the bernoulli.
    Augment the flow with standard normals to make it easier to learn.
    """
    ks = split(k, 4)
    
    b = tfd.Categorical(
        probs=jnp.ones((max_num_mixtures,)) / max_num_mixtures
    ).sample(seed=ks[0])
    
    class_labels_probs = jnp.where(
      jnp.arange(max_num_mixtures) <= b,
      jnp.ones((max_num_mixtures,)),
      jnp.zeros((max_num_mixtures,)),
    )

    class_labels = tfd.Categorical(probs=class_labels_probs).sample(
        seed=ks[3], sample_shape=(obs_size,)
    )
    
    s = normal(ks[1],shape=(obs_size,dims))
    s /= 100.0
    
    # mean = (jnp.stack([jnp.arange(max_num_mixtures)]*dims,axis=1)/max_num_mixtures)[class_labels].reshape(-1,dims)
    mean = tfd.Uniform(low=-1.0, high=1.0).sample(
        seed=ks[2], sample_shape=(max_num_mixtures, dims)
    )[class_labels].reshape(-1,dims)
    s += mean
    return s, jnp.int32(b)

  num_steps = 20000
  optim = optax.chain(
     optax.clip_by_global_norm(5.0),
        optax.adamw(
            learning_rate=optax.cosine_onecycle_schedule(
                num_steps,
                0.001,
                0.01,
                1e1,
                3e2,
            ),
            weight_decay=0.0005,
        ),
  )
    
  batch_size = 250
  opt_state = optim.init(eqx.filter(m, eqx.is_inexact_array))
  
  @eqx.filter_value_and_grad
  def loss(m, obs, b, key):
    assert obs.ndim == 3 and obs.shape[2]==1 and jnp.dtype(b) == jnp.int32 and b.ndim == 1
    logits = jax.vmap(m)(obs,split(key,obs.shape[0]))
    @jax.vmap
    def log_p(logits,b):
      assert b.ndim == 0 and logits.ndim == 1
      return logits[b] - jax.nn.logsumexp(logits)
    return -log_p(logits,b).mean()
  
  @eqx.filter_jit
  def make_step(model, opt_state, key):
    ks = split(key, batch_size + 2)
    obs_batch, b_batch = jax.vmap(sample)(ks[:-2])
    l, grads  = loss(model, obs_batch, b_batch, ks[-2])
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return l, model, opt_state, ks[-1]
  
  key = PRNGKey(574)
  for i in range(num_steps):
    start = time.time()
    l, m, opt_state, key = make_step(m, opt_state, key)
    end = time.time()
    
    if i % 20 == 0 or i == 1:
      print("l", l, "t", end - start)
      
  m = eqx.tree_inference(m, value=True)
  test_obs, test_b = jax.vmap(sample)(split(PRNGKey(3659),1000))
  logits = jax.vmap(m)(test_obs,split(key,test_obs.shape[0]))
  b_hat = jnp.argmax(logits,axis=1)
  print(b_hat == test_b, b_hat)
  assert (b_hat == test_b).sum() >= 0.9*len(test_b)
      
if __name__ == "__main__":
  test_encoder()