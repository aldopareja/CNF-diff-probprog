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
      d_model=52,
      num_input_variables=1,
      num_enc_layers=2
      ),
    )
    self.mlp = eqx.nn.MLP(52,6,width_size=52,depth=1,key=ks[1])
    
  def __call__(self,x, key):
    assert x.ndim == 2 and x.shape[1] == 1
    enc_obs = self.encode(x,key=key)
    return self.mlp(enc_obs)

def test_encoder():
  """
  test if an encoder can sumarize noisy data to predict
  one of two possible means (0,1)
  """
  
  m = EncoderClassifier(key=PRNGKey(5543))
  
  obs_size = 100
  def sample(k,*, max_num_mixtures=6):
    """sample a bernoulli and then sample the normal based on the value of the bernoulli.
    Augment the flow with standard normals to make it easier to learn.
    """
    ks = split(k, 3)
    # b = bernoulli(ks[0], 0.5)
    b = tfd.Categorical(
        probs=jnp.ones((max_num_mixtures,)) / max_num_mixtures
    ).sample(seed=ks[0])
    s = normal(ks[1],shape=(obs_size,1))
    s /= 10.0
    s += jnp.arange(max_num_mixtures)[b]/max_num_mixtures
    return s, jnp.int32(b)
  
  # def sample_obs(means, class_label, k):
  #   return tfd.MultivariateNormalDiag(
  #     loc = means[class_label],
  #     scale_diag=jnp.ones_like(means[class_label])/10.0
  #   ).sample(seed=k)
  
  # def sample(k, *, max_num_mixtures=6, dims=2, num_obs=100):
  #   ks = split(k, 5)
  #   num_mixtures = tfd.Categorical(
  #       probs=jnp.ones((max_num_mixtures,)) / max_num_mixtures
  #   ).sample(seed=ks[0])
    
  #   means = tfd.Uniform(low=-1.0, high=1.0).sample(
  #       seed=ks[1], sample_shape=(max_num_mixtures, dims)
  #   )
    
  #   class_labels_probs = jnp.where(
  #       jnp.arange(max_num_mixtures) < num_mixtures,
  #       jnp.ones((max_num_mixtures,)),
  #       jnp.zeros((max_num_mixtures,)),
  #   )
    
  #   class_labels = tfd.Categorical(probs=class_labels_probs).sample(
  #       seed=ks[3], sample_shape=(num_obs,)
  #   )
    
  #   obs = jax.vmap(sample_obs, in_axes=(None,0,0))(
  #     means,class_labels,split(ks[4],num_obs)
  #   )
    
  #   return obs, num_mixtures

  num_steps = 1000
  optim = optax.chain(
     optax.clip_by_global_norm(5.0),
        optax.adamw(
            learning_rate=optax.cosine_onecycle_schedule(
                num_steps,
                0.025,
                0.01,
                1e0,
                1e2,
            ),
            weight_decay=0.0005,
        ),
  )
    
  batch_size = 500
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
      
  test_obs, test_b = jax.vmap(sample)(split(PRNGKey(3659),1000))
  logits = jax.vmap(m)(test_obs,split(key,test_obs.shape[0]))
  b_hat = jnp.argmax(logits,axis=1)
  print(b_hat == test_b, b_hat)
  assert (b_hat == test_b).sum() >= 0.9*len(test_b)
      
if __name__ == "__main__":
  test_encoder()