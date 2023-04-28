from dataclasses import dataclass
from typing import List

import math

import jax
from jax import vmap, jit
from jax import numpy as jnp
from jax.random import split, PRNGKey, uniform
import equinox as eqx
from jaxtyping import Array

@dataclass
class EncoderCfg:
  num_heads: int = 4
  dropout_rate: float = 0.1
  d_model: int = 52
  num_input_variables: int = 2
  num_enc_layers: int = 2
  max_latents: int = 100
  num_observations: int = 100
  
  
class EncoderLayer(eqx.Module):
  multihead_Attention: eqx.nn.MultiheadAttention
  layer_norms: List[eqx.nn.LayerNorm]
  mlp: eqx.nn.MLP
  dropout: eqx.nn.Dropout
  num_heads: eqx.static_field()
  
  
  def __init__(self, *, key, c:EncoderCfg):
    ks = split(key, 3)
    self.multihead_Attention = eqx.nn.MultiheadAttention(
      num_heads = c.num_heads,
      query_size = c.d_model,
      dropout_p= c.dropout_rate,
      key=ks[0]
    )
    self.mlp = eqx.nn.MLP(
      in_size=c.d_model,
      out_size=c.d_model,
      width_size=c.d_model*2,
      depth=1, #1 hidden layer
      key=ks[1]
    )
    self.layer_norms = [eqx.nn.LayerNorm(c.d_model,) for _ in range(2)]
    self.dropout = eqx.nn.Dropout(p=c.dropout_rate)
    self.num_heads = c.num_heads
    
    
  def __call__(self,x,*,key, mask):
    assert x.ndim == 2 and x.shape[1] == self.layer_norms[0].shape
    assert mask.ndim == 1 and mask.shape[0] == x.shape[0]
    ks = split(key,3)
    
    mask_mha =  jnp.broadcast_to(mask,(self.num_heads,x.shape[0],x.shape[0]))
    x_ = self.multihead_Attention(x,x,x,mask=mask_mha,key=ks[0])
    
    x_ = self.dropout(x_,key=ks[2])
    
    # x = vmap(jnp.multiply)(x,mask) + x_
    x = x + x_
    x = vmap(self.layer_norms[0])(x)
    
    x = vmap(self.mlp)(x)
    
    x_ = self.dropout(x,key=ks[1])
    x = x + x_
    x = vmap(self.layer_norms[1])(x)
    return x
    
class Encoder(eqx.Module):
  obs_to_embed: eqx.nn.Linear
  enc_layers: List[EncoderLayer]
  latent_input_embeddings: Array
  num_input_vars: eqx.static_field()
  num_observations: eqx.static_field()
  
  def __init__(self, *, key:PRNGKey, c:EncoderCfg):
    ks = split(key,10)
    self.obs_to_embed = eqx.nn.Linear(
      in_features=c.num_input_variables, 
      out_features=c.d_model,
      key = ks[0]
      )
    
    lim = 1 / math.sqrt(c.d_model)
    self.latent_input_embeddings = uniform(ks[1], (c.max_latents, c.d_model), minval=-lim, maxval=lim)
    
    self.enc_layers = [EncoderLayer(key=ks[2],c=c) for _ in range(c.num_enc_layers)]
    self.num_input_vars = c.num_input_variables
    self.num_observations = c.num_observations
    
  @staticmethod
  @eqx.filter_jit
  def positional_encoding(num_tokens, d_model):
    # Initialize a matrix with shape (num_tokens, d_model)
    pos_enc = jnp.zeros((num_tokens, d_model))

    # Calculate positional encoding for each token
    for pos in range(num_tokens):
        for i in range(0, d_model, 2):
            pos_enc = pos_enc.at[pos, i].set(jnp.sin(pos / jnp.power(10000, (2 * i) / d_model)))
            pos_enc = pos_enc.at[pos, i + 1].set(jnp.cos(pos / jnp.power(10000, (2 * (i + 1)) / d_model)))

    return pos_enc
  
  def __call__(self,x,*,key, mask):
    assert x.ndim == 2 and x.shape[1] == self.num_input_vars
    assert mask.dtype == jnp.bool_
    obs = vmap(self.obs_to_embed)(x[:self.num_observations])
    latents = vmap(self.obs_to_embed)(x[self.num_observations:])
    x = jnp.concatenate([obs, latents, self.latent_input_embeddings[x.shape[0]][None]])
    
    x += self.positional_encoding(x.shape[0],x.shape[1])
    
    ks = split(key, len(self.enc_layers))
    for i,enc in enumerate(self.enc_layers):
      x = enc(x,key=ks[i],mask=mask)
      
    return x
    
if __name__ == "__main__":
  m = Encoder(key=PRNGKey(0), c=EncoderCfg())
  x = m(jnp.ones((100,2)),key=PRNGKey(1))
  print(x,x.shape)