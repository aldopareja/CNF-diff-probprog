from dataclasses import dataclass
from typing import List

import math

from jax import vmap
from jax import numpy as jnp
from jax.random import split, PRNGKey, uniform
import equinox as eqx
from jaxtyping import Array

@dataclass
class TransformerCfg:
  num_heads: int = 4
  dropout_rate: float = 0.1
  d_model: int = 52
  num_input_variables: int = 2
  num_enc_layers: int = 2
  
  
class EncoderLayer(eqx.Module):
  multihead_Attention: eqx.nn.MultiheadAttention
  layer_norms: List[eqx.nn.LayerNorm]
  mlp: eqx.nn.MLP
  dropout: eqx.nn.Dropout
  
  def __init__(self, *, key, c:TransformerCfg):
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
    
  def __call__(self,x,*,key):
    assert x.ndim == 2 and x.shape[1] == self.layer_norms[0].shape
    ks = split(key,3)
    
    x_ = self.multihead_Attention(x,x,x,key=ks[0])
    
    x_ = self.dropout(x_,key=ks[2])
    x = x + x_
    x = vmap(self.layer_norms[0])(x)
    
    x = vmap(self.mlp)(x)
    
    x_ = self.dropout(x,key=ks[1])
    x = x + x_
    x = vmap(self.layer_norms[1])(x)
    return x
    
class Encoder(eqx.Module):
  obs_to_embed: eqx.nn.Linear
  enc_token: Array
  enc_layers: List[EncoderLayer]
  
  def __init__(self, *, key:PRNGKey, c:TransformerCfg):
    ks = split(key,10)
    self.obs_to_embed = eqx.nn.Linear(
      in_features=c.num_input_variables, 
      out_features=c.d_model,
      key = ks[0]
      )
    
    lim = 1 / math.sqrt(c.d_model)
    self.enc_token = uniform(ks[1], (1,c.d_model), minval=-lim, maxval=lim)
    
    self.enc_layers = [EncoderLayer(key=ks[2],c=c) for _ in range(c.num_enc_layers)]
    
  def __call__(self,x,*,key):
    assert x.ndim == 2
    ks = split(key, len(self.enc_layers))
    x = vmap(self.obs_to_embed)(x)
    x = jnp.concatenate([x,self.enc_token])
    
    for i,enc in enumerate(self.enc_layers):
      x = enc(x,key=ks[i])
      
    return x[-1]
    
if __name__ == "__main__":
  m = Encoder(key=PRNGKey(0), c=TransformerCfg())
  x = m(jnp.ones((100,2)),key=PRNGKey(1))
  print(x,x.shape)