import equinox as eqx
import jax
from jax import lax, numpy as jnp, vmap
from jax.random import PRNGKey, split
from numpyro.handlers import substitute, trace

from typing import Dict, NamedTuple, Tuple
from tensorflow_probability.substrates import jax as tfp

from src.common_bijectors import make_bounding_and_standardization_bijector
tfb = tfp.bijectors
tfd = tfp.distributions

from dataclasses import dataclass
from src.encoder import Encoder, EncoderCfg
from src.real_nvp import RealNVP_Flow


@dataclass
class InferenceModelCfg:
  variable_metadata: NamedTuple = None
  d_model:int = 256
  dropout_rate:float = 0.1
  discrete_mlp_width:int = 512
  discrete_mlp_depth:int=4
  continuous_flow_blocks:int=8
  continuous_flow_num_layers_per_block:int=2
  continuous_flow_num_augment:int=91
  num_enc_layers:int=4
  max_discrete_choices:int =5
  num_input_variables:Tuple[int] = (1,2)
  num_observations:int =100


class InferenceModel(eqx.Module):
  input_encoder: Encoder
  continuous_flow_dist: RealNVP_Flow
  discrete_mlp_dist: eqx.nn.MLP
  obs_to_embed_dict: Dict[str,eqx.nn.MLP]
  num_input_variables: Tuple[int]
  num_observations: int
  variable_metadata: eqx.static_field() = eqx.static_field()

  def __init__(
    self,
    *,
    key: PRNGKey,
    c: InferenceModelCfg = InferenceModelCfg(),
  ):
    ks = split(key, 5)

    ec = EncoderCfg(num_enc_layers=c.num_enc_layers,
                    d_model=c.d_model,
                    dropout_rate=c.dropout_rate,)
    self.input_encoder = Encoder(
      key=ks[0],
      c=ec,
    )

    self.discrete_mlp_dist = eqx.nn.MLP(
      in_size = c.d_model,
      out_size = c.max_discrete_choices,
      width_size= c.discrete_mlp_width,
      depth= c.discrete_mlp_depth,
      key = ks[1]
    )

    self.continuous_flow_dist = RealNVP_Flow(
      num_blocks=c.continuous_flow_blocks,
      num_layers_per_block=c.continuous_flow_num_layers_per_block,
      block_hidden_size=c.d_model,
      num_augments=c.continuous_flow_num_augment,
      num_latents = 1,
      num_conds = c.d_model,
      normalizer_width = c.discrete_mlp_depth,
      key = ks[2],
    )

    self.obs_to_embed_dict = {i: eqx.nn.Linear(in_features=i,
                                          out_features=c.d_model,
                                          key=k)
                              for i,k in zip(c.num_input_variables,
                                             split(ks[3], len(c.num_input_variables)))}

    self.num_input_variables = c.num_input_variables
    self.num_observations = c.num_observations
    self.variable_metadata = c.variable_metadata

  def log_p(self, t, key):
    key, sk = split(key, 2)
    embs, is_discrete, outputs, output_log_det_jacobian = self.process_order_and_get_transformer_embs(t, sk)

    ######## DEBUGGING ########
    # log_p_ = self.get_causal_log_p(*map(lambda x: x[0], embs, is_discrete, outputs, output_log_det_jacobian, split(key, len(is_discrete))))

    all_log_p = vmap(self.get_causal_log_p)(embs, is_discrete, outputs, output_log_det_jacobian, split(key, len(is_discrete)))
    return all_log_p.sum()

  def eval_log_p(self, t, key):
    key, sk = split(key, 2)
    embs, is_discrete, outputs, output_log_det_jacobian = self.process_order_and_get_transformer_embs(t, sk)

    eval_log_p = vmap(self.get_causal_eval_log_p)(embs, is_discrete, outputs, output_log_det_jacobian, split(key, len(is_discrete)))
    return eval_log_p.sum()
  
  def bound_and_standardize(self, var_name, value):
    '''
    return the transformed value and the inverse log det jacobian of the original value if it is continuous
    otherwise returns the original value and 0
    '''
    metadata_k = getattr(self.variable_metadata, var_name)
    is_discrete = jax.lax.cond(value.dtype == jnp.int32, lambda: True, lambda: False)
    value = jnp.float32(value)
    
    bijector = make_bounding_and_standardization_bijector(metadata_k)
    if var_name != 'obs':
      inv_log_det_jacobian = jax.lax.cond(is_discrete, lambda: jnp.zeros((1,1)), lambda: bijector.inverse_log_det_jacobian(value))
    else:
      inv_log_det_jacobian = jnp.zeros((1,1))
    value = jax.lax.cond(is_discrete, lambda: value, lambda: bijector.inverse(value))
    return value, inv_log_det_jacobian, is_discrete

  def process_order_and_get_transformer_embs(self, t, key):
    mask, indices, variables = t['attention_mask'], t['indices'], t['trace']

    enc_input = []
    outputs = []
    is_discrete = []
    output_log_det_jacobian = []

    for k,v in variables.items():
      v = v['value']
      assert len(v.shape) == 2 and v.shape[1] in self.num_input_variables
      float_v, log_det_i, is_discrete_i = self.bound_and_standardize(k, v)
      
      input = vmap(self.obs_to_embed_dict[v.shape[1]])(float_v)
      enc_input.append(input)

      if k != 'obs':
        assert v.shape[1] == 1, 'only single dimensional latents are supported, consider flattening multivariate samples'
        is_discrete.append(is_discrete_i)
        output_log_det_jacobian.append(log_det_i)
        outputs.append(float_v)
      else:
        outputs.append(jnp.zeros((self.num_observations, 1)))
        output_log_det_jacobian.append(jnp.zeros((self.num_observations, 1)))
        is_discrete+=[False]*self.num_observations

    #order the data according to the order of the trace
    enc_input = jnp.concatenate(enc_input, axis=0)[indices]
    enc_input += self.positional_encoding(*enc_input.shape)

    outputs = jnp.concatenate(outputs, axis=0)[indices]
    output_log_det_jacobian = jnp.concatenate(output_log_det_jacobian, axis=0)[indices]
    is_discrete = jnp.stack(is_discrete)[indices]

    #remove the observation part of the output since it is not used for training.
    outputs = outputs[self.num_observations:]
    is_discrete = is_discrete[self.num_observations:]
    output_log_det_jacobian = output_log_det_jacobian[self.num_observations:].reshape(-1)

    key, sk = split(key)
    embs = self.get_causal_embs(enc_input, mask, sk)

    return embs, is_discrete, outputs, output_log_det_jacobian

  @staticmethod
  def get_causal_mask(mask, i):
    '''returns a mask that is true for all indices before i but no more than mask.sum() indices
    '''
    return jnp.where(jnp.arange(mask.shape[0]) < i, True, False)*mask

  @eqx.filter_jit
  def get_causal_log_p(self, emb, is_discrete_i, output_i, inverse_log_det_jac_i, key):
    log_p = jax.lax.cond(
      is_discrete_i,
      #round to avoid floating point errors and cast to int for indexing
      lambda: self.discrete_log_prob(emb, jnp.int32(jnp.round(output_i[0]))),
      lambda: self.continuous_log_prob(emb, jnp.float32(output_i), key, init_logp=inverse_log_det_jac_i),
    )
    return log_p

  @eqx.filter_jit
  def get_causal_eval_log_p(self, emb, is_discrete_i, output_i, inverse_log_det_jac_i, key):
    log_p = jax.lax.cond(
      is_discrete_i,
      #round to avoid floating point errors and cast to int for indexing
      lambda: self.discrete_log_prob(emb, jnp.int32(jnp.round(output_i[0]))),
      lambda: self.continuous_eval_log_prob(emb, jnp.float32(output_i), key, init_logp=inverse_log_det_jac_i),
    )

    return log_p

  @eqx.filter_jit
  def get_causal_embs(self, enc_input, mask, key):
    ks = split(key, enc_input.shape[0])
    masks = vmap(self.get_causal_mask, in_axes=(None,0))(mask, jnp.arange(self.num_observations,enc_input.shape[0]))
    enc_input_ = jnp.broadcast_to(enc_input, (enc_input.shape[0]-self.num_observations, *enc_input.shape))
    get_emb = lambda enc_input_, mask_, k, i: self.input_encoder(enc_input_, mask=mask_, key=k)[i-1]
    embs = vmap(get_emb)(enc_input_, masks, ks[self.num_observations:], i=jnp.arange(self.num_observations, enc_input.shape[0]))
    return embs

  def discrete_log_prob(self, emb, value):
    assert len(emb.shape) == 1
    assert value.shape == ()
    logits = self.discrete_mlp_dist(emb)
    assert logits.ndim == 1
    log_p = logits[value] - jax.nn.logsumexp(logits)
    return log_p

  def continuous_eval_log_prob(self, emb, value, key, init_logp=0.0):
    return self.continuous_flow_dist.eval_log_p(z=value, cond_vars=emb, key=key, init_logp=init_logp)

  def continuous_log_prob(self, emb, value, key, init_logp=0.0):
    return self.continuous_flow_dist.log_p(z=value, cond_vars=emb, key=key, init_logp=init_logp)

  @staticmethod
  @eqx.filter_jit
  def positional_encoding(num_tokens, d_model):
    def inner_loop_fn(carry, i, pos, d_model):
      sin_val = jnp.sin(pos / jnp.power(10000, (2 * i) / d_model))
      cos_val = jnp.cos(pos / jnp.power(10000, (2 * (i + 1)) / d_model))
      carry = carry.at[i].set(sin_val)
      carry = carry.at[i + 1].set(cos_val)
      return carry, None
    def outer_loop_fn(carry, pos, d_model):
      init_carry = carry[pos]
      i_vals = jnp.arange(0, d_model, 2)
      final_carry, _ = lax.scan(lambda c, i: inner_loop_fn(c, i, pos, d_model), init_carry, i_vals)
      carry = carry.at[pos].set(final_carry)
      return carry, None
    pos_enc = jnp.zeros((num_tokens, d_model))
    pos_vals = jnp.arange(num_tokens)
    pos_enc, _ = lax.scan(lambda c, pos: outer_loop_fn(c, pos, d_model), pos_enc, pos_vals)
    return pos_enc

  @staticmethod
  def get_next_latent(exec_trace, sampled_latents):
    for k,v in exec_trace.items():
      if k not in sampled_latents:
        return k,v
    return None, None

  @eqx.filter_jit
  def add_new_variable_to_sequence(self, new_variable, *, enc_input=None, key=None, name=None):
    #handle the case where the new variable is a scalar
    if new_variable.shape == ():
      new_variable = new_variable[None][None]

    float_v, _, is_discrete_i = self.bound_and_standardize(name, new_variable)
    input = vmap(self.obs_to_embed_dict[new_variable.shape[1]])(float_v)

    if enc_input is None:
      enc_input = jnp.zeros((0, input.shape[1]))
      
    enc_input = jnp.concatenate([enc_input, input])
    enc_input_ = enc_input + self.positional_encoding(*enc_input.shape)
    emb = self.input_encoder(enc_input_, mask=jnp.ones(len(enc_input_),dtype=jnp.bool_),key=key)[-1]
    return emb, enc_input

  @eqx.filter_jit
  def sample_continuous(self, emb, *, key, name):
    key, sk = split(key, 2)
    z = self.continuous_flow_dist.rsample(key=sk, cond_vars=emb)
    
    metadata_k = getattr(self.variable_metadata, name)
    bijector = make_bounding_and_standardization_bijector(metadata_k)
    
    z_out = bijector.forward(z).squeeze()
    init_log_p = bijector.inverse_log_det_jacobian(z_out)

    log_p = self.continuous_eval_log_prob(emb, z, key, init_logp=init_log_p)    
    return z_out, log_p


  def rsample(self, obs, gen_model_sampler,key):
    # assert obs.shape == () #TODO: this only works for unary observations, which is fine, but not necessarily
    key, sk1, sk2 = split(key,3)
    emb, enc_input = self.add_new_variable_to_sequence(obs, key=sk1, name='obs')

    exec_trace = trace(gen_model_sampler).get_trace(sk2)
    sampled_latents = {}
    log_p = 0.0
    while True:
      k,v = self.get_next_latent(exec_trace, sampled_latents)
      if k is None or k=='obs':
        break

      key, sk1, sk2, sk3 = split(key, 4)
      is_discrete = v['value'].dtype in (jnp.int32, jnp.int64)
      if is_discrete:
        logits = self.discrete_mlp_dist(emb)
        sample = tfd.Categorical(logits=logits).sample(seed=sk1)
        log_p_ = logits[sample] - jax.nn.logsumexp(logits)
      else:
        sample, log_p_ = self.sample_continuous(emb, key=sk1, name=k)
      
      log_p += log_p_

      emb, enc_input = self.add_new_variable_to_sequence(sample, enc_input=enc_input, key=sk2, name=k)
      sampled_latents[k] = sample

      sub_model = substitute(gen_model_sampler, sampled_latents)
      exec_trace = trace(sub_model).get_trace(sk3)

    return sampled_latents, exec_trace, log_p