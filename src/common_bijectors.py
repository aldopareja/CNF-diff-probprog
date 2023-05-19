from jax import numpy as jnp

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

class ClippingBijector(tfb.Bijector):
  '''
  dummy clipping bijector that just restricts the values of the input in the reversed direction
  keeping the forward direction unchanged and making the ILDJ 0.0 regardless.
  
  This is necessary because bounding bijectors using Sigmoid or Softplus are undefined at the extremes.
  '''
  def __init__(self, clip_value_min=None, clip_value_max=None, clip_epsilon=4e-6, validate_args=False, name="clipping_bijector"):
      super().__init__(
          validate_args=validate_args,
          forward_min_event_ndims=0,
          name=name)
      
      self._clip_value_min = clip_value_min + clip_epsilon if clip_value_min is not None else jnp.finfo(jnp.float32).min
      self._clip_value_max = clip_value_max - clip_epsilon if clip_value_max is not None else jnp.finfo(jnp.float32).max

  def _forward(self, x):
      return x
  
  def _inverse(self, y):
      return jnp.clip(y, self._clip_value_min, self._clip_value_max)

  def _inverse_log_det_jacobian(self, y):
      # Return 0.0 for the ILDJ
      return jnp.zeros_like(y)
    
  def _forward_log_det_jacobian(self, x):
      # Return 0.0 for the FLDJ
      return jnp.zeros_like(x)

def make_bounding_bijector(lower_bound=None, upper_bound=None):
  if lower_bound is not None and upper_bound is not None:
    scale = upper_bound - lower_bound
    shift = lower_bound
    return tfb.Chain([
      ClippingBijector(clip_value_min=lower_bound, clip_value_max=upper_bound),
      tfb.Shift(shift=shift),
      tfb.Scale(scale=scale),
      tfb.Sigmoid()
    ])
  elif lower_bound is not None:
    return tfb.Chain([
      ClippingBijector(clip_value_min=lower_bound),
      tfb.Shift(shift=lower_bound),
      tfb.Softplus()
    ])
  elif upper_bound is not None:
    return tfb.Chain([
      ClippingBijector(clip_value_max=upper_bound),
      tfb.Shift(shift=upper_bound),
      tfb.Scale(scale=-1),
      tfb.Softplus()
    ])
  else:
    return tfb.Identity()

def make_standardization_bijector(mean, std):
  return tfb.Chain([tfb.Shift(shift=mean), 
                    tfb.Scale(scale=std)])
  
def make_bounding_and_standardization_bijector(variable_metadata):
  mean, std, lower_bound, upper_bound = map(lambda x: getattr(variable_metadata, x), ['mean', 'std', 'lower_bound', 'upper_bound'])
  return tfb.Chain([make_bounding_bijector(lower_bound, upper_bound), 
                    make_standardization_bijector(mean, std)])
  