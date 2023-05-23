from jax import numpy as jnp

from tensorflow_probability.substrates import jax as tfp_jax
from tensorflow_probability.substrates import numpy as tfp_np
tfd_np = tfp_jax.distributions
tfb_np = tfp_jax.bijectors

class ClippingBijector(tfb_np.Bijector):
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

def make_bounding_bijector_np(lower_bound=None, upper_bound=None):
  if lower_bound is not None and upper_bound is not None:
    scale = upper_bound - lower_bound
    shift = lower_bound
    return tfb_np.Chain([
      ClippingBijector(clip_value_min=lower_bound, clip_value_max=upper_bound),
      tfb_np.Shift(shift=shift),
      tfb_np.Scale(scale=scale),
      tfb_np.Sigmoid()
    ])
  elif lower_bound is not None:
    return tfb_np.Chain([
      ClippingBijector(clip_value_min=lower_bound),
      tfb_np.Shift(shift=lower_bound),
      tfb_np.Softplus()
    ])
  elif upper_bound is not None:
    return tfb_np.Chain([
      ClippingBijector(clip_value_max=upper_bound),
      tfb_np.Shift(shift=upper_bound),
      tfb_np.Scale(scale=-1),
      tfb_np.Softplus()
    ])
  else:
    return tfb_np.Identity()

def make_standardization_bijector_np(mean, std):
  return tfb_np.Chain([tfb_np.Shift(shift=mean), 
                    tfb_np.Scale(scale=std)])
  
def make_bounding_and_standardization_bijector_np(variable_metadata):
  mean, std, lower_bound, upper_bound = map(lambda x: getattr(variable_metadata, x), ['mean', 'std', 'lower_bound', 'upper_bound'])
  return tfb_np.Chain([make_bounding_bijector_np(lower_bound, upper_bound), 
                    make_standardization_bijector_np(mean, std)])
  