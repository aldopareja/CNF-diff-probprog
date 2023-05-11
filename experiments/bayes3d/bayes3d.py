from functools import partial
from pathlib import Path
from typing import List
from numpyro import distributions as dist
import numpyro as npy
from jax.random import split
from jax import numpy as jnp
import jax
import bayes3d as j
import numpy as np
from scipy.spatial.transform import Rotation as R
import equinox as eqx

class SceneSampler():
  def __init__(self, mesh_paths: List[Path], intrinsics: j.Intrinsics = None, renderer: j.Renderer = None, num_objects=1):
    if intrinsics is None:
      intrinsics = j.Intrinsics(height=150,
                                width=150,
                                fx=200.0, fy=200.0,
                                cx=150.0, cy=150.0,
                                near=0.001, far=6.0
                            )
    self.intrinsics = intrinsics
    
    if renderer is None:
      renderer= j.Renderer(intrinsics)
    self.renderer = renderer
    
    for mesh_path in mesh_paths:
      if mesh_path.is_absolute():
        renderer.add_mesh_from_file(mesh_path)
      else:
        renderer.add_mesh_from_file(j.utils.get_assets_dir()/mesh_path)

    self.num_objects = num_objects
    self.num_types = len(mesh_paths)
    
  def __call__(self, key):
    key, ks = split(key)
    num_objects = npy.sample("num_objects", dist.Categorical(probs=jnp.arange(self.num_objects,dtype=jnp.float32)/self.num_objects), 
                             rng_key=ks)
    object_ids = []
    poses = []
    for i in range(num_objects+1):
      key, *ks = split(key, 4)
      type_of_object = npy.sample(f"type_of_object_{i}", dist.Categorical(probs=jnp.arange(self.num_types,dtype=jnp.float32)/self.num_types), rng_key=ks[0])
      object_ids.append(type_of_object)
      
      pose = jnp.eye(4)
      pose = pose.at[:3,3].set(self.random_pose(ks[1], prefix=f"pose_{i}_"))
      
      pose = pose.at[:3,:3].set(self.random_rotation(ks[2], prefix=f"rot_{i}_"))
      poses.append(pose)
    
    poses = jnp.stack(poses)
    img = self.renderer.render_multiobject(poses, object_ids)
    depth_img = img[...,2]
    flattened_patches = self.image_to_flattened_patches(depth_img, 15)
    
    npy.sample("obs", dist.Normal(loc=flattened_patches), rng_key=key)
    
    return img, flattened_patches
  
  @staticmethod
  def get_patch_from_img(i, j, patch_size, img):
      return jax.lax.dynamic_slice(img, [i, j], [patch_size, patch_size]).flatten()

  @staticmethod
  @eqx.filter_jit
  def image_to_flattened_patches(img, patch_size):
      h, w = img.shape
      i_vals = np.arange(0, h, patch_size)
      j_vals = np.arange(0, w, patch_size)

      # Double vmap
      get_patches = jax.vmap(
          jax.vmap(
              SceneSampler.get_patch_from_img, 
              in_axes=(None, 0, None, None),  # Vectorize over i
          ),
          in_axes=(0, None, None, None),  # Vectorize over j
      )
      patches = get_patches(i_vals, j_vals, patch_size, img)

      return patches.reshape(-1, patch_size * patch_size)
  
  @staticmethod
  def reconsctruct_from_patches(patches, img_shape, patch_size):
    h, w = img_shape
    # Reshape the patches into a grid of 2D patches
    patches_reshaped = patches.reshape(h // patch_size, w // patch_size, patch_size, patch_size)
    # Transpose and reshape the grid into the original image
    img = patches_reshaped.transpose((0, 2, 1, 3)).reshape(h, w)
    return img
  
  @staticmethod
  def gather_point(img, i, j):
    return img[i,j,:3]
      
      
  def random_rotation(self, key, prefix=""):
    # sample a random rotation matrix from the uniform distribution over SO(3)
    ks = split(key, 3)
    u1, u2, u3 = [npy.sample(f'{prefix}_u{i}', dist.Uniform(0,1),rng_key=ks[i]) for i in range(3)]
    
    w = jnp.sqrt(1 - u1) * jnp.sin(2 * np.pi * u2)
    x = jnp.sqrt(1 - u1) * jnp.cos(2 * np.pi * u2)
    y = jnp.sqrt(u1) * jnp.sin(2 * np.pi * u3)
    z = jnp.sqrt(u1) * jnp.cos(2 * np.pi * u3)
    
    rot = R.from_quat([x, y, z, w])
    return jnp.array(rot.as_matrix(), dtype=jnp.float32)
  
  def random_pose(self, key, prefix=""):
    ks = split(key, 3)
    #sample using numpyro
    depth = npy.sample(f'{prefix}depth', dist.Uniform(2.0,5.5),rng_key=ks[0])
    # Sample depth uniformly from 1.0 to 5.0
    # depth = np.random.uniform(1.0, 5.0)
    
    # Compute the half-width and half-height of the image plane at this depth
    half_width = depth * (self.intrinsics.width / 2) / self.intrinsics.fx
    half_height = depth * (self.intrinsics.height / 2) / self.intrinsics.fy
    
    # Sample a point uniformly from the image plane at this depth using numpyro
    
    x = npy.sample(f'{prefix}x', dist.Uniform(-half_width, half_width),rng_key=ks[1])
    y = npy.sample(f'{prefix}y', dist.Uniform(-half_height, half_height),rng_key=ks[2])
    
    # The pose is the 3D point (x, y, depth)
    pose = jnp.array([x, y, depth])
    
    return pose
      
  