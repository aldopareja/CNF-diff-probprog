from functools import partial
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt

from numpyro import distributions as dist
import numpyro as npy
from jax.random import split, PRNGKey
from jax import numpy as jnp
import jax
import bayes3d as j
import numpy as np
from scipy.spatial.transform import Rotation as R
import equinox as eqx


class SceneSampler():
  def __init__(self, *, mesh_paths: List[Path], intrinsics: j.Intrinsics = None, renderer: j.Renderer = None, num_objects=1, obs_depth_noise=0.1, pose_depth_noise=0.1, max_pose_xy_noise=0.3):
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
        scaling_factor = 1.0
        if "mug" in str(mesh_path):
          scaling_factor = 3.0
        renderer.add_mesh_from_file(j.utils.get_assets_dir()/mesh_path, scaling_factor=scaling_factor)

    self.num_objects = num_objects
    self.num_types = len(mesh_paths)
    self.obs_depth_noise = obs_depth_noise
    self.pose_depth_noise = pose_depth_noise
    self.max_pose_xy_noise = max_pose_xy_noise
    
    
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
    
    obs = npy.sample("obs", dist.Normal(loc=flattened_patches, scale=self.obs_depth_noise), rng_key=key)
    
    return img, flattened_patches, obs
  
  def debug_sampler(self):
      ks = split(PRNGKey(67589), 4)
      pose = jnp.eye(4)
      
      img, flattened_patches, obs = self(ks[0])
      img_ = j.viz.get_depth_image(img[:,:,2])
      img_.save("test.png")
      reconstructed = self.reconsctruct_from_patches(obs, (150,150), 15)
      j.viz.get_depth_image(reconstructed).save("test2.png")
      from IPython import embed; embed(using=False)
      
      poses = []
      for k in split(ks[0], 1000):
        poses.append(pose.at[:3,3].set(self.random_pose(k, prefix=f"pose_")))
      # poses = jnp.stack([pose.at[:3,3].set(self.random_pose(k, )) for k in split(ks[0],10000)])
      # poses = jnp.stack([p.at[:3,:3].set(self.random_rotation(k,)) for k,p in zip(split(ks[1],10000), poses)])
      imgs = self.renderer.render_parallel(poses, 0)
      range_fn = lambda x: jnp.max(x) - jnp.min(x)
      from jax import vmap
      unique_ranges = vmap(range_fn)(imgs[...,2])
      idx = unique_ranges.argsort()
      img_ = j.viz.get_depth_image(imgs[idx[-1]][...,2])
      img_.save(f"test.png")
      # get all x values flattened from imgs
      xs = imgs[...,0].flatten()
      ys = imgs[...,1].flatten()
      depths = imgs[...,2].flatten()
      # get the sizes of the 10 quantiles
      quantiles = jnp.quantile(xs, jnp.linspace(0,1,10))
      sizes = quantiles[1:] - quantiles[:-1]
      from IPython import embed; embed(using=False)
      
      pose = pose.at[:3,3].set(jnp.array([-0.1, -0.1, 5.5]))
      img = self.renderer.render_single_object(pose,0)
      j.viz.get_depth_image(img[:,:,2]).save("test.png")
      # from IPython import embed; embed(using=False)
      
      rots = []
      for k in split(ks[-3], 10000):
        rots.append(self.random_rotation(k,))
      
      angles = []
      for r in rots:
          quat = R.from_matrix(r).as_quat()
          w = np.clip(quat[3], -1, 1)  # Ensure w is within the valid range
          angle = 2 * np.arccos(w)
          angles.append(angle)

      angles = jnp.stack(angles)*180/np.pi
      np.quantile(angles, np.arange(11)/10.0)
      from IPython import embed; embed(using=False)
  
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
      
  @staticmethod
  @eqx.filter_jit
  def make_noise_quaternion(key):
    ks = split(key, 2)
    u4, u5 = dist.Uniform(0,1).sample(ks[0], sample_shape=(2,))    
    
    # sample the angle from a Normal distribution with mean 0 and std 0.1 * pi / 2
    angle = dist.Normal(0, 0.1 * np.pi / 2).sample(ks[1])
    
    w2 = jnp.cos(angle / 2)
    x2 = jnp.sqrt(1 - w2**2) * jnp.sin(2 * np.pi * u4)
    y2 = jnp.sqrt(1 - w2**2) * jnp.cos(2 * np.pi * u4)
    z2 = jnp.sqrt(1 - w2**2) * jnp.sin(2 * np.pi * u5)
    return jnp.stack([w2, x2, y2, z2])
      
  def random_rotation(self, key, prefix=""):
    # sample a random rotation matrix from the uniform distribution over SO(3) Marsaglia method 
    ks = split(key, 4)
    u1, u2, u3 = [npy.sample(f'{prefix}u{i}', dist.Uniform(0,1),rng_key=ks[i]) for i in range(3)]

    rot = R.from_quat(make_quaternion(u1, u2, u3))
    noise = R.from_quat(self.make_noise_quaternion(ks[3]))

    # Apply the small rotation to the original quaternion
    new_rot_quat = rot * noise
    
    return jnp.array(new_rot_quat.as_matrix(), dtype=jnp.float32)
  
  def random_pose(self, key, prefix=""):
    ks = split(key, 6)
    # sample some depth from the uniform distribution over [2, 5.5], this samples uniformly over the viewable cone since closer planes are smaller
    sqrt_depth = npy.sample(f'{prefix}sqrt_depth', dist.Uniform(jnp.sqrt(2.0), jnp.sqrt(5.5)),rng_key=ks[0])
    # Square it to get the depth
    depth = sqrt_depth ** 2
    
    # add untraced noise to the depth
    depth += dist.Normal(0, self.pose_depth_noise).sample(ks[1])

    # Compute the full width and height of the image plane at this depth
    full_width = depth * self.intrinsics.width / self.intrinsics.fx
    full_height = depth * self.intrinsics.height / self.intrinsics.fy
    
    # The bottom-right corner of the image plane maps to (0,0) in the world, so the
    # other corners are at (-full_width, 0), (0, -full_height), and (-full_width, -full_height).
    # Sample a point uniformly from this rectangle using numpyro
    x = npy.sample(f'{prefix}x', dist.Uniform(-full_width, -0.0),rng_key=ks[2])
    y = npy.sample(f'{prefix}y', dist.Uniform(-full_height, -0.0),rng_key=ks[3])
    
    # Add untraced noise to the x and y coordinates proportional to the depth
    x += dist.Normal(0, self.max_pose_xy_noise/self.intrinsics.far * depth).sample(ks[4])
    y += dist.Normal(0, self.max_pose_xy_noise/self.intrinsics.far * depth).sample(ks[5])
    
    # The pose is the 3D point (x, y, depth)
    pose = jnp.array([x, y, depth])
    
    return pose
 
############# utils ################     
@eqx.filter_jit
def make_quaternion(u1, u2, u3):
      
  w = jnp.sqrt(1 - u1) * jnp.sin(2 * np.pi * u2)
  x = jnp.sqrt(1 - u1) * jnp.cos(2 * np.pi * u2)
  y = jnp.sqrt(u1) * jnp.sin(2 * np.pi * u3)
  z = jnp.sqrt(u1) * jnp.cos(2 * np.pi * u3)
  
  return jnp.stack([x, y, z, w])

def plot_heatmap_on_sphere(data, num_bins=50):
    # Convert Cartesian coordinates to spherical coordinates
    r = np.linalg.norm(data, axis=1)
    theta = np.arctan2(data[:,1], data[:,0])
    phi = np.arccos(data[:,2]/r)
    
    # Create a 2D histogram in theta-phi space
    hist, theta_edges, phi_edges = np.histogram2d(theta, phi, bins=num_bins)
    from IPython import embed; embed(using=False)
    # Convert the bin edges to bin centers
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
    phi_centers = (phi_edges[:-1] + phi_edges[1:]) / 2
    
    # Create a grid in theta-phi space
    theta_grid, phi_grid = np.meshgrid(theta_centers, phi_centers)
    
    # Convert the grid back to Cartesian coordinates
    x = np.sin(phi_grid) * np.cos(theta_grid)
    y = np.sin(phi_grid) * np.sin(theta_grid)
    z = np.cos(phi_grid)
    
    # Plot the sphere with the colors set by the histogram count values
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=plt.cm.viridis(hist/hist.max()), linewidth=0, antialiased=False)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])

    # Add a colorbar
    m = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    m.set_array(hist)
    plt.colorbar(m, ax=ax)
    fig.savefig("test.png")