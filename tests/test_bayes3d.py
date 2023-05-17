from experiments.bayes3d.bayes3d import SceneSampler
from pathlib import Path
from jax.random import PRNGKey, split
import bayes3d as j
from numpyro.handlers import trace

model = SceneSampler(mesh_paths=[
                                # Path("sample_objs/cube.obj"),
                                #  Path("sample_objs/sphere.obj"),
                                #  Path("sample_objs/bunny.obj"),
                                 Path("ycb_video_models/models/025_mug/textured_simple.obj"),
                                 ],
                     num_objects=1,
                     max_pose_xy_noise=0.3,
                     obs_depth_noise=0.5
                     )

model.debug_sampler()


img, flattened_patches = model(PRNGKey(67589))

img_ = j.viz.get_depth_image(img[:,:,2])
img_.save("test.png")

points_img = j.render_point_cloud(points, model.intrinsics)
points_img_ = j.viz.get_depth_image(points_img[:,:,2])
points_img_.save("test_points.png")