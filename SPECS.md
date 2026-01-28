# 3dgs customized training

## Status: Implemented

Implementation complete. See `docs/plans/2026-01-28-dmvsplat-design.md` for design.

## about

This project aims to implement another nerfstudio method based on splatfacto. This method supports:

- a custom dataset, and
- a custom training algorithm.

Consult [docs](https://docs.nerf.studio/developer_guides/new_methods.html) to see how to register a new method.

### Dataset

The input dataset is a COLMAP-based dataset, called LiDARCOLMAP:

- images/: images directory, contains RGB images file with flattened named
  - [camera_id]_[timestamp_s].[timestamp_ns]_[patch_id].jpg
  - eg: `L2PRO_camera_0_1739427221.019772_0.jpg`
- masks/: masks directory, 1:1 paired with above images. Masks are binary, 0 means ignore the pixel in images
  - [camera_id]_[timestamp_s].[timestamp_ns]_[patch_id].png
  - eg: `L2PRO_camera_0_1739427221.019772_0.png`
- depths/: depth map directory, 1:1 paired with images. Depth are 16bit encoded, 0 is reserved for invalid depth. [1,65535] encode a depth range [min, max] stored in depths/depth_range.csv
  - [camera_id]_[timestamp_s].[timestamp_ns]_[patch_id].png
  - eg: `L2PRO_camera_0_1739427221.019772_0.png`
- pointcloud.las: the dense pointcloud (not yet used)
- sparse/0:
  - images.bin: image poses
  - cameras.bin: camera intrinsics
  - points3D.bin: calibrated image features.

### Initialization

current simple_trainer applied a series of transformation on the cameras/points after loading. For our dataset:

- pointcloud +z is already up. so no rotation needed.
- we don't want +z to align to shortest principal axis
- the only transformation needed is:
  - centering: translate the scene to center of camera positions
  - scaling: scale the scene so that median of camera distance (to center) is 1.0
- This transformation needs to be remember so that it can be inversed after the model is trained.

Refer to current transformation and adapt accordingly.

### Depth regularization and masking

Each view contains:

- camera and pose
- rgb image
- depth map: 0=invalid depth
- ignore mask: 1=update, 0=ignore

If depth_regularize=True, the rasterizer should run in RGB+ED mode, then the rendered depth map is compared with provided depth map to get geometry_loss (L2).

The gradient is applied using a given mask.

### Anchoring

Each guassian orignal position is remembered, called that its anchor. The optimizer is not allowed to move a gaussian away from its anchor more than an anchor_distance.

If a gaussian is splited or duplicated, the 2 new gaussian share its original anchor.
In case the gaussian is splitted, the child is not allowed to be placed away from the anchor more than anchor_distance.

Anchoring is enabled with enable_anchoring

### View sampling

The cameras is rigged on a chassis. There are 2 fisheye camera looking in opposite direction, called camera_0 and camera_1. Each fished is cube-mapped on to 3 rectilinear patches. The image name is formed as
<camera_id>/<timestamp>_<patch_id>.jpg or <camera_id>_<timestamp>_<patch_id>.jpg

Transpose the names to <timestamp>_<camera_id>_<patch_id>.jpg, then sort and sample sequentially

```text
1739427221.019772_camera_0_0.jpg
1739427221.019772_camera_0_1.jpg
1739427221.019772_camera_0_2.jpg
1739427221.019772_camera_1_0.jpg
1739427221.019772_camera_1_1.jpg
1739427221.019772_camera_1_2.jpg
1739427221.219651_camera_0_0.jpg
1739427221.219651_camera_0_1.jpg
1739427221.219651_camera_0_2.jpg
1739427221.219651_camera_1_0.jpg
1739427221.219651_camera_1_1.jpg
1739427221.219651_camera_1_2.jpg
```

### ADC

Adaptive Density Control is an essential part of 3dgs training loop. Gsplat implement a DefaultStrategy for reconstruction based on SfM sparse model. Since our input dataset has dense points from LiDAR-IMU SLAM, this strategy should be modified

1. Duplicate - Gaussians with high image-plane gradients but small 3D scales are duplicated
2. Split - Gaussians with high image-plane gradients and large 3D scales are split into smaller ones
3. Reset opacity - Periodically resets all Gaussians to lower opacity values 
The initial point cloud is quite dense, so densification and opacity reset can be disabled.
4. Prune - Gaussians with low opacity (or excessively large scales) are removed
Temporarily skip this to get more stats in a few trial runs.

### Scheduling / hyperparameters

One step is one per-view update.
One epoch is one iteration through the whole dataset.

Model is trained over num_epoch. Total steps is num_epoch * num_image.
Learning rate is updated every epoch, starting from epoch_lr_update_start to the end.

Means is freezed for the first num_epoch_freeze_means.
