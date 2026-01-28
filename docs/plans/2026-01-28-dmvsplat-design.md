# DMVSplat: Custom 3DGS Training Method

## Overview

DMVSplat is a custom nerfstudio method based on splatfacto, designed for training 3D Gaussian Splatting models on LiDAR-based datasets with dense point clouds, depth maps, and binary masks.

## Components

| File | Type | Purpose |
|------|------|---------|
| `nerfstudio/data/dataparsers/dmv_dataparser.py` | New | `DMVDataParserConfig`, `DMVDataParser` |
| `nerfstudio/data/datasets/dmv_dataset.py` | New | `DMVDataset` |
| `nerfstudio/models/dmvsplat.py` | New | `DMVSplatModelConfig`, `DMVSplatModel`, `DMVStrategy` |
| `nerfstudio/engine/epoch_trainer.py` | New | `EpochTrainerConfig`, `EpochTrainer` |
| `nerfstudio/configs/method_configs.py` | Modify | Register `dmvsplat` method |

## Dataset Format

Expected directory structure:
```
dataset/
├── images/          (.jpg files)
├── masks/           (.png files, 1:1 with images)
├── depths/          (.png files + depth_range.csv)
│   └── depth_range.csv
├── sparse/0/        (cameras.bin, images.bin, points3D.bin)
└── *.las            (point cloud, optional)
```

File naming: `{device}_camera_{cam_id}_{timestamp}_{patch_id}.ext`

Example: `L2PRO_camera_0_1739427221.019772_0.jpg`

### depth_range.csv Format

Per-image depth range with header:
```csv
depth_name,depth_min,depth_max
L2PRO_camera_0_1739427222.019168_0.png,0.446,10.618
```

### Depth Encoding

- 16-bit PNG: values in [0, 65535]
- 0 = invalid depth
- [1, 65535] maps to [depth_min, depth_max] from CSV

Decoding formula:
```python
depth = torch.where(
    raw > 0,
    (raw - 1) / 65534 * (depth_max - depth_min) + depth_min,
    torch.zeros_like(raw)  # 0 remains invalid
)
```

## DMVDataParser

Extends `ColmapDataParser`.

### Custom Initialization

Only centering + scaling (no rotation, no axis reorder):

```python
# Center = mean of camera positions
camera_positions = poses[:, :3, 3]
center = camera_positions.mean(dim=0)

# Scale = 1 / median distance to center
distances = (camera_positions - center).norm(dim=-1)
scale = 1.0 / distances.median()

# Apply to cameras and points
poses[:, :3, 3] = (poses[:, :3, 3] - center) * scale
points3D = (points3D - center) * scale
```

Store `scene_center` and `scene_scale` in metadata for later inversion.

### View Sampling Order

Sort by (timestamp, camera_id, patch_id):

```python
def _sort_key(filename):
    parts = filename.stem.rsplit("_", 2)
    timestamp = float(parts[-2])
    patch_id = int(parts[-1])
    camera_id = int(parts[-3].split("_")[-1])
    return (timestamp, camera_id, patch_id)
```

### Depth Range Loading

```python
depth_range_df = pd.read_csv(data_path / "depths" / "depth_range.csv")
depth_range = {
    row["depth_name"]: (row["depth_min"], row["depth_max"])
    for _, row in depth_range_df.iterrows()
}
metadata["depth_range"] = depth_range
```

## DMVDataset

Extends `InputDataset`.

Overrides `get_metadata()` to decode 16-bit depth using per-image min/max from `depth_range` metadata.

## DMVSplatModel

Extends `SplatfactoModel`.

### Config Fields

```python
@dataclass
class DMVSplatModelConfig(SplatfactoModelConfig):
    # Depth regularization
    depth_regularize: bool = True
    depth_loss_weight: float = 0.8

    # Anchoring
    enable_anchoring: bool = True
    anchor_distance: float = 0.1  # meters, scaled at runtime

    # ADC (pruning only)
    pruning_enable: bool = False
    prune_alpha_thresh: float = 0.1
    prune_every: int = 100

    # Scheduling
    num_epoch_freeze_means: int = 10
```

### Depth Regularization

In `get_loss_dict()`:

```python
if self.config.depth_regularize and "depth_image" in batch:
    gt_depth = batch["depth_image"]
    pred_depth = outputs["depth"]

    # Valid = has depth AND not ignored
    valid_mask = (gt_depth > 0)
    if "mask" in batch:
        valid_mask = valid_mask & batch["mask"]

    depth_loss = ((pred_depth - gt_depth) ** 2 * valid_mask).sum() / valid_mask.sum().clamp(min=1)
    loss_dict["depth_loss"] = self.config.depth_loss_weight * depth_loss
```

Mask handling:
- RGB loss: uses `batch["mask"]` only (ignore mask)
- Depth loss: uses `batch["mask"]` AND `gt_depth > 0`

### Gaussian Anchoring

Store anchors after initialization:

```python
def populate_modules(self):
    super().populate_modules()
    if self.config.enable_anchoring:
        self.register_buffer("anchors", self.means.detach().clone())
        self.anchor_distance_scaled = self.config.anchor_distance * self.scale_factor
```

Enforce constraint in `step_post_backward()`:

```python
if self.config.enable_anchoring:
    with torch.no_grad():
        displacement = self.means - self.anchors
        distance = displacement.norm(dim=-1, keepdim=True)
        exceeded = distance > self.anchor_distance_scaled

        clamped = self.anchors + displacement / distance.clamp(min=1e-6) * self.anchor_distance_scaled
        self.means.data = torch.where(exceeded, clamped, self.means.data)
```

When pruning, also prune anchors:
```python
if self.config.enable_anchoring:
    self.anchors = self.anchors[mask]
```

### Means Freezing

In `step_cb()`:

```python
current_epoch = step // self.num_images
if current_epoch < self.config.num_epoch_freeze_means:
    if self.gauss_params["means"].grad is not None:
        self.gauss_params["means"].grad = None
```

## DMVStrategy

Standalone class (does not extend gsplat.DefaultStrategy).

Only supports optional opacity-based pruning. No densification, no opacity reset, no split/duplicate.

```python
class DMVStrategy:
    def __init__(self, prune_alpha_thresh: float = 0.1, prune_every: int = 100):
        self.prune_alpha_thresh = prune_alpha_thresh
        self.prune_every = prune_every

    def step_pre_backward(self, *args, **kwargs):
        pass

    def step_post_backward(
        self,
        params: Dict[str, torch.Tensor],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict,
        step: int,
        info: Dict,
        pruning_enable: bool = False,
    ):
        if not pruning_enable:
            return

        if step > 0 and step % self.prune_every == 0:
            opacities = torch.sigmoid(params["opacities"].squeeze())
            mask = opacities >= self.prune_alpha_thresh
            self._prune(params, optimizers, mask)

    def _prune(self, params, optimizers, mask):
        """Remove gaussians where mask is False."""
        for name, param in params.items():
            params[name] = torch.nn.Parameter(param[mask])
            # Update optimizer state accordingly
```

## EpochTrainer

Extends `Trainer`.

### Config Fields

```python
@dataclass
class EpochTrainerConfig(TrainerConfig):
    num_epochs: int = 100
```

### Implementation

```python
class EpochTrainer(Trainer):
    config: EpochTrainerConfig

    def setup(self):
        super().setup()
        num_images = len(self.pipeline.datamanager.train_dataset)

        # Set max_num_iterations based on epochs
        self.config.max_num_iterations = self.config.num_epochs * num_images

        # Update scheduler max_steps to match
        for scheduler in self.optimizers.schedulers.values():
            if hasattr(scheduler, 'config'):
                scheduler.config.max_steps = self.config.max_num_iterations
```

Uses existing `ExponentialDecaySchedulerConfig` for smooth per-step LR decay.

## Method Registration

In `nerfstudio/configs/method_configs.py`:

```python
from nerfstudio.data.dataparsers.dmv_dataparser import DMVDataParserConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.models.dmvsplat import DMVSplatModelConfig
from nerfstudio.engine.epoch_trainer import EpochTrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

descriptions["dmvsplat"] = "DMV 3DGS method with depth regularization and anchoring"

method_configs["dmvsplat"] = EpochTrainerConfig(
    method_name="dmvsplat",
    num_epochs=100,
    pipeline=VanillaPipelineConfig(
        datamanager=FullImageDatamanagerConfig(
            dataparser=DMVDataParserConfig(),
            cache_images_type="uint8",
        ),
        model=DMVSplatModelConfig(
            output_depth_during_training=True,
        ),
    ),
    optimizers={
        # Same as splatfacto, with ExponentialDecaySchedulerConfig
    },
)
```

## Configuration Summary

| Feature | Config Flag | Default |
|---------|-------------|---------|
| Depth regularization | `depth_regularize` | `True` |
| Depth loss weight | `depth_loss_weight` | `0.8` |
| Gaussian anchoring | `enable_anchoring` | `True` |
| Anchor distance (meters) | `anchor_distance` | `0.1` |
| Opacity pruning | `pruning_enable` | `False` |
| Prune threshold | `prune_alpha_thresh` | `0.1` |
| Prune interval | `prune_every` | `100` |
| Means frozen epochs | `num_epoch_freeze_means` | `10` |
| Total epochs | `num_epochs` | `100` |

## Export / Inverse Transform

To recover world coordinates after training:

```python
world_xyz = (scene_xyz / scene_scale) + scene_center
```

`scene_center` and `scene_scale` are stored in dataparser metadata.
