# DMVSplat Standalone Model Design

## Overview

Decouple DMVSplatModel from SplatfactoModel to create a minimal, focused 3DGS model for LiDAR-initialized scenes. Remove unused features and add new constraints and telemetry.

## Changes

### Removed Features
- Camera optimizer
- Bilateral grid (ISP correction)
- Resolution schedule (progressive upscaling)
- Crop box support
- MCMC strategy
- Scale regularization (PhysGauss)
- Densification (split/duplicate/opacity reset)

### New Features
- **Scale clamping**: Per-axis limit to max `max_scale_factor` times initial scales
- **WandB histograms**: Log anchor distances, scales, scale ratios every N steps

## Config

```python
@dataclass
class DMVSplatModelConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: DMVSplatModel)

    # Rendering
    sh_degree: int = 3
    background_color: Literal["random", "black", "white"] = "random"
    rasterize_mode: Literal["classic", "antialiased"] = "classic"

    # Losses
    ssim_lambda: float = 0.2
    depth_regularize: bool = True
    depth_loss_weight: float = 0.8

    # Constraints
    enable_anchoring: bool = True
    anchor_distance: float = 0.1  # meters, scaled by scene_scale
    enable_scale_clamp: bool = True
    max_scale_factor: float = 2.0

    # Pruning
    pruning_enable: bool = False
    prune_alpha_thresh: float = 0.1
    prune_every: int = 100

    # Telemetry
    histogram_log_every: int = 1000
```

## Model Structure

```python
class DMVSplatModel(Model):
    # Buffers
    anchors: Tensor          # [N, 3] initial positions
    anchor_scales: Tensor    # [N, 3] initial log-scales

    # Params
    gauss_params: ParameterDict  # means, scales, quats, features_dc, features_rest, opacities
```

### Key Methods

#### `populate_modules()`
1. Initialize gaussian params from seed points
2. Store `anchors` buffer (initial means)
3. Store `anchor_scales` buffer (initial scales)
4. Compute `anchor_distance_scaled = anchor_distance * scene_scale`
5. Setup DMVStrategy, metrics (PSNR, SSIM, LPIPS)

#### `get_outputs(camera)`
- gsplat rasterization
- Always render RGB + depth
- No camera optimizer, no bilateral grid, no crop

#### `get_loss_dict(outputs, batch, metrics_dict)`
- L1 loss + SSIM loss (weighted by `ssim_lambda`)
- Depth loss (L2, masked, weighted by `depth_loss_weight`)

#### `step_post_backward(step)`
1. Enforce anchor constraint (clamp means within `anchor_distance_scaled`)
2. Enforce scale constraint (clamp scales to `anchor_scales + log(max_scale_factor)`)
3. Optional pruning via DMVStrategy
4. Log WandB histograms every `histogram_log_every` steps

## Constraint Implementation

### Anchor Constraint (existing)
```python
displacement = means - anchors
distance = displacement.norm(dim=-1, keepdim=True)
exceeded = distance > anchor_distance_scaled
clamped = anchors + displacement / distance.clamp(min=1e-6) * anchor_distance_scaled
means = torch.where(exceeded, clamped, means)
```

### Scale Constraint (new)
```python
# Scales stored in log-space
max_scales = anchor_scales + math.log(max_scale_factor)
scales = torch.clamp(scales, max=max_scales)
```

Per-axis clamping: each of the 3 scale dimensions clamped independently.

## Telemetry

Logged via WandB every `histogram_log_every` steps:

| Metric | Description |
|--------|-------------|
| `dmv/anchor_distances` | Distance of each gaussian from its anchor |
| `dmv/scales` | Absolute scale values (real space) |
| `dmv/scale_ratios` | Ratio vs initial scales (1.0 = unchanged) |

```python
import wandb

def _log_histograms(self, step: int):
    distances = (means - anchors).norm(dim=-1)
    wandb.log({"dmv/anchor_distances": wandb.Histogram(distances.cpu())}, step=step)

    scales_real = torch.exp(scales)
    wandb.log({"dmv/scales": wandb.Histogram(scales_real.cpu())}, step=step)

    scale_ratios = torch.exp(scales - anchor_scales)
    wandb.log({"dmv/scale_ratios": wandb.Histogram(scale_ratios.cpu())}, step=step)
```

## Migration

The new standalone DMVSplatModel replaces the previous version that inherited from SplatfactoModel. Config changes:

| Old (inherited) | New (standalone) |
|-----------------|------------------|
| All SplatfactoModelConfig options | Only listed options above |
| `camera_optimizer` | Removed |
| `use_bilateral_grid` | Removed |
| `num_downscales`, `resolution_schedule` | Removed |
| N/A | `enable_scale_clamp`, `max_scale_factor` |
| N/A | `histogram_log_every` |
