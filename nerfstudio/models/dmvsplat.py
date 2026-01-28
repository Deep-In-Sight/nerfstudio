"""DMVSplat: 3DGS model with depth regularization and gaussian anchoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch
from torch.nn import Parameter

from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig


def compute_depth_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute L2 depth loss with masking.

    Args:
        pred_depth: Predicted depth [H, W, 1]
        gt_depth: Ground truth depth [H, W, 1], 0 = invalid
        mask: Optional ignore mask [H, W, 1], 1 = use, 0 = ignore

    Returns:
        Scalar loss value
    """
    # Valid = has depth (gt > 0)
    valid_mask = (gt_depth > 0).float()

    # Combine with ignore mask if provided
    if mask is not None:
        valid_mask = valid_mask * mask.float()

    # L2 loss on valid pixels only
    diff_sq = (pred_depth - gt_depth) ** 2
    loss = (diff_sq * valid_mask).sum() / valid_mask.sum().clamp(min=1)

    return loss


@dataclass
class DMVSplatModelConfig(SplatfactoModelConfig):
    """DMVSplat Model Config"""

    _target: Type = field(default_factory=lambda: DMVSplatModel)
    """target class to instantiate"""

    # Depth regularization
    depth_regularize: bool = True
    """Whether to use depth regularization loss"""
    depth_loss_weight: float = 0.8
    """Weight for depth loss (higher = favor geometry over appearance)"""

    # Anchoring
    enable_anchoring: bool = True
    """Whether to anchor gaussians to their initial positions"""
    anchor_distance: float = 0.1
    """Maximum distance (meters) gaussians can move from anchor"""

    # ADC (pruning only)
    pruning_enable: bool = False
    """Whether to enable opacity-based pruning"""
    prune_alpha_thresh: float = 0.1
    """Opacity threshold for pruning"""
    prune_every: int = 100
    """Prune every N steps"""

    # Scheduling
    num_epoch_freeze_means: int = 10
    """Number of epochs to freeze gaussian means"""

    # Override defaults
    output_depth_during_training: bool = True
    """Always output depth during training for depth loss"""


class DMVSplatModel(SplatfactoModel):
    """DMVSplat model with depth regularization and anchoring."""

    config: DMVSplatModelConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_loss_dict(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], metrics_dict: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute losses including depth regularization."""
        # Get base losses from splatfacto
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # Add depth loss if enabled and depth available
        if self.config.depth_regularize and "depth_image" in batch and "depth" in outputs:
            gt_depth = batch["depth_image"].to(self.device)
            pred_depth = outputs["depth"]

            # Downscale gt_depth to match pred_depth if needed
            if gt_depth.shape[:2] != pred_depth.shape[:2]:
                gt_depth = self._downscale_if_required(gt_depth)

            # Get mask if available
            mask = batch.get("mask", None)
            if mask is not None:
                mask = self._downscale_if_required(mask).to(self.device)

            depth_loss = compute_depth_loss(pred_depth, gt_depth, mask)
            loss_dict["depth_loss"] = self.config.depth_loss_weight * depth_loss

        return loss_dict
