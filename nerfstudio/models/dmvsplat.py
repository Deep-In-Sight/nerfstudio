"""DMVSplat: 3DGS model with depth regularization and gaussian anchoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch
from torch.nn import Parameter

from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig


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
