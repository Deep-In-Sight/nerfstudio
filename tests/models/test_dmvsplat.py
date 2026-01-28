"""Tests for DMVSplat model"""

import pytest
import torch


def test_dmvsplat_config_exists():
    """Test that DMVSplatModelConfig can be imported"""
    from nerfstudio.models.dmvsplat import DMVSplatModelConfig

    config = DMVSplatModelConfig()

    # Check new config fields exist with defaults
    assert config.depth_regularize == True
    assert config.depth_loss_weight == 0.8
    assert config.enable_anchoring == True
    assert config.anchor_distance == 0.1
    assert config.pruning_enable == False
    assert config.num_epoch_freeze_means == 10


def test_dmvsplat_depth_loss():
    """Test depth loss computation with masking"""
    from nerfstudio.models.dmvsplat import compute_depth_loss

    # pred_depth: [H, W, 1], gt_depth: [H, W, 1], mask: [H, W, 1]
    pred_depth = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
    gt_depth = torch.tensor([[[1.0], [2.5]], [[0.0], [4.0]]])  # 0 = invalid
    mask = torch.tensor([[[1], [1]], [[1], [0]]])  # 0 = ignore

    loss = compute_depth_loss(pred_depth, gt_depth, mask)

    # Valid pixels: (0,0), (0,1) - gt>0 AND mask==1
    # (0,0): (1-1)^2 = 0
    # (0,1): (2-2.5)^2 = 0.25
    # Mean = 0.125
    assert abs(loss.item() - 0.125) < 0.01
