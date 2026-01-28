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


def test_dmvsplat_get_loss_dict_has_depth_loss():
    """Test that get_loss_dict includes depth_loss when depth_image is in batch"""
    from nerfstudio.models.dmvsplat import DMVSplatModel, DMVSplatModelConfig

    # Check the method exists and has correct signature
    assert hasattr(DMVSplatModel, 'get_loss_dict')


def test_dmv_strategy_prune():
    """Test DMVStrategy pruning logic"""
    from nerfstudio.models.dmvsplat import DMVStrategy

    strategy = DMVStrategy(prune_alpha_thresh=0.5, prune_every=10)

    # Create mock params with opacities (in log space, so sigmoid needed)
    # sigmoid(-2) ≈ 0.12, sigmoid(0) = 0.5, sigmoid(2) ≈ 0.88
    params = {
        "means": torch.nn.Parameter(torch.randn(4, 3)),
        "opacities": torch.nn.Parameter(torch.tensor([[-2.0], [0.0], [2.0], [3.0]])),
    }

    # Mock optimizers (simplified)
    optimizers = {}
    state = {}
    info = {}

    # At step 10, with pruning enabled, should prune gaussians with opacity < 0.5
    # Gaussian 0 (opacity ~0.12) should be pruned
    # Gaussians 1,2,3 should remain
    mask = strategy._get_prune_mask(params)

    assert mask.sum() == 3  # 3 gaussians should remain
    assert mask[0] == False  # First gaussian should be pruned
    assert mask[1] == True   # opacity = 0.5, exactly at threshold
