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
