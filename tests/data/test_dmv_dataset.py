"""Tests for DMV dataset"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
import pytest


def test_dmv_dataset_depth_decoding():
    """Test 16-bit depth decoding with per-image min/max"""
    from nerfstudio.data.datasets.dmv_dataset import decode_depth_16bit

    # Create test depth image: raw values [0, 1, 32768, 65535]
    # 0 = invalid, 1 = min, 65535 = max
    raw = torch.tensor([[0, 1], [32768, 65535]], dtype=torch.int32)
    depth_min, depth_max = 2.0, 10.0

    depth = decode_depth_16bit(raw, depth_min, depth_max)

    # Expected:
    # 0 -> 0 (invalid)
    # 1 -> 2.0 (min)
    # 32768 -> ~6.0 (middle)
    # 65535 -> 10.0 (max)
    assert depth[0, 0, 0] == 0.0  # invalid (note: output has shape [H, W, 1])
    assert abs(depth[0, 1, 0] - 2.0) < 0.01  # min
    assert abs(depth[1, 1, 0] - 10.0) < 0.01  # max
    assert 5.5 < depth[1, 0, 0] < 6.5  # middle
