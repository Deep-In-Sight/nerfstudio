"""Integration test for DMVSplat with real dataset."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_dmvsplat_method_registered():
    """Test that dmvsplat method is recognized by ns-train"""
    result = subprocess.run(
        [sys.executable, "-m", "nerfstudio.scripts.train", "dmvsplat", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "dmvsplat" in result.stdout.lower() or "method_name" in result.stdout


def test_dmvsplat_dataparser_on_real_data():
    """Test DMVDataParser on real dataset if available"""
    dataset_path = Path.home() / "3dgs_ws" / "perspective_processed"

    if not dataset_path.exists():
        pytest.skip("Real dataset not available")

    from nerfstudio.data.dataparsers.dmv_dataparser import DMVDataParser, DMVDataParserConfig

    config = DMVDataParserConfig(data=dataset_path)
    parser = config.setup()
    outputs = parser.get_dataparser_outputs(split="train")

    assert len(outputs.image_filenames) > 0
    assert "scene_center" in outputs.metadata
    assert "scene_scale" in outputs.metadata
    assert "depth_range" in outputs.metadata
    assert len(outputs.metadata["depth_range"]) > 0
