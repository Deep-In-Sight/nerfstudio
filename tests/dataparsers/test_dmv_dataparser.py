"""Tests for DMV dataparser"""

import numpy as np
import torch
from pytest import fixture
import pytest
from pathlib import Path


def test_dmv_dataparser_config_exists():
    """Test that DMVDataParserConfig can be imported and instantiated"""
    from nerfstudio.data.dataparsers.dmv_dataparser import DMVDataParserConfig

    config = DMVDataParserConfig(data=Path("/tmp/test"))
    assert config.data == Path("/tmp/test")
    assert config.colmap_path == Path("sparse/0")


@fixture
def mocked_dmv_dataset(tmp_path: Path):
    """Create minimal DMV dataset structure for testing"""
    from PIL import Image
    import struct

    # Create directories
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    (tmp_path / "depths").mkdir()
    (tmp_path / "sparse" / "0").mkdir(parents=True)

    # Create 3 test images with proper naming
    for i, (ts, cam, patch) in enumerate([
        ("1739427221.019772", "0", "0"),
        ("1739427221.019772", "0", "1"),
        ("1739427221.219651", "0", "0"),
    ]):
        img_name = f"L2PRO_camera_{cam}_{ts}_{patch}"
        Image.new("RGB", (100, 100)).save(tmp_path / "images" / f"{img_name}.jpg")
        Image.new("L", (100, 100)).save(tmp_path / "masks" / f"{img_name}.png")
        # 16-bit depth image
        depth_img = Image.new("I;16", (100, 100))
        depth_img.save(tmp_path / "depths" / f"{img_name}.png")

    # Create depth_range.csv
    with open(tmp_path / "depths" / "depth_range.csv", "w") as f:
        f.write("depth_name,depth_min,depth_max\n")
        f.write("L2PRO_camera_0_1739427221.019772_0.png,1.0,10.0\n")
        f.write("L2PRO_camera_0_1739427221.019772_1.png,1.0,10.0\n")
        f.write("L2PRO_camera_0_1739427221.219651_0.png,1.0,10.0\n")

    # Create minimal COLMAP files (binary format)
    # cameras.bin - single camera
    with open(tmp_path / "sparse" / "0" / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))  # num cameras
        f.write(struct.pack("<i", 1))  # camera_id
        f.write(struct.pack("<i", 1))  # model (PINHOLE)
        f.write(struct.pack("<QQ", 100, 100))  # width, height
        f.write(struct.pack("<dddd", 50.0, 50.0, 50.0, 50.0))  # fx, fy, cx, cy

    # images.bin - 3 images at different positions
    with open(tmp_path / "sparse" / "0" / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", 3))  # num images
        positions = [(0, 0, 0), (2, 0, 0), (0, 2, 0)]  # Camera positions
        names = [
            "L2PRO_camera_0_1739427221.019772_0.jpg",
            "L2PRO_camera_0_1739427221.019772_1.jpg",
            "L2PRO_camera_0_1739427221.219651_0.jpg",
        ]
        for idx, (pos, name) in enumerate(zip(positions, names), 1):
            f.write(struct.pack("<i", idx))  # image_id
            # quaternion (w, x, y, z) for identity rotation
            f.write(struct.pack("<dddd", 1.0, 0.0, 0.0, 0.0))
            # translation (this is -R^T @ position in COLMAP)
            f.write(struct.pack("<ddd", -pos[0], -pos[1], -pos[2]))
            f.write(struct.pack("<i", 1))  # camera_id
            # name as null-terminated string
            f.write(name.encode() + b"\x00")
            # num_points2D = 0
            f.write(struct.pack("<Q", 0))

    # points3D.bin - minimal points
    with open(tmp_path / "sparse" / "0" / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", 2))  # num points
        for i in range(2):
            f.write(struct.pack("<Q", i + 1))  # point3D_id
            f.write(struct.pack("<ddd", float(i), 0.0, 0.0))  # xyz
            f.write(struct.pack("<BBB", 128, 128, 128))  # rgb
            f.write(struct.pack("<d", 0.1))  # error
            f.write(struct.pack("<Q", 0))  # track_length

    return tmp_path


def test_dmv_dataparser_custom_initialization(mocked_dmv_dataset):
    """Test that DMVDataParser applies centering and scaling only"""
    from nerfstudio.data.dataparsers.dmv_dataparser import DMVDataParser, DMVDataParserConfig

    config = DMVDataParserConfig(data=mocked_dmv_dataset)
    parser = config.setup()
    outputs = parser.get_dataparser_outputs(split="train")

    # Check that scene_center and scene_scale are in metadata
    assert "scene_center" in outputs.metadata
    assert "scene_scale" in outputs.metadata

    # Camera positions after transform should be centered
    camera_positions = outputs.cameras.camera_to_worlds[:, :3, 3]
    center = camera_positions.mean(dim=0)

    # Center should be close to origin
    assert torch.allclose(center, torch.zeros(3), atol=0.1)

    # Median distance should be close to 1.0
    distances = torch.norm(camera_positions - center, dim=1)
    median_dist = torch.median(distances)
    assert abs(median_dist - 1.0) < 0.1


def test_dmv_dataparser_view_sampling_order(mocked_dmv_dataset):
    """Test that views are sorted by (timestamp, camera_id, patch_id)"""
    from nerfstudio.data.dataparsers.dmv_dataparser import DMVDataParser, DMVDataParserConfig

    config = DMVDataParserConfig(data=mocked_dmv_dataset, eval_mode="all")
    parser = config.setup()
    outputs = parser.get_dataparser_outputs(split="train")

    # Extract filenames
    filenames = [f.name for f in outputs.image_filenames]

    # Expected order: sorted by (timestamp, camera_id, patch_id)
    expected = [
        "L2PRO_camera_0_1739427221.019772_0.jpg",
        "L2PRO_camera_0_1739427221.019772_1.jpg",
        "L2PRO_camera_0_1739427221.219651_0.jpg",
    ]

    assert filenames == expected


def test_dmv_dataparser_depth_range_loading(mocked_dmv_dataset):
    """Test that depth_range.csv is loaded into metadata"""
    from nerfstudio.data.dataparsers.dmv_dataparser import DMVDataParser, DMVDataParserConfig

    config = DMVDataParserConfig(data=mocked_dmv_dataset)
    parser = config.setup()
    outputs = parser.get_dataparser_outputs(split="train")

    # Check depth_range is in metadata
    assert "depth_range" in outputs.metadata
    depth_range = outputs.metadata["depth_range"]

    # Check it's a dict with expected keys
    assert isinstance(depth_range, dict)
    assert "L2PRO_camera_0_1739427221.019772_0.png" in depth_range

    # Check values are (min, max) tuples
    min_d, max_d = depth_range["L2PRO_camera_0_1739427221.019772_0.png"]
    assert min_d == 1.0
    assert max_d == 10.0
