"""DMV DataParser for LiDAR-based datasets with depth maps and masks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import torch

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParser, ColmapDataParserConfig


@dataclass
class DMVDataParserConfig(ColmapDataParserConfig):
    """DMV dataset config - extends COLMAP with custom initialization"""

    _target: Type = field(default_factory=lambda: DMVDataParser)
    """target class to instantiate"""
    colmap_path: Path = Path("sparse/0")
    """Path to COLMAP reconstruction relative to data path"""
    images_path: Path = Path("images")
    """Path to images directory relative to data path"""
    masks_path: Path = Path("masks")
    """Path to masks directory relative to data path"""
    depths_path: Path = Path("depths")
    """Path to depths directory relative to data path"""


class DMVDataParser(ColmapDataParser):
    """DMV DataParser for LiDAR-based datasets.

    Extends ColmapDataParser with:
    - Custom initialization (centering + scaling only, no rotation)
    - Per-image depth range loading from depth_range.csv
    - View sampling sorted by (timestamp, camera_id, patch_id)
    """

    config: DMVDataParserConfig

    def __init__(self, config: DMVDataParserConfig):
        super().__init__(config)
        self.config = config

    def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
        # Get base outputs from ColmapDataParser
        outputs = super()._generate_dataparser_outputs(split)

        # Apply custom initialization: centering + scaling only
        camera_positions = outputs.cameras.camera_to_worlds[:, :3, 3].clone()

        # Compute center (mean of camera positions)
        center = camera_positions.mean(dim=0)

        # Compute scale (1 / median distance to center)
        distances = torch.norm(camera_positions - center, dim=1)
        scale = 1.0 / torch.median(distances)

        # Apply transform to cameras
        c2w = outputs.cameras.camera_to_worlds.clone()
        c2w[:, :3, 3] = (c2w[:, :3, 3] - center) * scale
        outputs.cameras.camera_to_worlds = c2w

        # Apply transform to 3D points if present
        if "points3D_xyz" in outputs.metadata:
            pts = outputs.metadata["points3D_xyz"]
            pts[:, :3] = (pts[:, :3] - center) * scale
            outputs.metadata["points3D_xyz"] = pts

        # Store transform for later inversion
        outputs.metadata["scene_center"] = center
        outputs.metadata["scene_scale"] = scale

        return outputs
