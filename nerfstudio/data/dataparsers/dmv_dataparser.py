"""DMV DataParser for LiDAR-based datasets with depth maps and masks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import pandas as pd
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

    def _get_sorting_key(self, filename: Path) -> tuple:
        """Extract (timestamp, camera_id, patch_id) for sorting.

        Args:
            filename: Path with format L2PRO_camera_{cam_id}_{timestamp}_{patch_id}.ext

        Returns:
            Tuple of (timestamp, camera_id, patch_id) for sorting
        """
        stem = filename.stem
        parts = stem.rsplit("_", 2)  # Split from right to get [prefix, timestamp, patch]

        patch_id = int(parts[-1])
        timestamp = float(parts[-2])
        # Extract camera_id from prefix (e.g., "L2PRO_camera_0")
        camera_id = int(parts[-3].split("_")[-1])

        return (timestamp, camera_id, patch_id)

    def _load_depth_range(self) -> dict:
        """Load depth range from CSV file"""
        csv_path = self.config.data / self.config.depths_path / "depth_range.csv"
        if not csv_path.exists():
            return {}

        df = pd.read_csv(csv_path)
        return {
            row["depth_name"]: (row["depth_min"], row["depth_max"])
            for _, row in df.iterrows()
        }

    def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
        # Get base outputs from ColmapDataParser
        outputs = super()._generate_dataparser_outputs(split)

        # Sort by (timestamp, camera_id, patch_id)
        if len(outputs.image_filenames) > 0:
            indices = sorted(
                range(len(outputs.image_filenames)),
                key=lambda i: self._get_sorting_key(outputs.image_filenames[i])
            )

            # Reorder all lists
            outputs.image_filenames = [outputs.image_filenames[i] for i in indices]
            if outputs.mask_filenames:
                outputs.mask_filenames = [outputs.mask_filenames[i] for i in indices]

            # Reorder cameras (use tensor for indexing)
            outputs.cameras = outputs.cameras[torch.tensor(indices)]

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

        # Load depth range from CSV
        outputs.metadata["depth_range"] = self._load_depth_range()

        # Build depth_filenames from image filenames
        depths_dir = self.config.data / self.config.depths_path
        if depths_dir.exists():
            depth_filenames = []
            for img_path in outputs.image_filenames:
                depth_path = depths_dir / f"{img_path.stem}.png"
                depth_filenames.append(depth_path)
            outputs.metadata["depth_filenames"] = depth_filenames

        return outputs
