"""DMV DataParser for LiDAR-based datasets with depth maps and masks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

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
