# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DMV Dataset with 16-bit depth decoding."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset


def decode_depth_16bit(raw: torch.Tensor, depth_min: float, depth_max: float) -> torch.Tensor:
    """Decode 16-bit depth image to metric depth.

    Args:
        raw: Raw 16-bit values [H, W] or [H, W, 1], dtype int
        depth_min: Minimum depth value (maps to raw=1)
        depth_max: Maximum depth value (maps to raw=65535)

    Returns:
        Depth in meters [H, W, 1]. 0 = invalid.
    """
    raw = raw.squeeze(-1) if raw.dim() == 3 else raw
    raw = raw.float()

    # [1, 65535] -> [min, max], 0 stays 0
    depth = torch.where(
        raw > 0,
        (raw - 1) / 65534 * (depth_max - depth_min) + depth_min,
        torch.zeros_like(raw)
    )

    return depth.unsqueeze(-1)  # [H, W, 1]


class DMVDataset(InputDataset):
    """Dataset for DMV format with 16-bit depth maps and per-image depth range."""

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["depth_image"]

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0, **kwargs):
        super().__init__(dataparser_outputs, scale_factor, **kwargs)
        self.depth_filenames = dataparser_outputs.metadata.get("depth_filenames", None)
        self.depth_range = dataparser_outputs.metadata.get("depth_range", {})

    def get_metadata(self, data: Dict) -> Dict:
        """Load depth image with per-image decoding."""
        metadata = {}

        if self.depth_filenames is not None and len(self.depth_filenames) > data["image_idx"]:
            depth_filepath = self.depth_filenames[data["image_idx"]]
            depth_name = depth_filepath.name

            # Load 16-bit depth
            pil_depth = Image.open(depth_filepath)
            if self.scale_factor != 1.0:
                width, height = pil_depth.size
                newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
                pil_depth = pil_depth.resize(newsize, resample=Image.Resampling.NEAREST)

            raw = torch.from_numpy(np.array(pil_depth)).int()

            # Get per-image depth range
            if depth_name in self.depth_range:
                depth_min, depth_max = self.depth_range[depth_name]
            else:
                # Fallback to default range
                depth_min, depth_max = 0.1, 100.0

            metadata["depth_image"] = decode_depth_16bit(raw, depth_min, depth_max)

        return metadata
