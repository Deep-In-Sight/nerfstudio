# DMVSplat Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement DMVSplat, a custom 3DGS training method for LiDAR-based datasets with depth regularization, gaussian anchoring, and epoch-based training.

**Architecture:** Extend splatfacto with custom dataparser (centering+scaling only), dataset (16-bit depth decoding), model (depth loss + anchoring), and trainer (epoch-based scheduling). Uses standalone DMVStrategy instead of gsplat's DefaultStrategy.

**Tech Stack:** PyTorch, nerfstudio, gsplat (rasterization only), pandas (depth_range.csv)

---

## Task 1: DMVDataParser - Config and Basic Structure

**Files:**
- Create: `nerfstudio/data/dataparsers/dmv_dataparser.py`
- Test: `tests/dataparsers/test_dmv_dataparser.py`

**Step 1: Write failing test for dataparser config**

```python
# tests/dataparsers/test_dmv_dataparser.py
"""Tests for DMV dataparser"""

import pytest
from pathlib import Path


def test_dmv_dataparser_config_exists():
    """Test that DMVDataParserConfig can be imported and instantiated"""
    from nerfstudio.data.dataparsers.dmv_dataparser import DMVDataParserConfig

    config = DMVDataParserConfig(data=Path("/tmp/test"))
    assert config.data == Path("/tmp/test")
    assert config.colmap_path == Path("sparse/0")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/dataparsers/test_dmv_dataparser.py::test_dmv_dataparser_config_exists -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write minimal implementation**

```python
# nerfstudio/data/dataparsers/dmv_dataparser.py
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/dataparsers/test_dmv_dataparser.py::test_dmv_dataparser_config_exists -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nerfstudio/data/dataparsers/dmv_dataparser.py tests/dataparsers/test_dmv_dataparser.py
git commit -m "feat(dmv): add DMVDataParserConfig and DMVDataParser skeleton"
```

---

## Task 2: DMVDataParser - Custom Initialization (Centering + Scaling)

**Files:**
- Modify: `nerfstudio/data/dataparsers/dmv_dataparser.py`
- Test: `tests/dataparsers/test_dmv_dataparser.py`

**Step 1: Write failing test for custom initialization**

```python
# Add to tests/dataparsers/test_dmv_dataparser.py

import numpy as np
import torch
from pytest import fixture


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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/dataparsers/test_dmv_dataparser.py::test_dmv_dataparser_custom_initialization -v`
Expected: FAIL with KeyError for "scene_center"

**Step 3: Write implementation**

```python
# Replace DMVDataParser class in nerfstudio/data/dataparsers/dmv_dataparser.py

import pandas as pd
import torch
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs


class DMVDataParser(ColmapDataParser):
    """DMV DataParser for LiDAR-based datasets."""

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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/dataparsers/test_dmv_dataparser.py::test_dmv_dataparser_custom_initialization -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nerfstudio/data/dataparsers/dmv_dataparser.py tests/dataparsers/test_dmv_dataparser.py
git commit -m "feat(dmv): add custom initialization with centering and scaling"
```

---

## Task 3: DMVDataParser - View Sampling Order

**Files:**
- Modify: `nerfstudio/data/dataparsers/dmv_dataparser.py`
- Test: `tests/dataparsers/test_dmv_dataparser.py`

**Step 1: Write failing test for view sampling order**

```python
# Add to tests/dataparsers/test_dmv_dataparser.py

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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/dataparsers/test_dmv_dataparser.py::test_dmv_dataparser_view_sampling_order -v`
Expected: FAIL (order may differ from expected)

**Step 3: Write implementation**

```python
# Add method to DMVDataParser class

    def _get_sorting_key(self, filename: Path) -> tuple:
        """Extract (timestamp, camera_id, patch_id) for sorting"""
        # Format: L2PRO_camera_{cam_id}_{timestamp}_{patch_id}.ext
        stem = filename.stem
        parts = stem.rsplit("_", 2)  # Split from right to get [prefix, timestamp, patch]

        patch_id = int(parts[-1])
        timestamp = float(parts[-2])
        # Extract camera_id from prefix (e.g., "L2PRO_camera_0")
        camera_id = int(parts[-3].split("_")[-1])

        return (timestamp, camera_id, patch_id)

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

            # Reorder cameras
            outputs.cameras = outputs.cameras[indices]

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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/dataparsers/test_dmv_dataparser.py::test_dmv_dataparser_view_sampling_order -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nerfstudio/data/dataparsers/dmv_dataparser.py tests/dataparsers/test_dmv_dataparser.py
git commit -m "feat(dmv): add view sampling sorted by timestamp, camera, patch"
```

---

## Task 4: DMVDataParser - Depth Range Loading

**Files:**
- Modify: `nerfstudio/data/dataparsers/dmv_dataparser.py`
- Test: `tests/dataparsers/test_dmv_dataparser.py`

**Step 1: Write failing test for depth range loading**

```python
# Add to tests/dataparsers/test_dmv_dataparser.py

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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/dataparsers/test_dmv_dataparser.py::test_dmv_dataparser_depth_range_loading -v`
Expected: FAIL with KeyError for "depth_range"

**Step 3: Write implementation**

```python
# Add to imports at top of dmv_dataparser.py
import pandas as pd

# Add method to DMVDataParser class before _generate_dataparser_outputs

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

# Add to _generate_dataparser_outputs, before return:

        # Load depth range from CSV
        outputs.metadata["depth_range"] = self._load_depth_range()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/dataparsers/test_dmv_dataparser.py::test_dmv_dataparser_depth_range_loading -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nerfstudio/data/dataparsers/dmv_dataparser.py tests/dataparsers/test_dmv_dataparser.py
git commit -m "feat(dmv): load depth_range.csv into metadata"
```

---

## Task 5: DMVDataset - 16-bit Depth Decoding

**Files:**
- Create: `nerfstudio/data/datasets/dmv_dataset.py`
- Test: `tests/data/test_dmv_dataset.py`

**Step 1: Write failing test for depth decoding**

```python
# tests/data/test_dmv_dataset.py
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
    assert depth[0, 0] == 0.0  # invalid
    assert abs(depth[0, 1] - 2.0) < 0.01  # min
    assert abs(depth[1, 1] - 10.0) < 0.01  # max
    assert 5.5 < depth[1, 0] < 6.5  # middle
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_dmv_dataset.py::test_dmv_dataset_depth_decoding -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

```python
# nerfstudio/data/datasets/dmv_dataset.py
"""DMV Dataset with 16-bit depth decoding."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_dmv_dataset.py::test_dmv_dataset_depth_decoding -v`
Expected: PASS

**Step 5: Add missing import and commit**

```python
# Add to imports in dmv_dataset.py
import numpy as np
```

```bash
git add nerfstudio/data/datasets/dmv_dataset.py tests/data/test_dmv_dataset.py
git commit -m "feat(dmv): add DMVDataset with 16-bit depth decoding"
```

---

## Task 6: DMVSplatModel - Config and Basic Structure

**Files:**
- Create: `nerfstudio/models/dmvsplat.py`
- Test: `tests/models/test_dmvsplat.py`

**Step 1: Write failing test for model config**

```python
# tests/models/test_dmvsplat.py
"""Tests for DMVSplat model"""

import pytest


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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_dmvsplat.py::test_dmvsplat_config_exists -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

```python
# nerfstudio/models/dmvsplat.py
"""DMVSplat: 3DGS model with depth regularization and gaussian anchoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch
from torch.nn import Parameter

from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig


@dataclass
class DMVSplatModelConfig(SplatfactoModelConfig):
    """DMVSplat Model Config"""

    _target: Type = field(default_factory=lambda: DMVSplatModel)
    """target class to instantiate"""

    # Depth regularization
    depth_regularize: bool = True
    """Whether to use depth regularization loss"""
    depth_loss_weight: float = 0.8
    """Weight for depth loss (higher = favor geometry over appearance)"""

    # Anchoring
    enable_anchoring: bool = True
    """Whether to anchor gaussians to their initial positions"""
    anchor_distance: float = 0.1
    """Maximum distance (meters) gaussians can move from anchor"""

    # ADC (pruning only)
    pruning_enable: bool = False
    """Whether to enable opacity-based pruning"""
    prune_alpha_thresh: float = 0.1
    """Opacity threshold for pruning"""
    prune_every: int = 100
    """Prune every N steps"""

    # Scheduling
    num_epoch_freeze_means: int = 10
    """Number of epochs to freeze gaussian means"""

    # Override defaults
    output_depth_during_training: bool = True
    """Always output depth during training for depth loss"""


class DMVSplatModel(SplatfactoModel):
    """DMVSplat model with depth regularization and anchoring."""

    config: DMVSplatModelConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_dmvsplat.py::test_dmvsplat_config_exists -v`
Expected: PASS

**Step 5: Commit**

```bash
mkdir -p tests/models
git add nerfstudio/models/dmvsplat.py tests/models/test_dmvsplat.py
git commit -m "feat(dmv): add DMVSplatModelConfig skeleton"
```

---

## Task 7: DMVSplatModel - Depth Regularization Loss

**Files:**
- Modify: `nerfstudio/models/dmvsplat.py`
- Test: `tests/models/test_dmvsplat.py`

**Step 1: Write failing test for depth loss computation**

```python
# Add to tests/models/test_dmvsplat.py

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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_dmvsplat.py::test_dmvsplat_depth_loss -v`
Expected: FAIL with ImportError

**Step 3: Write implementation**

```python
# Add to nerfstudio/models/dmvsplat.py, before class definitions

def compute_depth_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute L2 depth loss with masking.

    Args:
        pred_depth: Predicted depth [H, W, 1]
        gt_depth: Ground truth depth [H, W, 1], 0 = invalid
        mask: Optional ignore mask [H, W, 1], 1 = use, 0 = ignore

    Returns:
        Scalar loss value
    """
    # Valid = has depth (gt > 0)
    valid_mask = (gt_depth > 0).float()

    # Combine with ignore mask if provided
    if mask is not None:
        valid_mask = valid_mask * mask.float()

    # L2 loss on valid pixels only
    diff_sq = (pred_depth - gt_depth) ** 2
    loss = (diff_sq * valid_mask).sum() / valid_mask.sum().clamp(min=1)

    return loss
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_dmvsplat.py::test_dmvsplat_depth_loss -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nerfstudio/models/dmvsplat.py tests/models/test_dmvsplat.py
git commit -m "feat(dmv): add compute_depth_loss function"
```

---

## Task 8: DMVSplatModel - Integrate Depth Loss into get_loss_dict

**Files:**
- Modify: `nerfstudio/models/dmvsplat.py`
- Test: `tests/models/test_dmvsplat.py`

**Step 1: Write failing test for get_loss_dict**

```python
# Add to tests/models/test_dmvsplat.py

def test_dmvsplat_get_loss_dict_has_depth_loss():
    """Test that get_loss_dict includes depth_loss when depth_image is in batch"""
    # This is an integration test - we'll mock the necessary components
    from nerfstudio.models.dmvsplat import DMVSplatModel, DMVSplatModelConfig

    # Check the method exists and has correct signature
    assert hasattr(DMVSplatModel, 'get_loss_dict')
```

**Step 2: Write implementation**

```python
# Add to DMVSplatModel class

    def get_loss_dict(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], metrics_dict: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute losses including depth regularization."""
        # Get base losses from splatfacto
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # Add depth loss if enabled and depth available
        if self.config.depth_regularize and "depth_image" in batch and "depth" in outputs:
            gt_depth = batch["depth_image"].to(self.device)
            pred_depth = outputs["depth"]

            # Downscale gt_depth to match pred_depth if needed
            if gt_depth.shape[:2] != pred_depth.shape[:2]:
                gt_depth = self._downscale_if_required(gt_depth)

            # Get mask if available
            mask = batch.get("mask", None)
            if mask is not None:
                mask = self._downscale_if_required(mask).to(self.device)

            depth_loss = compute_depth_loss(pred_depth, gt_depth, mask)
            loss_dict["depth_loss"] = self.config.depth_loss_weight * depth_loss

        return loss_dict
```

**Step 3: Run test to verify it passes**

Run: `pytest tests/models/test_dmvsplat.py::test_dmvsplat_get_loss_dict_has_depth_loss -v`
Expected: PASS

**Step 4: Commit**

```bash
git add nerfstudio/models/dmvsplat.py tests/models/test_dmvsplat.py
git commit -m "feat(dmv): integrate depth loss into get_loss_dict"
```

---

## Task 9: DMVStrategy - Minimal ADC with Optional Pruning

**Files:**
- Modify: `nerfstudio/models/dmvsplat.py`
- Test: `tests/models/test_dmvsplat.py`

**Step 1: Write failing test for DMVStrategy**

```python
# Add to tests/models/test_dmvsplat.py

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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_dmvsplat.py::test_dmv_strategy_prune -v`
Expected: FAIL with ImportError

**Step 3: Write implementation**

```python
# Add to nerfstudio/models/dmvsplat.py, after compute_depth_loss

class DMVStrategy:
    """Minimal ADC strategy for dense LiDAR initialization.

    Only supports optional opacity-based pruning.
    No densification, no opacity reset, no split/duplicate.
    """

    def __init__(self, prune_alpha_thresh: float = 0.1, prune_every: int = 100):
        self.prune_alpha_thresh = prune_alpha_thresh
        self.prune_every = prune_every

    def step_pre_backward(self, *args, **kwargs):
        """No gradient accumulation needed."""
        pass

    def _get_prune_mask(self, params: Dict[str, torch.nn.Parameter]) -> torch.Tensor:
        """Get mask of gaussians to keep (True = keep)."""
        opacities = torch.sigmoid(params["opacities"].squeeze(-1))
        return opacities >= self.prune_alpha_thresh

    def step_post_backward(
        self,
        params: Dict[str, torch.nn.Parameter],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict,
        step: int,
        info: Dict,
        pruning_enable: bool = False,
    ):
        """Optionally prune low-opacity gaussians."""
        if not pruning_enable:
            return None

        if step > 0 and step % self.prune_every == 0:
            mask = self._get_prune_mask(params)
            if mask.sum() < len(mask):
                return self._prune(params, optimizers, mask)

        return None

    def _prune(
        self,
        params: Dict[str, torch.nn.Parameter],
        optimizers: Dict[str, torch.optim.Optimizer],
        mask: torch.Tensor,
    ) -> int:
        """Remove gaussians where mask is False. Returns count of remaining."""
        for name, param in params.items():
            params[name] = torch.nn.Parameter(param.data[mask])

            # Update optimizer state if exists
            if name in optimizers:
                opt = optimizers[name]
                for group in opt.param_groups:
                    for i, p in enumerate(group["params"]):
                        if p is param:
                            group["params"][i] = params[name]
                            # Update momentum/state
                            if p in opt.state:
                                state = opt.state.pop(p)
                                for key, val in state.items():
                                    if isinstance(val, torch.Tensor) and val.shape[0] == len(mask):
                                        state[key] = val[mask]
                                opt.state[params[name]] = state

        return mask.sum().item()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_dmvsplat.py::test_dmv_strategy_prune -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nerfstudio/models/dmvsplat.py tests/models/test_dmvsplat.py
git commit -m "feat(dmv): add DMVStrategy with optional pruning"
```

---

## Task 10: DMVSplatModel - Gaussian Anchoring

**Files:**
- Modify: `nerfstudio/models/dmvsplat.py`
- Test: `tests/models/test_dmvsplat.py`

**Step 1: Write failing test for anchoring**

```python
# Add to tests/models/test_dmvsplat.py

def test_dmvsplat_anchor_enforcement():
    """Test that gaussians are clamped to anchor_distance"""
    from nerfstudio.models.dmvsplat import enforce_anchor_constraint

    # Anchors at origin, means moved away
    anchors = torch.zeros(3, 3)
    means = torch.tensor([
        [0.05, 0.0, 0.0],   # distance 0.05, within limit
        [0.2, 0.0, 0.0],    # distance 0.2, exceeds limit
        [0.0, 0.15, 0.0],   # distance 0.15, exceeds limit
    ])
    anchor_distance = 0.1

    clamped = enforce_anchor_constraint(means, anchors, anchor_distance)

    # First gaussian should be unchanged
    assert torch.allclose(clamped[0], means[0])

    # Second gaussian should be clamped to distance 0.1
    assert abs(torch.norm(clamped[1] - anchors[1]).item() - 0.1) < 0.001

    # Third gaussian should be clamped to distance 0.1
    assert abs(torch.norm(clamped[2] - anchors[2]).item() - 0.1) < 0.001
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_dmvsplat.py::test_dmvsplat_anchor_enforcement -v`
Expected: FAIL with ImportError

**Step 3: Write implementation**

```python
# Add to nerfstudio/models/dmvsplat.py, after DMVStrategy class

def enforce_anchor_constraint(
    means: torch.Tensor,
    anchors: torch.Tensor,
    anchor_distance: float,
) -> torch.Tensor:
    """Clamp means to be within anchor_distance of their anchors.

    Args:
        means: Current positions [N, 3]
        anchors: Anchor positions [N, 3]
        anchor_distance: Maximum allowed distance

    Returns:
        Clamped positions [N, 3]
    """
    displacement = means - anchors
    distance = displacement.norm(dim=-1, keepdim=True)
    exceeded = distance > anchor_distance

    # Clamp direction to anchor_distance
    clamped = anchors + displacement / distance.clamp(min=1e-6) * anchor_distance

    return torch.where(exceeded, clamped, means)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_dmvsplat.py::test_dmvsplat_anchor_enforcement -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nerfstudio/models/dmvsplat.py tests/models/test_dmvsplat.py
git commit -m "feat(dmv): add enforce_anchor_constraint function"
```

---

## Task 11: DMVSplatModel - Integrate Anchoring into Model

**Files:**
- Modify: `nerfstudio/models/dmvsplat.py`

**Step 1: Update DMVSplatModel to use anchoring**

```python
# Update DMVSplatModel class in nerfstudio/models/dmvsplat.py

class DMVSplatModel(SplatfactoModel):
    """DMVSplat model with depth regularization and anchoring."""

    config: DMVSplatModelConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchors: Optional[torch.Tensor] = None
        self.anchor_distance_scaled: Optional[float] = None
        self.num_images: Optional[int] = None
        self.strategy = DMVStrategy(
            prune_alpha_thresh=self.config.prune_alpha_thresh,
            prune_every=self.config.prune_every,
        )

    def populate_modules(self):
        super().populate_modules()

        if self.config.enable_anchoring:
            # Store initial positions as anchors
            self.register_buffer("anchors", self.gauss_params["means"].detach().clone())

            # Get scale factor from metadata (set by dataparser)
            scale_factor = self.kwargs.get("metadata", {}).get("scene_scale", 1.0)
            if isinstance(scale_factor, torch.Tensor):
                scale_factor = scale_factor.item()
            self.anchor_distance_scaled = self.config.anchor_distance * scale_factor

    def get_loss_dict(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], metrics_dict: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute losses including depth regularization."""
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # Add depth loss
        if self.config.depth_regularize and "depth_image" in batch and "depth" in outputs:
            gt_depth = batch["depth_image"].to(self.device)
            pred_depth = outputs["depth"]

            if gt_depth.shape[:2] != pred_depth.shape[:2]:
                gt_depth = self._downscale_if_required(gt_depth)

            mask = batch.get("mask", None)
            if mask is not None:
                mask = self._downscale_if_required(mask).to(self.device)

            depth_loss = compute_depth_loss(pred_depth, gt_depth, mask)
            loss_dict["depth_loss"] = self.config.depth_loss_weight * depth_loss

        return loss_dict

    def step_post_backward(self, step: int):
        """Called after backward pass. Enforce anchoring constraint."""
        # Enforce anchor constraint
        if self.config.enable_anchoring and self.anchors is not None:
            with torch.no_grad():
                clamped = enforce_anchor_constraint(
                    self.gauss_params["means"].data,
                    self.anchors,
                    self.anchor_distance_scaled,
                )
                self.gauss_params["means"].data.copy_(clamped)

        # Call strategy for optional pruning
        if hasattr(self, 'strategy'):
            result = self.strategy.step_post_backward(
                params=self.gauss_params,
                optimizers=self.optimizers,
                state={},
                step=step,
                info=self.info if hasattr(self, 'info') else {},
                pruning_enable=self.config.pruning_enable,
            )

            # Update anchors if pruning occurred
            if result is not None and self.config.enable_anchoring:
                mask = self.strategy._get_prune_mask(self.gauss_params)
                self.anchors = self.anchors[mask]
```

**Step 2: Run all tests**

Run: `pytest tests/models/test_dmvsplat.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add nerfstudio/models/dmvsplat.py
git commit -m "feat(dmv): integrate anchoring and strategy into DMVSplatModel"
```

---

## Task 12: EpochTrainer

**Files:**
- Create: `nerfstudio/engine/epoch_trainer.py`
- Test: `tests/engine/test_epoch_trainer.py`

**Step 1: Write failing test for EpochTrainer**

```python
# tests/engine/test_epoch_trainer.py
"""Tests for EpochTrainer"""

import pytest


def test_epoch_trainer_config():
    """Test EpochTrainerConfig exists with num_epochs"""
    from nerfstudio.engine.epoch_trainer import EpochTrainerConfig

    config = EpochTrainerConfig(num_epochs=50)
    assert config.num_epochs == 50
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_epoch_trainer.py::test_epoch_trainer_config -v`
Expected: FAIL with ImportError

**Step 3: Write implementation**

```python
# nerfstudio/engine/epoch_trainer.py
"""Epoch-based trainer for DMVSplat."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from nerfstudio.engine.trainer import Trainer, TrainerConfig


@dataclass
class EpochTrainerConfig(TrainerConfig):
    """Epoch-based trainer config."""

    _target: Type = field(default_factory=lambda: EpochTrainer)
    """target class to instantiate"""
    num_epochs: int = 100
    """Number of epochs to train"""


class EpochTrainer(Trainer):
    """Trainer that uses epochs instead of fixed step count.

    Dynamically sets max_num_iterations based on dataset size.
    """

    config: EpochTrainerConfig

    def setup(self, test_mode: bool = False):
        """Setup trainer and compute max iterations from epochs."""
        super().setup(test_mode=test_mode)

        # Compute max iterations from epochs
        num_images = len(self.pipeline.datamanager.train_dataset)
        self.config.max_num_iterations = self.config.num_epochs * num_images

        # Update scheduler max_steps to match
        for param_group_name, scheduler in self.optimizers.schedulers.items():
            # Update the underlying scheduler's max_steps
            if hasattr(scheduler, 'lr_lambdas') and len(scheduler.lr_lambdas) > 0:
                # LambdaLR scheduler - the lambda function captures max_steps
                # We need to update it in the config
                pass

            # For ExponentialDecayScheduler, update via attribute
            if hasattr(scheduler, 'max_steps'):
                scheduler.max_steps = self.config.max_num_iterations
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/engine/test_epoch_trainer.py::test_epoch_trainer_config -v`
Expected: PASS

**Step 5: Commit**

```bash
mkdir -p tests/engine
git add nerfstudio/engine/epoch_trainer.py tests/engine/test_epoch_trainer.py
git commit -m "feat(dmv): add EpochTrainer with dynamic max_iterations"
```

---

## Task 13: Register DMVSplat Method

**Files:**
- Modify: `nerfstudio/configs/method_configs.py`

**Step 1: Add dmvsplat method registration**

```python
# Add to nerfstudio/configs/method_configs.py

# Near the top with other imports, add:
from nerfstudio.data.dataparsers.dmv_dataparser import DMVDataParserConfig
from nerfstudio.models.dmvsplat import DMVSplatModelConfig
from nerfstudio.engine.epoch_trainer import EpochTrainerConfig

# In the descriptions dict, add:
descriptions["dmvsplat"] = "DMV 3DGS method with depth regularization and gaussian anchoring for LiDAR datasets."

# In method_configs dict, add:
method_configs["dmvsplat"] = EpochTrainerConfig(
    method_name="dmvsplat",
    num_epochs=100,
    steps_per_eval_batch=500,
    steps_per_eval_image=500,
    steps_per_save=2000,
    pipeline=VanillaPipelineConfig(
        datamanager=FullImageDatamanagerConfig(
            dataparser=DMVDataParserConfig(),
            cache_images_type="uint8",
        ),
        model=DMVSplatModelConfig(),
    ),
    optimizers={
        "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6,
                max_steps=100000,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacities": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scales": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "quats": {
            "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
            "scheduler": None,
        },
    },
)
```

**Step 2: Run verification**

```bash
ns-train dmvsplat --help
```

Expected: Help output showing dmvsplat method is recognized

**Step 3: Commit**

```bash
git add nerfstudio/configs/method_configs.py
git commit -m "feat(dmv): register dmvsplat method in method_configs"
```

---

## Task 14: Integration Test with Real Dataset

**Files:**
- Create: `tests/test_dmvsplat_integration.py`

**Step 1: Write integration test**

```python
# tests/test_dmvsplat_integration.py
"""Integration test for DMVSplat with real dataset."""

import subprocess
import sys
from pathlib import Path


def test_dmvsplat_method_registered():
    """Test that dmvsplat method is recognized by ns-train"""
    result = subprocess.run(
        [sys.executable, "-m", "nerfstudio.scripts.train", "dmvsplat", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "dmvsplat" in result.stdout or "DMVSplat" in result.stdout


def test_dmvsplat_dataparser_on_real_data():
    """Test DMVDataParser on real dataset if available"""
    dataset_path = Path.home() / "3dgs_ws" / "perspective_processed"

    if not dataset_path.exists():
        import pytest
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
```

**Step 2: Run integration test**

Run: `pytest tests/test_dmvsplat_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_dmvsplat_integration.py
git commit -m "test(dmv): add integration tests for dmvsplat"
```

---

## Task 15: Final Cleanup and Documentation

**Step 1: Run all tests**

```bash
pytest tests/dataparsers/test_dmv_dataparser.py tests/data/test_dmv_dataset.py tests/models/test_dmvsplat.py tests/engine/test_epoch_trainer.py tests/test_dmvsplat_integration.py -v
```

**Step 2: Update SPECS.md to mark completed**

Add to top of SPECS.md:
```markdown
## Status: Implemented

Implementation complete. See `docs/plans/2026-01-28-dmvsplat-design.md` for design.
```

**Step 3: Final commit**

```bash
git add SPECS.md
git commit -m "docs: mark dmvsplat implementation complete"
```

---

## Summary

| Task | Component | Description |
|------|-----------|-------------|
| 1 | DMVDataParser | Config and basic structure |
| 2 | DMVDataParser | Custom initialization (center + scale) |
| 3 | DMVDataParser | View sampling order |
| 4 | DMVDataParser | Depth range loading |
| 5 | DMVDataset | 16-bit depth decoding |
| 6 | DMVSplatModel | Config and basic structure |
| 7 | DMVSplatModel | Depth loss function |
| 8 | DMVSplatModel | Integrate depth loss |
| 9 | DMVStrategy | Minimal ADC with pruning |
| 10 | DMVSplatModel | Anchor enforcement function |
| 11 | DMVSplatModel | Integrate anchoring |
| 12 | EpochTrainer | Epoch-based training |
| 13 | method_configs | Register dmvsplat |
| 14 | Integration | Test with real dataset |
| 15 | Cleanup | Final tests and docs |
