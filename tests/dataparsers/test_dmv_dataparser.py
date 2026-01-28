"""Tests for DMV dataparser"""

import pytest
from pathlib import Path


def test_dmv_dataparser_config_exists():
    """Test that DMVDataParserConfig can be imported and instantiated"""
    from nerfstudio.data.dataparsers.dmv_dataparser import DMVDataParserConfig

    config = DMVDataParserConfig(data=Path("/tmp/test"))
    assert config.data == Path("/tmp/test")
    assert config.colmap_path == Path("sparse/0")
