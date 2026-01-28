"""Tests for EpochTrainer"""

import pytest


def test_epoch_trainer_config():
    """Test EpochTrainerConfig exists with num_epochs"""
    from nerfstudio.engine.epoch_trainer import EpochTrainerConfig

    config = EpochTrainerConfig(num_epochs=50)
    assert config.num_epochs == 50
