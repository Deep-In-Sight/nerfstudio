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
