"""
Generic trainer for the ML framework.

See ``training/trainer.py`` for the full implementation.
"""

from ml.training.trainer import (
    Trainer,
    TrainableModel,
    build_model,
    records_to_annotations,
    PipelineState,
    LoadedDataset,
)

__all__ = [
    "Trainer",
    "TrainableModel",
    "build_model",
    "records_to_annotations",
    "PipelineState",
    "LoadedDataset",
]
