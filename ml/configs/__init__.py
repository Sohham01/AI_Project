"""
Task configuration and model registry for the ML training framework.
"""

from ml.configs.config import (
    TaskConfig,
    TaskType,
    ModelRegistry,
    TASK_METADATA,
    get_task_metadata,
)

__all__ = [
    "TaskConfig",
    "TaskType",
    "ModelRegistry",
    "TASK_METADATA",
    "get_task_metadata",
]
