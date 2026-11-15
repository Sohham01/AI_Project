"""
Task configuration and model registry for the ML training framework.
"""

from ml.configs.config import (
    TaskConfig,
    TaskType,
    ModelRegistry,
    TASK_METADATA,
    get_task_metadata,
    project_root,
    default_artifact_directory,
    default_registry_path,
)

__all__ = [
    "TaskConfig",
    "TaskType",
    "ModelRegistry",
    "TASK_METADATA",
    "get_task_metadata",
    "project_root",
    "default_artifact_directory",
    "default_registry_path",
]
