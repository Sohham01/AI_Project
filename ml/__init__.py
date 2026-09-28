"""ML Training Framework - classical baseline training, evaluation, and artifact management.

This package orchestrates the already-completed NLP classifiers without modifying
them.  It does not implement new models; it loads annotation data, creates
reproducible train/validation/test splits, trains the existing baselines through
their public APIs, evaluates them consistently, and saves artifacts and metrics.

Current stage: classical ML baseline infrastructure.
ANN/CNN/transformers are NOT implemented yet.

See docs/ml_training_pipeline.md for the full design.
"""