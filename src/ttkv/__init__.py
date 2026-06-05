"""TTKV - Three Tiered Key Value Cache Compression"""

__version__ = "1.0.0"
__author__ = "Zabdiel Pérez Muñiz"
__license__ = "MIT"

from .core import (
    CacheConfig,
    SalienceScorer,
    TieredKVCache,
)

from .attention_scorer import (
    AttentionGuidedScorer,
    AttentionBasedKVCache,
    AttentionGuidedWrapper,
    extract_attention_weights,
)

from .type_prior import (
    MockTypePriorClassifier,
    create_mock_retention,
    compute_type_prior_retention,
)

from .auto_config import auto_config, build_retention_mask

__all__ = [
    "CacheConfig",
    "SalienceScorer",
    "TieredKVCache",
    "AttentionGuidedScorer",
    "AttentionBasedKVCache",
    "AttentionGuidedWrapper",
    "extract_attention_weights",
    "MockTypePriorClassifier",
    "create_mock_retention",
    "compute_type_prior_retention",
    "auto_config",
    "build_retention_mask",
]
