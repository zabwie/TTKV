# TTKV - Tiered KV Cache Compression

[![PyPI version](https://badge.fury.io/py/ttkv.svg)](https://badge.fury.io/py/ttkv)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Three-tier progressive KV cache compression (1:1 - 4:1 - 16:1) with structural retention floor. Reduces KV cache memory by 86% at 8K context on consumer GPUs without evicting tokens.

## Installation

```bash
pip install ttkv
```

Optional dependencies:

```bash
pip install ttkv[quantization]   # 4-bit quantization
pip install ttkv[viz]            # visualization
pip install ttkv[dev]            # development (pytest, mypy, black)
```

## Quick Start

```python
from ttkv import TieredKVCache, CacheConfig
import torch

config = CacheConfig(
    hidden_dim=896,
    num_heads=2,
    tier0_size=16,
    tier1_size=512,
    tier1_compression=4,
    tier2_compression=16,
    tau_threshold=0.9
)

cache = TieredKVCache(config)
k = torch.randn(1, 2, 8192, 64)
v = torch.randn(1, 2, 8192, 64)
retention = torch.rand(1, 8192)
positions = torch.arange(8192).unsqueeze(0)

cache.add(k, v, retention, positions)
k_comp, v_comp, pos_comp = cache.get_compressed_cache()
stats = cache.get_stats()
print(f"Compression: {stats['compression_ratio']:.2f}x")
```

## Key Features

- **Three-tier progressive compression**: Tokens degrade through tiers (full, 4:1, 16:1) rather than being evicted
- **Retention floor**: Structurally critical tokens (codes, numbers, proper nouns) receive minimum priority, preventing destruction before the attention signal evaluates their importance
- **Signal-to-noise optimization**: Aggressive compression improves retrieval by suppressing filler noise while the floor preserves critical signal
- **Auto-configuration**: `auto_config()` determines optimal tier geometry from protected token positions
- **Consumer GPU ready**: 7.4x compression at 8K tokens on RTX 3060 (12GB)

## Verified Performance

All measurements on Qwen2.5-0.5B-Instruct (FP16, RTX 3060 12GB):

| Context | Uncompressed KV | Compressed KV | Ratio | Memory Saved |
|---------|----------------|---------------|-------|-------------|
| 1,024 | 12.0 MB | 5.4 MB | 2.20x | 55% |
| 2,048 | 24.0 MB | 8.5 MB | 2.83x | 65% |
| 4,096 | 48.0 MB | 10.0 MB | 4.79x | 79% |
| 8,192 | 96.0 MB | 13.0 MB | 7.39x | 86% |

Slow-burn retrieval (needle at position 10, 3,600-token context, attention score < 0.005): TTKV achieves 9.0x compression with perfect recall, vs H2O's 7.1x best where it fails. See [paper](paper/main.tex) for full results.

## API Reference

### Core

```python
from ttkv import CacheConfig, TieredKVCache, SalienceScorer
```

`CacheConfig` configures tier geometry: `hidden_dim`, `num_heads`, `head_dim`, `tier0_size`, `tier1_size`, `tier1_compression`, `tier2_compression`, `tau_threshold`.

`TieredKVCache` is the main compression cache. Methods: `add(k, v, retention, positions)`, `get_compressed_cache()`, `get_stats()`, `clear()`.

### Auto-Configuration

```python
from ttkv import auto_config, build_retention_mask

retention = build_retention_mask(n_tokens, needle_ranges=[(10, 28)], query_ranges=[(3590, 3600)])
config = auto_config(n_tokens, retention > 0.9, target_signal_pct=3.5)
```

`build_retention_mask` creates a retention tensor marking critical token ranges. `auto_config` determines optimal `tier0_size` and `tier1_size` to achieve the target signal percentage.

### Attention-Guided Components

```python
from ttkv import AttentionGuidedScorer, AttentionGuidedWrapper, extract_attention_weights
```

`AttentionGuidedScorer` learns token importance from attention patterns (EMA-decayed). `AttentionGuidedWrapper` wraps HuggingFace models for attention-guided compression. `extract_attention_weights` extracts attention from model outputs.

### Structural Priors

```python
from ttkv import MockTypePriorClassifier, compute_type_prior_retention, create_mock_retention
```

Regex-based token classification into named entities, numbers, function words, and content words. Used as a lightweight alternative to the learned salience scorer.

## Project Structure

```
├── src/ttkv/
│   ├── __init__.py           # Package exports
│   ├── core.py               # TieredKVCache, CacheConfig, SalienceScorer
│   ├── attention_scorer.py   # AttentionGuidedScorer, AttentionGuidedWrapper
│   ├── type_prior.py         # MockTypePriorClassifier, retention utilities
│   ├── auto_config.py        # Auto-configuration of tier geometry
│   └── train.py              # Salience scorer training pipeline
├── tests/
│   ├── core/                 # Unit tests and baselines (H2O, ScissorHands)
│   ├── experiments/          # Paper experiments and ablation study
│   └── real_experiments/     # Real model benchmarks (Qwen2.5-0.5B)
├── paper/
│   └── main.tex              # Paper source
├── models/                   # Trained scorer checkpoints
├── results/                  # Benchmark outputs (JSON)
├── reproduce.sh              # One-command reproducibility
└── pyproject.toml            # Project configuration
```

## Reproducibility

```bash
pip install -e ".[dev]"
bash reproduce.sh
```

Requires: Python 3.9+, PyTorch 2.0+, RTX 3060 (12GB) or equivalent.

## Citation

```bibtex
@software{ttkv2026,
  title={TTKV: Tiered KV Cache Compression with Retention Floor},
  author={Perez Muniz, Zabdiel},
  year={2026},
  url={https://github.com/zabwie/TTKV}
}
```

## License

MIT - see [LICENSE](LICENSE).
