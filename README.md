# TTKV - Three Tiered Key Value Cache Compression

[![PyPI version](https://badge.fury.io/py/ttkv.svg)](https://badge.fury.io/py/ttkv)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Three-tier progressive compression (full → 4:1 → 16:1) with structural survival floor.**

TTKV is a PyTorch library that enables long-context inference on consumer GPUs by compressing the KV cache rather than evicting tokens. It solves the "slow-burn problem" where critical tokens mentioned early get evicted before the model attends to them.

## Key Features

- 🎯 **Three-Tier Compression**: Uncompressed (Tier 0) → 4:1 (Tier 1) → 16:1 (Tier 2)
- 🛡️ **Structural Survival Floor**: Rare-but-critical tokens survive until attended
- 🧠 **Attention-Guided Scoring**: Uses model's own attention patterns
- ⚡ **Consumer GPU Ready**: Enables 16K context on RTX 3060 (12GB)
- 🔌 **Transformers Compatible**: Works with HuggingFace models

## Installation

```bash
pip install ttkv
```

### Optional Dependencies

```bash
# For 4-bit quantization support
pip install ttkv[quantization]

# For visualization
pip install ttkv[viz]

# Development dependencies
pip install ttkv[dev]
```

## Quick Start

```python
from ttkv import TieredKVCache, CacheConfig
import torch

# Configure the cache
config = CacheConfig(
    hidden_dim=768,
    num_heads=12,
    tier0_size=256,      # Recent tokens (uncompressed)
    tier1_size=2048,     # Middle tier
    tier1_compression=4,  # 4:1 compression
    tier2_compression=16, # 16:1 compression
    tau_threshold=0.9
)

# Create cache
cache = TieredKVCache(config)

# Use with your model
k = torch.randn(1, 12, 8192, 64)  # [batch, heads, seq, head_dim]
v = torch.randn(1, 12, 8192, 64)
retention = torch.rand(1, 8192)    # Salience scores
positions = torch.arange(8192).unsqueeze(0)

cache.add(k, v, retention, positions)

# Get compressed cache
k_comp, v_comp, pos_comp = cache.get_compressed_cache()
stats = cache.get_stats()
print(f"Compression: {stats['compression_ratio']:.2f}x")
```

## Performance

| Context | GPU | Baseline | TTKV | Compression |
|---------|-----|----------|------|-------------|
| 16K tokens | RTX 3060 (12GB) | OOM | ✅ Works | **7.36x** |

Quality: <0.2% perplexity increase at 7.36x compression on GPT-2.

## Documentation

- [Paper](paper/main.tex) - Full technical details
- [API Reference](#api-reference) - Below
- [Examples](tests/) - Test scripts and experiments

## API Reference

### Core Components

#### `CacheConfig`
Configuration for tiered KV cache.

```python
config = CacheConfig(
    hidden_dim=768,
    num_heads=12,
    head_dim=64,
    tier0_size=256,
    tier1_size=2048,
    tier1_compression=4,
    tier2_compression=16,
    tau_threshold=0.8
)
```

#### `TieredKVCache`
Main cache implementation.

```python
cache = TieredKVCache(config)
cache.add(k, v, retention, positions)
k_comp, v_comp, pos_comp = cache.get_compressed_cache()
stats = cache.get_stats()
```

### Attention-Guided Components

#### `AttentionGuidedScorer`
Learns token importance from attention patterns.

```python
from ttkv import AttentionGuidedScorer

scorer = AttentionGuidedScorer(ema_decay=0.95, structural_floor=0.1)
scorer.update_from_attention(attention_weights, query_position, token_id)
salience = scorer.get_salience_scores(seq_len)
```

#### `AttentionGuidedWrapper`
Full wrapper for models.

```python
from ttkv import AttentionGuidedWrapper

wrapper = AttentionGuidedWrapper(model, tokenizer, cache_config)
result, stats = wrapper.generate_with_attention_guidance(input_ids)
```

## Project Structure

```
ttkv/
├── src/ttkv/
│   ├── __init__.py          # Package exports
│   ├── core.py              # TieredKVCache, CacheConfig
│   ├── attention_scorer.py  # Attention-guided scoring
│   └── type_prior.py        # Structural priors
├── tests/
│   ├── core/               # Core functionality tests
│   └── experiments/        # Paper experiments
├── paper/
│   └── main.tex            # LaTeX source
└── README.md               # This file
```

## Citation

If you use TTKV in your research:

```bibtex
@software{ttkv2026,
  title={TTKV: Salience-Aware Tiered KV Cache Compression},
  author={Pérez Muñiz, Zabdiel},
  year={2026},
  url={https://github.com/zabwie/ttkv}
}
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)

## Contact

- GitHub Issues: [github.com/zabwie/ttkv/issues](https://github.com/zabwie/ttkv/issues)
- Email: zabdielperez00@gmail.com