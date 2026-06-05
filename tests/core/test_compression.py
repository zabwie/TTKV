"""Real pytest tests for TieredKVCache core compression logic."""
import torch
import pytest
from ttkv import CacheConfig, TieredKVCache


class TestTieredKVCache:
    def test_empty_cache_returns_none(self):
        """Empty cache (nothing added) should return (None, None, None), not crash."""
        config = CacheConfig()
        cache = TieredKVCache(config)
        k, v, pos = cache.get_compressed_cache()
        assert k is None
        assert v is None
        assert pos is None
        stats = cache.get_stats()
        assert stats['total_tokens'] == 0
        assert stats['compression_ratio'] == 1.0

    def test_add_and_compress_reduces_size(self):
        """Compression should reduce token count below the original."""
        config = CacheConfig(tau_threshold=0.5)
        cache = TieredKVCache(config)
        k = torch.randn(1, 12, 512, 64)
        v = torch.randn(1, 12, 512, 64)
        retention = torch.full((1, 512), 0.3)
        positions = torch.arange(512).unsqueeze(0)
        cache.add(k, v, retention, positions)
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()
        assert k_comp is not None
        assert k_comp.size(2) < 512  # compression happened
        assert k_comp.size(2) > 0    # not everything evicted

    def test_high_retention_tokens_survive(self):
        """Tokens with retention > tau must survive uncompressed."""
        config = CacheConfig(tau_threshold=0.8)
        cache = TieredKVCache(config)
        k = torch.randn(1, 12, 100, 64)
        v = torch.randn(1, 12, 100, 64)
        retention = torch.full((1, 100), 0.5)
        retention[0, 50] = 0.95  # this should survive as protected
        positions = torch.arange(100).unsqueeze(0)
        cache.add(k, v, retention, positions)
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()
        # Position 50 should be in the compressed output
        assert (pos_comp == 50).any(), "High-retention token was lost"

    def test_compress_deterministic(self):
        """Same input should give same output (no randomness in compression)."""
        config = CacheConfig(tau_threshold=0.5)
        k = torch.randn(1, 12, 256, 64)
        v = torch.randn(1, 12, 256, 64)
        retention = torch.rand(1, 256)
        positions = torch.arange(256).unsqueeze(0)

        cache1 = TieredKVCache(config)
        cache1.add(k.clone(), v.clone(), retention.clone(), positions.clone())
        k1, v1, p1 = cache1.get_compressed_cache()

        cache2 = TieredKVCache(config)
        cache2.add(k.clone(), v.clone(), retention.clone(), positions.clone())
        k2, v2, p2 = cache2.get_compressed_cache()

        assert torch.equal(k1, k2)
        assert torch.equal(p1, p2)

    def test_clear_resets_state(self):
        """Clear should reset all internal state."""
        config = CacheConfig()
        cache = TieredKVCache(config)
        k = torch.randn(1, 12, 100, 64)
        v = torch.randn(1, 12, 100, 64)
        retention = torch.rand(1, 100)
        positions = torch.arange(100).unsqueeze(0)
        cache.add(k, v, retention, positions)
        cache.clear()
        assert cache.total_tokens == 0
        assert len(cache.k_cache) == 0
        k_c, v_c, p_c = cache.get_compressed_cache()
        assert k_c is None  # empty cache returns None

    def test_stats_accurate(self):
        """get_stats should report accurate compression ratio."""
        config = CacheConfig(tau_threshold=0.5)
        cache = TieredKVCache(config)
        k = torch.randn(1, 12, 512, 64)
        v = torch.randn(1, 12, 512, 64)
        retention = torch.full((1, 512), 0.3)
        positions = torch.arange(512).unsqueeze(0)
        cache.add(k, v, retention, positions)
        stats = cache.get_stats()
        assert stats['total_tokens'] == 512
        assert stats['compressed_tokens'] < 512
        ratio = stats['compression_ratio']
        assert ratio > 1.0
