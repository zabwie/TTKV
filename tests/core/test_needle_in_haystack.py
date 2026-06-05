"""
Needle-in-Haystack Test: Retrieval Validation

Tests whether compression preserves critical information that appears
infrequently in long contexts. This catches catastrophic failures that
average metrics (perplexity) miss.
"""

import torch
import torch.nn.functional as F
import numpy as np

from ttkv import CacheConfig, TieredKVCache
from ttkv import compute_type_prior_retention


def create_needle_context(seq_len, needle_position, needle_token_idx):
    """
    Create a context with a single critical 'needle' token at specific position.
    The needle should have high retrieval probability if the system works.
    """
    num_heads = 12
    head_dim = 64
    
    # Fill with generic content (low retention)
    k = torch.randn(1, num_heads, seq_len, head_dim) * 0.5
    v = torch.randn(1, num_heads, seq_len, head_dim) * 0.5
    
    # Make needle distinctive (high magnitude = high attention weight)
    k[0, :, needle_token_idx, :] = torch.randn(num_heads, head_dim) * 5.0
    v[0, :, needle_token_idx, :] = torch.randn(num_heads, head_dim) * 5.0
    
    positions = torch.arange(seq_len).unsqueeze(0)
    token_ids = torch.randint(0, 50000, (1, seq_len))
    
    # Ensure needle has high retention
    retention = torch.full((1, seq_len), 0.3)
    retention[0, needle_token_idx] = 0.95  # High salience
    
    return k, v, positions, token_ids, retention, needle_token_idx


def run_needle_retrieval(seq_len, needle_position, tau):
    """Test if needle is retrievable after compression."""
    k, v, positions, token_ids, retention, needle_idx = create_needle_context(
        seq_len, needle_position, needle_position
    )
    
    # Baseline: no compression
    config_baseline = CacheConfig(tau_threshold=0.0)
    cache_baseline = TieredKVCache(config_baseline)
    cache_baseline.add(k, v, retention, positions)
    k_base, v_base, pos_base = cache_baseline.get_compressed_cache()
    
    # Compressed
    config = CacheConfig(tau_threshold=tau)
    cache = TieredKVCache(config)
    cache.add(k.clone(), v.clone(), retention.clone(), positions.clone())
    k_comp, v_comp, pos_comp = cache.get_compressed_cache()
    
    # Check if needle is preserved
    needle_preserved = (pos_comp == needle_idx).any() if pos_comp is not None else False
    
    # Compute attention to needle
    query = torch.randn(1, 1, 64)
    scale = 64 ** -0.5
    
    # Baseline attention
    if k_base is not None and k_base.size(2) > 0:
        scores_base = torch.matmul(query, k_base[0, 0, :, :].T) * scale
        attn_base = F.softmax(scores_base, dim=-1)
        needle_attn_base = attn_base[0, (pos_base[0] == needle_idx).nonzero(as_tuple=True)[0]]
        needle_prob_base = needle_attn_base.sum().item() if len(needle_attn_base) > 0 else 0.0
    else:
        needle_prob_base = 0.0
    
    # Compressed attention
    if k_comp is not None and k_comp.size(2) > 0:
        scores_comp = torch.matmul(query, k_comp[0, 0, :, :].T) * scale
        attn_comp = F.softmax(scores_comp, dim=-1)
        needle_positions = (pos_comp[0] == needle_idx).nonzero(as_tuple=True)[0]
        needle_attn_comp = attn_comp[0, needle_positions] if len(needle_positions) > 0 else torch.tensor([0.0])
        needle_prob_comp = needle_attn_comp.sum().item()
    else:
        needle_prob_comp = 0.0
    
    return {
        'seq_len': seq_len,
        'needle_idx': needle_idx,
        'tau': tau,
        'needle_preserved': needle_preserved,
        'needle_prob_baseline': needle_prob_base,
        'needle_prob_compressed': needle_prob_comp,
        'retention_score': retention[0, needle_idx].item(),
        'compression_ratio': cache.get_stats()['compression_ratio']
    }


def run_needle_tests():
    """Run comprehensive needle-in-haystack tests."""
    print("=" * 80)
    print("NEEDLE-IN-HAYSTACK TEST")
    print("Testing retrieval of critical tokens in long contexts")
    print("=" * 80)
    
    configs = [
        (2048, 1000, 0.9),
        (2048, 1500, 0.9),
        (4096, 2000, 0.9),
        (4096, 3500, 0.9),
        (8192, 4000, 0.9),
        (8192, 7000, 0.9),
    ]
    
    results = []
    
    for seq_len, needle_pos, tau in configs:
        print(f"\nTest: {seq_len} tokens, needle at position {needle_pos}, τ={tau}")
        
        result = run_needle_retrieval(seq_len, needle_pos, tau)
        results.append(result)
        
        status = "✓ PASS" if result['needle_preserved'] and result['needle_prob_compressed'] > 0.01 else "✗ FAIL"
        print(f"  Needle preserved: {result['needle_preserved']}")
        print(f"  Retention score: {result['retention_score']:.2f}")
        print(f"  Baseline P(needle): {result['needle_prob_baseline']:.4f}")
        print(f"  Compressed P(needle): {result['needle_prob_compressed']:.4f}")
        print(f"  Compression: {result['compression_ratio']:.2f}x")
        print(f"  Result: {status}")
    
    # Summary
    print("\n" + "=" * 80)
    print("NEEDLE TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results if r['needle_preserved'] and r['needle_prob_compressed'] > 0.01)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total} tests")
    
    if passed == total:
        print("\n✓ All needles preserved - critical information not dropped")
    else:
        print(f"\n✗ {total - passed} needles lost - catastrophic retrieval failure")
        failed = [r for r in results if not (r['needle_preserved'] and r['needle_prob_compressed'] > 0.01)]
        for r in failed:
            print(f"  - {r['seq_len']} tokens, pos {r['needle_idx']}: P={r['needle_prob_compressed']:.4f}")
    
    # Check worst-case behavior
    min_prob = min(r['needle_prob_compressed'] for r in results)
    print(f"\nWorst-case needle probability: {min_prob:.4f}")
    
    if min_prob > 0.01:
        print("✓ Worst-case retrieval acceptable")
    else:
        print("✗ Some needles lost - system unreliable")
    
    print("=" * 80)


if __name__ == "__main__":
    run_needle_tests()
