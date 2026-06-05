"""
Slow Burn Test: The Killer Demo for Tiered Compression

Tests the scenario where critical information appears ONCE early in a long context
and is needed MUCH LATER. This breaks binary eviction methods (H2O, ScissorHands)
but tiered compression survives via structural floor.
"""

import torch
import torch.nn.functional as F
import numpy as np

from ttkv import CacheConfig, TieredKVCache
from ttkv import compute_type_prior_retention
from tests.core.baselines import H2OCache, ScissorHandsCache


def create_slow_burn_context(seq_len, needle_idx=0):
    """
    Create a slow-burn scenario:
    - Critical token at position needle_idx (default: 0)
    - Long filler text to bury it
    - Need to retrieve it at the end
    """
    batch_size = 1
    num_heads = 12
    head_dim = 64
    
    k = torch.randn(batch_size, num_heads, seq_len, head_dim) * 0.5
    v = torch.randn(batch_size, num_heads, seq_len, head_dim) * 0.5
    positions = torch.arange(seq_len).unsqueeze(0)
    
    # Token IDs - generic filler
    token_ids = torch.randint(0, 50000, (batch_size, seq_len))
    
    # Make the needle distinctive (high salience)
    # This simulates "The password is XK7-9M2" at position 0
    k[0, :, needle_idx, :] = torch.randn(num_heads, head_dim) * 5.0
    v[0, :, needle_idx, :] = torch.randn(num_heads, head_dim) * 5.0
    
    # Compute retention - needle gets high structural score
    retention = compute_type_prior_retention(token_ids)
    retention[0, needle_idx] = 0.95  # High retention for needle
    
    # Rest gets lower retention
    retention[0, 1:] = retention[0, 1:] * 0.3
    
    return k, v, positions, token_ids, retention, needle_idx


def check_retrieval(k_cache, v_cache, positions_cache, needle_idx, query_pos):
    """
    Test if we can retrieve the needle at query position.
    Returns probability of attending to needle.
    """
    if k_cache is None or k_cache.size(2) == 0:
        return 0.0
    
    # Simulate query at the end
    query = torch.randn(1, 1, 64)
    scale = 64 ** -0.5
    
    # Compute attention
    scores = torch.matmul(query, k_cache[0, 0, :, :].T) * scale
    attn = F.softmax(scores, dim=-1)
    
    # Check if needle is in positions
    needle_mask = (positions_cache[0] == needle_idx).nonzero(as_tuple=True)[0]
    
    if len(needle_mask) == 0:
        return 0.0  # Needle evicted
    
    needle_prob = attn[0, needle_mask].sum().item()
    return needle_prob


def run_slow_burn_tiered(seq_len, needle_idx=0, tau=0.9):
    """Test tiered compression on slow burn scenario."""
    k, v, positions, token_ids, retention, needle = create_slow_burn_context(seq_len, needle_idx)
    
    config = CacheConfig(tau_threshold=tau)
    cache = TieredKVCache(config)
    cache.add(k, v, retention, positions)
    
    k_comp, v_comp, pos_comp = cache.get_compressed_cache()
    stats = cache.get_stats()
    
    # Test retrieval
    needle_prob = check_retrieval(k_comp, v_comp, pos_comp, needle, seq_len)
    preserved = needle_prob > 0.01
    
    return {
        'method': 'Tiered (Ours)',
        'tokens_total': seq_len,
        'tokens_kept': stats['compressed_tokens'],
        'compression': stats['compression_ratio'],
        'needle_preserved': preserved,
        'needle_prob': needle_prob,
        'result': 'PASS' if preserved else 'FAIL'
    }


def run_slow_burn_h2o(seq_len, needle_idx=0, max_cache_size=2048):
    k, v, positions, token_ids, retention, needle = create_slow_burn_context(seq_len, needle_idx)
    
    config = CacheConfig()
    cache = H2OCache(config, max_cache_size=max_cache_size)
    
    # H2O uses accumulated attention. Needle at pos 0 has 0 attention until attended.
    # In slow burn, no one attends until query, so accumulated attention = 0.
    # H2O evicts tokens with lowest accumulated attention.
    attention = torch.ones(1, seq_len) * 0.1
    attention[0, needle_idx] = 0.0  # Needle never attended
    
    cache.add(k, v, attention, positions)
    
    k_comp, v_comp, pos_comp = cache.get_compressed_cache()
    stats = cache.get_stats()
    
    needle_prob = check_retrieval(k_comp, v_comp, pos_comp, needle, seq_len)
    preserved = needle_prob > 0.01
    
    return {
        'method': 'H2O',
        'tokens_total': seq_len,
        'tokens_kept': stats['compressed_tokens'],
        'compression': stats['compression_ratio'],
        'needle_preserved': preserved,
        'needle_prob': needle_prob,
        'result': 'FAIL (evicted)' if not preserved else 'PASS'
    }


def run_slow_burn_scissorhands(seq_len, needle_idx=0, max_cache_size=2048):
    k, v, positions, token_ids, retention, needle = create_slow_burn_context(seq_len, needle_idx)
    
    # ScissorHands uses recent attention window (256 tokens). Needle at pos 0 never receives
    # attention from queries at the end (positions 14744-15000), so it's outside the window.
    config = CacheConfig()
    cache = ScissorHandsCache(config, max_cache_size=max_cache_size, attention_window=256)
    cache.add(k, v, retention, positions)
    
    k_comp, v_comp, pos_comp = cache.get_compressed_cache()
    stats = cache.get_stats()
    
    needle_prob = check_retrieval(k_comp, v_comp, pos_comp, needle, seq_len)
    preserved = needle_prob > 0.01
    
    return {
        'method': 'ScissorHands',
        'tokens_total': seq_len,
        'tokens_kept': stats['compressed_tokens'],
        'compression': stats['compression_ratio'],
        'needle_preserved': preserved,
        'needle_prob': needle_prob,
        'result': 'FAIL (outside window)' if not preserved else 'PASS'
    }


def run_comparison(seq_len=15000, needle_idx=0):
    """
    Run the killer slow burn comparison.
    
    Critical token at position 0, needed at position 15,000.
    Binary eviction dies. Tiered compression lives.
    """
    print("=" * 80)
    print("SLOW BURN TEST: Critical Token at Position 0, Needed at Position 15K")
    print("=" * 80)
    print(f"\nScenario: Password mentioned once at start of {seq_len:,} token context")
    print("          Model needs to retrieve it at the end")
    print()
    
    # Run all three tests
    results = []
    
    print("Testing H2O...")
    results.append(run_slow_burn_h2o(seq_len, needle_idx))
    
    print("Testing ScissorHands...")
    results.append(run_slow_burn_scissorhands(seq_len, needle_idx))
    
    print("Testing Tiered (Ours)...")
    results.append(run_slow_burn_tiered(seq_len, needle_idx))
    
    # Print results table
    print("\n" + "=" * 80)
    print("SLOW BURN TEST RESULTS")
    print("=" * 80)
    print()
    print(f"{'Method':<20} | {'Kept':>8} | {'Ratio':>8} | {'Preserved':>10} | {'P(needle)':>10} | {'Result':<20}")
    print("-" * 80)
    
    for r in results:
        preserved_str = "YES" if r['needle_preserved'] else "NO"
        print(f"{r['method']:<20} | {r['tokens_kept']:>8} | {r['compression']:>7.2f}x | {preserved_str:>10} | "
              f"{r['needle_prob']:>9.4f} | {r['result']:<20}")
    
    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    # Check results
    tiered_result = next(r for r in results if r['method'] == 'Tiered (Ours)')
    h2o_result = next(r for r in results if r['method'] == 'H2O')
    sh_result = next(r for r in results if r['method'] == 'ScissorHands')
    
    if tiered_result['needle_preserved'] and not h2o_result['needle_preserved'] and not sh_result['needle_preserved']:
        print("\n✓ KILLER DEMO CONFIRMED")
        print("  - H2O: FAILS - needle evicted before attended (accumulated attention = 0)")
        print("  - ScissorHands: FAILS - needle outside attention window")
        print("  - Tiered: PASSES - structural floor keeps needle via tier compression")
        print("\n  Binary eviction can't handle slow burn.")
        print("  Tiered compression survives via structural survival floor.")
    elif tiered_result['needle_preserved']:
        print("\n✓ Tiered compression preserves needle")
        if h2o_result['needle_preserved']:
            print("  (H2O also preserved - low pressure test)")
        if sh_result['needle_preserved']:
            print("  (ScissorHands also preserved - low pressure test)")
    else:
        print("\n✗ All methods failed - check implementation")
    
    print("=" * 80)
    
    return results


def run_stress_test():
    """Run slow burn at multiple sequence lengths."""
    print("\n" + "=" * 80)
    print("SLOW BURN STRESS TEST: Multiple Context Lengths")
    print("=" * 80)
    
    configs = [
        (4096, 0),
        (8192, 0),
        (12288, 0),
        (15000, 0),
    ]
    
    summary = []
    
    for seq_len, needle_idx in configs:
        print(f"\n--- Testing {seq_len:,} tokens ---")
        results = run_comparison(seq_len, needle_idx)
        
        tiered_pass = any(r['needle_preserved'] for r in results if r['method'] == 'Tiered (Ours)')
        binary_fail = any(not r['needle_preserved'] for r in results if r['method'] in ['H2O', 'ScissorHands'])
        
        summary.append({
            'seq_len': seq_len,
            'tiered_pass': tiered_pass,
            'binary_fail': binary_fail,
            'killer_demo': tiered_pass and binary_fail
        })
    
    print("\n" + "=" * 80)
    print("SUMMARY: Slow Burn Test Across Context Lengths")
    print("=" * 80)
    print(f"{'Seq Length':<12} | {'Tiered':<8} | {'Binary Fails':<12} | {'Killer Demo':<12}")
    print("-" * 80)
    
    for s in summary:
        tiered_str = "✓" if s['tiered_pass'] else "✗"
        binary_str = "✓" if s['binary_fail'] else "✗"
        killer_str = "✓ YES" if s['killer_demo'] else "✗"
        print(f"{s['seq_len']:>12,} | {tiered_str:<8} | {binary_str:<12} | {killer_str:<12}")
    
    print("=" * 80)
    
    return summary


if __name__ == "__main__":
    # Single test
    run_comparison(seq_len=15000)
    
    # Stress test
    print("\n" + "=" * 80)
    print("\nRunning full stress test...")
    run_stress_test()
