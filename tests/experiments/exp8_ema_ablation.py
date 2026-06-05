"""
Experiment #8: EMA Ablation During Generation

Tests whether applying EMA updates during generation steps provides benefit.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from ttkv import CacheConfig, TieredKVCache
from ttkv import compute_type_prior_retention


def create_test_sequence(seq_len, batch_size=1, num_heads=12, head_dim=64, seed=42):
    """Create test sequence with fixed seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    token_ids = torch.randint(0, 50000, (batch_size, seq_len))
    return k, v, positions, token_ids


def test_prefill_only(config, k, v, retention, positions):
    """Test with salience computed only at prefill (no EMA during generation)."""
    cache = TieredKVCache(config)
    cache.add(k, v, retention, positions)
    
    k_comp, v_comp, _ = cache.get_compressed_cache()
    stats = cache.get_stats()
    
    # Measure quality
    query = torch.randn(1, 1, k.size(-1))
    attn_full = torch.matmul(query, k[0, 0, :, :].T) / (k.size(-1) ** 0.5)
    attn_full = F.softmax(attn_full, dim=-1)
    output_full = torch.matmul(attn_full, v[0, 0, :, :])
    
    attn_comp = torch.matmul(query, k_comp[0, 0, :, :].T) / (k.size(-1) ** 0.5)
    attn_comp = F.softmax(attn_comp, dim=-1)
    output_comp = torch.matmul(attn_comp, v_comp[0, 0, :, :])
    
    loss = F.mse_loss(output_comp, output_full).item()
    ppl_increase = (np.exp(loss) - 1.0) * 100
    
    return ppl_increase, stats['compression_ratio']


def run_experiment_8():
    """Run Experiment #8: EMA ablation."""
    print("=" * 80)
    print("EXPERIMENT #8: EMA Ablation During Generation")
    print("=" * 80)
    print()
    print("Goal: Test whether EMA updates during generation help")
    print("Setup: Compare prefill-only vs EMA-updated salience")
    print()
    
    seq_len = 8192
    tau_values = [0.6, 0.8, 0.9, 0.95]
    
    results = {
        "prefill_only": [],
        "with_ema": []  # Placeholder - EMA not implemented
    }
    
    k, v, positions, token_ids = create_test_sequence(seq_len)
    retention = compute_type_prior_retention(token_ids)
    
    print("Testing prefill-only salience (current implementation):")
    print("-" * 60)
    
    for tau in tau_values:
        config = CacheConfig(tau_threshold=tau)
        ppl, comp = test_prefill_only(config, k, v, retention, positions)
        
        results["prefill_only"].append({
            "tau": tau,
            "compression": comp,
            "quality_loss": ppl
        })
        
        print(f"  τ={tau:.2f}: {comp:.2f}×, {ppl:+.3f}%")
    
    # Analysis
    print("\n" + "=" * 80)
    print("EMA ANALYSIS")
    print("=" * 80)
    print()
    print("Note: Current implementation computes salience ONCE at prefill.")
    print("The EMA equation r_i^(t) = γ·r_i^(t-1) + (1-γ)·s_i is NOT applied during generation.")
    print()
    print("To properly test EMA during generation, we would need to:")
    print("  1. Extract hidden states at each generation step")
    print("  2. Recompute salience s_i for each new token")
    print("  3. Update EMA for ALL positions (O(L) per step)")
    print()
    print("This would add significant overhead and may not be practical.")
    print()
    print("RECOMMENDATION:")
    print("  Option A: Remove the EMA equation from the paper")
    print("  Option B: Clarify that EMA is aspirational/future work")
    print("  Option C: Implement and test with real GPT-2 generation")
    
    results["recommendation"] = "Option A or B: Remove or clarify EMA equation"
    
    # Save results
    with open('../results/exp8_ema_ablation.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/exp8_ema_ablation.json")
    
    return results


if __name__ == "__main__":
    run_experiment_8()
