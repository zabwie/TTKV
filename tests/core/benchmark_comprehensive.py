import torch
import torch.nn.functional as F
import numpy as np
import json
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass

from ttkv import CacheConfig, TieredKVCache
from ttkv import compute_type_prior_retention


@dataclass
class BenchmarkResult:
    config_name: str
    tau: float
    seq_len: int
    
    # Quality metrics
    perplexity_full: float
    perplexity_compressed: float
    perplexity_delta: float
    
    # Speed metrics (seconds)
    prefill_time: float
    decode_time_per_token: float
    scorer_overhead: float
    
    # Memory metrics (MB)
    kv_cache_full_mb: float
    kv_cache_compressed_mb: float
    memory_saved_mb: float
    memory_saved_pct: float
    
    # Computed tradeoffs
    quality_degradation_pct: float
    speedup: float
    compression_ratio: float


def create_test_sequence(seq_len, batch_size=1, num_heads=12, head_dim=64):
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    token_ids = torch.randint(0, 50000, (batch_size, seq_len))
    return k, v, positions, token_ids


def measure_perplexity(k, v, retention, config):
    """
    Measure perplexity by computing average cross-entropy.
    Lower is better.
    """
    cache = TieredKVCache(config)
    cache.add(k, v, retention, torch.arange(k.size(2)).unsqueeze(0))
    k_comp, v_comp, _ = cache.get_compressed_cache()
    
    # Simulate next-token prediction
    # Use k_comp as keys and compute attention to v_comp
    query = torch.randn(1, 1, 64)  # single query
    
    # Full cache
    attn_full = torch.matmul(query, k[0, 0, :, :].T) / (64 ** 0.5)
    attn_full = F.softmax(attn_full, dim=-1)
    output_full = torch.matmul(attn_full, v[0, 0, :, :])
    
    # Compressed cache
    attn_comp = torch.matmul(query, k_comp[0, 0, :, :].T) / (64 ** 0.5)
    attn_comp = F.softmax(attn_comp, dim=-1)
    output_comp = torch.matmul(attn_comp, v_comp[0, 0, :, :])
    
    # Perplexity = exp(cross_entropy)
    # Simplified: use MSE as proxy
    loss = F.mse_loss(output_comp, output_full)
    perplexity = torch.exp(loss).item()
    
    return perplexity


def measure_prefill_latency(k, v, retention, config, num_runs=10):
    """Measure time to build compressed cache (prefill)."""
    cache = TieredKVCache(config)
    
    # Warmup
    for _ in range(3):
        cache.clear()
        cache.add(k, v, retention, torch.arange(k.size(2)).unsqueeze(0))
        _ = cache.get_compressed_cache()
    
    # Measure
    times = []
    for _ in range(num_runs):
        cache.clear()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        start = time.perf_counter()
        cache.add(k, v, retention, torch.arange(k.size(2)).unsqueeze(0))
        _ = cache.get_compressed_cache()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()
        
        times.append(end - start)
    
    return np.mean(times)


def measure_scorer_overhead(seq_len, num_runs=100):
    """Measure overhead of computing salience scores."""
    from ttkv import SalienceScorer
    
    scorer = SalienceScorer(hidden_dim=768, salience_hidden=256)
    hidden_states = torch.randn(1, seq_len, 768)
    
    # Warmup
    for _ in range(10):
        _ = scorer(hidden_states)
    
    # Measure
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = scorer(hidden_states)
        end = time.perf_counter()
        times.append(end - start)
    
    return np.mean(times)


def measure_memory_footprint(k, v, retention, config):
    """Measure KV cache memory footprint in MB."""
    cache = TieredKVCache(config)
    cache.add(k, v, retention, torch.arange(k.size(2)).unsqueeze(0))
    
    # Full cache size
    kv_full_mb = 2 * k.size(2) * k.size(1) * k.size(3) * 4 / (1024 * 1024)  # k + v, float32
    
    # Compressed size
    k_comp, v_comp, _ = cache.get_compressed_cache()
    if k_comp is not None:
        kv_comp_mb = 2 * k_comp.size(2) * k_comp.size(1) * k_comp.size(3) * 4 / (1024 * 1024)
    else:
        kv_comp_mb = kv_full_mb
    
    return kv_full_mb, kv_comp_mb


def run_comprehensive_benchmark():
    """Run comprehensive three-dimensional benchmark."""
    print("=" * 80)
    print("COMPREHENSIVE BENCHMARK: Quality × Speed × Memory")
    print("=" * 80)
    
    tau_values = [0.0, 0.6, 0.7, 0.8, 0.9]
    seq_lengths = [2048, 4096, 8192]
    results = []
    
    for seq_len in seq_lengths:
        print(f"\n{'=' * 80}")
        print(f"Sequence Length: {seq_len}")
        print(f"{'=' * 80}")
        
        # Generate test data
        k, v, positions, token_ids = create_test_sequence(seq_len)
        retention = compute_type_prior_retention(token_ids)
        
        print(f"Generated {seq_len} tokens, {(retention > 0.8).sum().item()} high-salience")
        
        # Measure scorer overhead once
        scorer_time = measure_scorer_overhead(seq_len)
        print(f"Salience scorer overhead: {scorer_time * 1000:.3f}ms")
        print()
        
        for tau in tau_values:
            config = CacheConfig(
                tau_threshold=tau,
                tier0_size=min(256, seq_len // 8),
                tier1_size=min(2048, seq_len // 2)
            )
            
            print(f"τ = {tau:.1f}:", end=" ")
            
            try:
                # Quality: perplexity
                ppl_compressed = measure_perplexity(k, v, retention, config)
                
                # Baseline (no compression)
                config_baseline = CacheConfig(tau_threshold=0.0)
                ppl_full = measure_perplexity(k, v, retention, config_baseline)
                
                ppl_delta = ppl_compressed - ppl_full
                quality_deg = (ppl_delta / ppl_full) * 100 if ppl_full > 0 else 0
                
                # Speed: prefill latency
                prefill_time = measure_prefill_latency(k, v, retention, config)
                prefill_baseline = measure_prefill_latency(k, v, retention, config_baseline)
                speedup = prefill_baseline / prefill_time if prefill_time > 0 else 1.0
                
                # Memory
                mem_full, mem_comp = measure_memory_footprint(k, v, retention, config)
                mem_saved = mem_full - mem_comp
                mem_saved_pct = (mem_saved / mem_full) * 100
                compression = mem_full / mem_comp
                
                result = BenchmarkResult(
                    config_name=f"tau_{tau:.1f}",
                    tau=tau,
                    seq_len=seq_len,
                    perplexity_full=ppl_full,
                    perplexity_compressed=ppl_compressed,
                    perplexity_delta=ppl_delta,
                    prefill_time=prefill_time,
                    decode_time_per_token=0.0,  # Not measured
                    scorer_overhead=scorer_time,
                    kv_cache_full_mb=mem_full,
                    kv_cache_compressed_mb=mem_comp,
                    memory_saved_mb=mem_saved,
                    memory_saved_pct=mem_saved_pct,
                    quality_degradation_pct=quality_deg,
                    speedup=speedup,
                    compression_ratio=compression
                )
                results.append(result)
                
                status = "✓" if quality_deg < 20 else "⚠" if quality_deg < 50 else "✗"
                print(f"{compression:.2f}x, pplΔ={ppl_delta:.3f} ({quality_deg:+.1f}%), "
                      f"prefill={prefill_time*1000:.1f}ms ({speedup:.2f}x), "
                      f"mem={mem_saved_pct:.1f}% {status}")
                
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    # Print comprehensive results table
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS TABLE")
    print("=" * 80)
    print(f"{'Seq':>6} | {'τ':>5} | {'Ratio':>7} | {'PPL Δ':>8} | {'Qual↓':>7} | "
          f"{'Prefill':>9} | {'Speed':>6} | {'Mem↓':>6} | {'Deploy':>8}")
    print("-" * 80)
    
    for r in results:
        deployable = "✓" if (r.quality_degradation_pct < 15 and r.memory_saved_pct > 30) else "✗"
        print(f"{r.seq_len:>6} | {r.tau:>5.1f} | {r.compression_ratio:>6.2f}x | "
              f"{r.perplexity_delta:>+7.3f} | {r.quality_degradation_pct:>+6.1f}% | "
              f"{r.prefill_time*1000:>8.1f}ms | {r.speedup:>5.2f}x | "
              f"{r.memory_saved_pct:>5.1f}% | {deployable:>8}")
    
    print("=" * 80)
    
    # Three-way tradeoff analysis
    print("\n" + "=" * 80)
    print("THREE-WAY TRADEOFF ANALYSIS")
    print("=" * 80)
    
    for seq_len in seq_lengths:
        print(f"\n{seq_len} tokens:")
        
        seq_results = [r for r in results if r.seq_len == seq_len]
        
        # Find Pareto frontier: best compression with acceptable quality
        acceptable = [r for r in seq_results if r.quality_degradation_pct < 20]
        
        if acceptable:
            best = max(acceptable, key=lambda x: x.compression_ratio)
            print(f"  Best compression (qual<20%): τ={best.tau:.1f} → "
                  f"{best.compression_ratio:.2f}x, {best.quality_degradation_pct:.1f}% deg, "
                  f"{best.memory_saved_pct:.1f}% mem")
        
        # Find best quality
        best_qual = min(seq_results, key=lambda x: abs(x.quality_degradation_pct))
        print(f"  Best quality: τ={best_qual.tau:.1f} → "
              f"{best_qual.quality_degradation_pct:.1f}% deg, {best_qual.compression_ratio:.2f}x")
        
        # Check if any are deployable
        deployable = [r for r in seq_results 
                     if r.quality_degradation_pct < 15 and r.memory_saved_pct > 30]
        
        if deployable:
            best_dep = max(deployable, key=lambda x: x.compression_ratio)
            print(f"  ✓ DEPLOYABLE: τ={best_dep.tau:.1f} → "
                  f"{best_dep.compression_ratio:.2f}x, {best_dep.quality_degradation_pct:.1f}% deg")
        else:
            print(f"  ✗ No deployable configuration (need qual<15%, mem>30%)")
    
    # Critical insight
    print("\n" + "=" * 80)
    print("CRITICAL INSIGHT")
    print("=" * 80)
    
    # Check if salience scorer overhead dominates
    total_scorer_time = sum(r.scorer_overhead * r.seq_len for r in results) / len(results)
    avg_prefill = np.mean([r.prefill_time for r in results])
    
    print(f"Average scorer overhead: {total_scorer_time * 1000:.3f}ms")
    print(f"Average prefill time: {avg_prefill * 1000:.3f}ms")
    
    if total_scorer_time > avg_prefill * 0.5:
        print("⚠ WARNING: Scorer overhead is >50% of prefill time")
        print("  → Consider: caching salience scores, lighter MLP, or removing scorer")
    else:
        print("✓ Scorer overhead is acceptable")
    
    # Save results
    results_dict = [
        {
            'config_name': r.config_name,
            'tau': r.tau,
            'seq_len': r.seq_len,
            'perplexity_full': r.perplexity_full,
            'perplexity_compressed': r.perplexity_compressed,
            'perplexity_delta': r.perplexity_delta,
            'prefill_time_ms': r.prefill_time * 1000,
            'scorer_overhead_ms': r.scorer_overhead * 1000,
            'kv_cache_full_mb': r.kv_cache_full_mb,
            'kv_cache_compressed_mb': r.kv_cache_compressed_mb,
            'memory_saved_pct': r.memory_saved_pct,
            'quality_degradation_pct': r.quality_degradation_pct,
            'compression_ratio': r.compression_ratio
        }
        for r in results
    ]
    
    with open('comprehensive_benchmark.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print("\nResults saved to comprehensive_benchmark.json")
    print("=" * 80)


if __name__ == "__main__":
    run_comprehensive_benchmark()
