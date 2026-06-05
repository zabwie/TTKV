"""
Latency Overhead Measurements

Measures runtime overhead of tiered compression: prefill latency, decode
latency, tokens/sec, and memory usage. Compares Tiered against uncompressed
baseline, H2O, and ScissorHands.

Produces a comparison table suitable for the paper.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from ttkv import CacheConfig, TieredKVCache
from ttkv import compute_type_prior_retention, SalienceScorer
from tests.core.baselines import H2OCache, ScissorHandsCache


@dataclass
class LatencyResult:
    method: str
    seq_len: int
    tau: float

    # Timing (seconds)
    prefill_time: float
    decode_time_per_token: float
    scorer_time: float

    # Throughput
    tokens_per_sec_prefill: float
    tokens_per_sec_decode: float

    # Memory (MB)
    kv_cache_mb: float
    kv_cache_compressed_mb: float
    memory_saved_pct: float

    # Quality proxy
    reconstruction_mse: float
    compression_ratio: float

    # Overhead
    overhead_pct: float  # % increase in prefill time vs uncompressed


def generate_test_kv(seq_len: int, batch_size: int = 1,
                     num_heads: int = 12, head_dim: int = 64):
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    token_ids = torch.randint(0, 50000, (batch_size, seq_len))
    retention = compute_type_prior_retention(token_ids)
    return k, v, positions, retention


def warmup_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def time_it(fn, warmup: int = 3, runs: int = 10):
    for _ in range(warmup):
        fn()
    warmup_gpu()

    times = []
    for _ in range(runs):
        warmup_gpu()
        start = time.perf_counter()
        fn()
        warmup_gpu()
        times.append(time.perf_counter() - start)
    return np.mean(times), np.std(times)


def measure_prefill(k, v, positions, retention, config, use_tiered=True):
    if use_tiered:
        cache = TieredKVCache(config)
    else:
        cache = TieredKVCache(CacheConfig(tau_threshold=0.0))

    def run():
        cache.clear()
        cache.add(k, v, retention, positions)
        _ = cache.get_compressed_cache()
    return time_it(run)


def measure_decode(k_comp, v_comp, pos_comp, head_dim: int = 64):
    if k_comp is None or k_comp.size(2) == 0:
        return 0.0, 0.0

    query = torch.randn(1, 1, head_dim)
    scale = head_dim ** -0.5

    def run():
        scores = torch.matmul(query, k_comp[0, 0, :, :].T) * scale
        attn = F.softmax(scores, dim=-1)
        _ = torch.matmul(attn, v_comp[0, 0, :, :])
    return time_it(run, warmup=5, runs=50)


def measure_scorer(seq_len: int, hidden_dim: int = 768):
    scorer = SalienceScorer(hidden_dim=hidden_dim, salience_hidden=256)
    hidden_states = torch.randn(1, seq_len, hidden_dim)

    def run():
        _ = scorer(hidden_states)
    return time_it(run, warmup=5, runs=20)


def measure_memory(k, v):
    kv_full_mb = (k.element_size() * k.numel() + v.element_size() * v.numel()) / (1024 * 1024)
    return kv_full_mb


def compute_reconstruction_mse(k_orig, v_orig, k_comp, v_comp):
    if k_comp is None or k_comp.size(2) == 0:
        return float('inf')
    comp_len = k_comp.size(2)
    orig_len = k_orig.size(2)
    indices = torch.linspace(0, orig_len - 1, comp_len).long()
    k_sampled = k_orig[:, :, indices, :]
    v_sampled = v_orig[:, :, indices, :]
    mse_k = F.mse_loss(k_comp, k_sampled).item()
    mse_v = F.mse_loss(v_comp, v_sampled).item()
    return (mse_k + mse_v) / 2


def run_latency_benchmark():
    print("=" * 90)
    print("LATENCY OVERHEAD BENCHMARK")
    print("Prefill / Decode / Memory / Overhead Comparison")
    print("=" * 90)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()

    tau = 0.9
    seq_lengths = [1024, 2048, 4096, 8192, 12288]
    results: List[LatencyResult] = []

    for seq_len in seq_lengths:
        print(f"{'─' * 90}")
        print(f"Sequence Length: {seq_len:,} tokens")
        print(f"{'─' * 90}")

        k, v, positions, retention = generate_test_kv(seq_len)
        head_dim = k.size(3)

        config_tiered = CacheConfig(tau_threshold=tau)
        config_baseline = CacheConfig(tau_threshold=0.0)

        # ── Uncompressed baseline ──────────────────────────────────────
        print("  Baseline (uncompressed)...")
        prefill_base_mean, prefill_base_std = measure_prefill(
            k.clone(), v.clone(), positions.clone(), retention.clone(),
            config_baseline, use_tiered=False
        )

        cache_base = TieredKVCache(config_baseline)
        cache_base.add(k.clone(), v.clone(), retention.clone(), positions.clone())
        k_base, v_base, pos_base = cache_base.get_compressed_cache()
        stats_base = cache_base.get_stats()

        decode_base_mean, decode_base_std = measure_decode(k_base, v_base, pos_base, head_dim)
        mem_base = measure_memory(k, v)
        tokens_per_sec_prefill_base = seq_len / prefill_base_mean if prefill_base_mean > 0 else float('inf')
        tokens_per_sec_decode_base = 1.0 / decode_base_mean if decode_base_mean > 0 else float('inf')

        results.append(LatencyResult(
            method="Baseline",
            seq_len=seq_len, tau=0.0,
            prefill_time=prefill_base_mean,
            decode_time_per_token=decode_base_mean,
            scorer_time=0.0,
            tokens_per_sec_prefill=tokens_per_sec_prefill_base,
            tokens_per_sec_decode=tokens_per_sec_decode_base,
            kv_cache_mb=mem_base, kv_cache_compressed_mb=mem_base,
            memory_saved_pct=0.0,
            reconstruction_mse=0.0,
            compression_ratio=1.0,
            overhead_pct=0.0,
        ))

        # ── Salience scorer overhead ──────────────────────────────────
        scorer_mean, scorer_std = measure_scorer(seq_len)
        print(f"    Scorer overhead: {scorer_mean*1000:.2f}ms")

        # ── Tiered compression ────────────────────────────────────────
        print("  Tiered (Ours)...")
        prefill_tiered_mean, prefill_tiered_std = measure_prefill(
            k.clone(), v.clone(), positions.clone(), retention.clone(),
            config_tiered, use_tiered=True
        )

        cache_tiered = TieredKVCache(config_tiered)
        cache_tiered.add(k.clone(), v.clone(), retention.clone(), positions.clone())
        k_tiered, v_tiered, pos_tiered = cache_tiered.get_compressed_cache()
        stats_tiered = cache_tiered.get_stats()

        decode_tiered_mean, decode_tiered_std = measure_decode(k_tiered, v_tiered, pos_tiered, head_dim)
        mem_comp = measure_memory(k_tiered, v_tiered) if k_tiered is not None else mem_base
        mem_saved = (1 - mem_comp / mem_base) * 100 if mem_base > 0 else 0
        mse = compute_reconstruction_mse(k, v, k_tiered, v_tiered)
        overhead_pct = ((prefill_tiered_mean - prefill_base_mean) / prefill_base_mean) * 100

        tokens_per_sec_prefill = seq_len / prefill_tiered_mean if prefill_tiered_mean > 0 else float('inf')
        tokens_per_sec_decode = 1.0 / decode_tiered_mean if decode_tiered_mean > 0 else float('inf')

        results.append(LatencyResult(
            method="Tiered (Ours)",
            seq_len=seq_len, tau=tau,
            prefill_time=prefill_tiered_mean,
            decode_time_per_token=decode_tiered_mean,
            scorer_time=scorer_mean,
            tokens_per_sec_prefill=tokens_per_sec_prefill,
            tokens_per_sec_decode=tokens_per_sec_decode,
            kv_cache_mb=mem_base, kv_cache_compressed_mb=mem_comp,
            memory_saved_pct=mem_saved,
            reconstruction_mse=mse,
            compression_ratio=stats_tiered['compression_ratio'],
            overhead_pct=overhead_pct,
        ))

        # ── H2O ──────────────────────────────────────────────────────
        print("  H2O...")
        config_h2o = CacheConfig()

        def run_h2o():
            cache = H2OCache(config_h2o, max_cache_size=2048)
            attn = torch.ones(1, seq_len) * 0.1
            if seq_len > 0:
                attn[0, 0] = 0.0
            cache.add(k, v, attn, positions)
            return cache.get_compressed_cache()

        prefill_h2o_mean, prefill_h2o_std = time_it(run_h2o)

        cache_h2o = H2OCache(config_h2o, max_cache_size=2048)
        attn = torch.ones(1, seq_len) * 0.1
        if seq_len > 0:
            attn[0, 0] = 0.0
        cache_h2o.add(k, v, attn, positions)
        k_h2o, v_h2o, pos_h2o = cache_h2o.get_compressed_cache()
        stats_h2o = cache_h2o.get_stats()

        decode_h2o_mean, decode_h2o_std = measure_decode(k_h2o, v_h2o, pos_h2o, head_dim)
        mem_h2o_comp = measure_memory(k_h2o, v_h2o) if k_h2o is not None else mem_base
        mem_h2o_saved = (1 - mem_h2o_comp / mem_base) * 100 if mem_base > 0 else 0
        mse_h2o = compute_reconstruction_mse(k, v, k_h2o, v_h2o)
        overhead_h2o = ((prefill_h2o_mean - prefill_base_mean) / prefill_base_mean) * 100

        tokens_per_sec_h2o = seq_len / prefill_h2o_mean if prefill_h2o_mean > 0 else float('inf')
        tokens_per_sec_decode_h2o = 1.0 / decode_h2o_mean if decode_h2o_mean > 0 else float('inf')

        results.append(LatencyResult(
            method="H2O",
            seq_len=seq_len, tau=0.0,
            prefill_time=prefill_h2o_mean,
            decode_time_per_token=decode_h2o_mean,
            scorer_time=0.0,
            tokens_per_sec_prefill=tokens_per_sec_h2o,
            tokens_per_sec_decode=tokens_per_sec_decode_h2o,
            kv_cache_mb=mem_base, kv_cache_compressed_mb=mem_h2o_comp,
            memory_saved_pct=mem_h2o_saved,
            reconstruction_mse=mse_h2o,
            compression_ratio=stats_h2o['compression_ratio'],
            overhead_pct=overhead_h2o,
        ))

        # ── ScissorHands ─────────────────────────────────────────────
        print("  ScissorHands...")
        config_sh = CacheConfig()

        def run_sh():
            cache = ScissorHandsCache(config_sh, max_cache_size=2048, attention_window=256)
            cache.add(k, v, retention, positions)
            return cache.get_compressed_cache()

        prefill_sh_mean, prefill_sh_std = time_it(run_sh)

        cache_sh = ScissorHandsCache(config_sh, max_cache_size=2048, attention_window=256)
        cache_sh.add(k, v, retention, positions)
        k_sh, v_sh, pos_sh = cache_sh.get_compressed_cache()
        stats_sh = cache_sh.get_stats()

        decode_sh_mean, decode_sh_std = measure_decode(k_sh, v_sh, pos_sh, head_dim)
        mem_sh_comp = measure_memory(k_sh, v_sh) if k_sh is not None else mem_base
        mem_sh_saved = (1 - mem_sh_comp / mem_base) * 100 if mem_base > 0 else 0
        mse_sh = compute_reconstruction_mse(k, v, k_sh, v_sh)
        overhead_sh = ((prefill_sh_mean - prefill_base_mean) / prefill_base_mean) * 100

        tokens_per_sec_sh = seq_len / prefill_sh_mean if prefill_sh_mean > 0 else float('inf')
        tokens_per_sec_decode_sh = 1.0 / decode_sh_mean if decode_sh_mean > 0 else float('inf')

        results.append(LatencyResult(
            method="ScissorHands",
            seq_len=seq_len, tau=0.0,
            prefill_time=prefill_sh_mean,
            decode_time_per_token=decode_sh_mean,
            scorer_time=0.0,
            tokens_per_sec_prefill=tokens_per_sec_sh,
            tokens_per_sec_decode=tokens_per_sec_decode_sh,
            kv_cache_mb=mem_base, kv_cache_compressed_mb=mem_sh_comp,
            memory_saved_pct=mem_sh_saved,
            reconstruction_mse=mse_sh,
            compression_ratio=stats_sh['compression_ratio'],
            overhead_pct=overhead_sh,
        ))

        # ── Per-length summary ───────────────────────────────────────
        for r in results[-4:]:
            label = "✓" if r.overhead_pct < 15 else ("⚠" if r.overhead_pct < 50 else "✗")
            print(f"    {r.method:<18} | prefill={r.prefill_time*1000:.1f}ms "
                  f"({r.tokens_per_sec_prefill:.0f} tok/s) | "
                  f"decode={r.decode_time_per_token*1000:.2f}ms/tok | "
                  f"mem↓={r.memory_saved_pct:.0f}% | "
                  f"overhead={r.overhead_pct:+.0f}% {label}")

    return results


def print_latency_table(results: List[LatencyResult]):
    print("\n" + "=" * 90)
    print("COMPREHENSIVE LATENCY RESULTS TABLE (for paper)")
    print("=" * 90)

    seq_lengths = sorted(set(r.seq_len for r in results))
    methods = ["Baseline", "H2O", "ScissorHands", "Tiered (Ours)"]

    # ── Table 1: Prefill Latency & Throughput ─────────────────────────
    print(f"\n{'Table 1: Prefill Performance':^90}")
    header = f"{'Seq Len':<10} {'Method':<18} {'Prefill (ms)':<14} {'Tok/s':<12} {'Overhead %':<12} {'Ratio':<8}"
    print(header)
    print("-" * 90)

    for seq_len in seq_lengths:
        for method in methods:
            matches = [r for r in results if r.seq_len == seq_len and r.method == method]
            if matches:
                r = matches[0]
                overhead_str = f"{r.overhead_pct:+.1f}%" if r.method != "Baseline" else "baseline"
                print(f"{seq_len:<10} {method:<18} {r.prefill_time*1000:>12.1f}  "
                      f"{r.tokens_per_sec_prefill:>10.0f}  {overhead_str:>10}  "
                      f"{r.compression_ratio:>6.2f}x")

    # ── Table 2: Decode Performance ───────────────────────────────────
    print(f"\n{'Table 2: Decode Performance (per-token)':^90}")
    header = f"{'Seq Len':<10} {'Method':<18} {'Decode (ms/tok)':<16} {'Tok/s':<12} {'Ratio':<8}"
    print(header)
    print("-" * 90)

    for seq_len in seq_lengths:
        for method in methods:
            matches = [r for r in results if r.seq_len == seq_len and r.method == method]
            if matches:
                r = matches[0]
                print(f"{seq_len:<10} {method:<18} {r.decode_time_per_token*1000:>14.3f}  "
                      f"{r.tokens_per_sec_decode:>10.0f}  {r.compression_ratio:>6.2f}x")

    # ── Table 3: Memory Usage ─────────────────────────────────────────
    print(f"\n{'Table 3: Memory Usage':^90}")
    header = f"{'Seq Len':<10} {'Method':<18} {'Full (MB)':<12} {'Comp (MB)':<12} {'Saved %':<10} {'Ratio':<8}"
    print(header)
    print("-" * 90)

    for seq_len in seq_lengths:
        for method in methods:
            matches = [r for r in results if r.seq_len == seq_len and r.method == method]
            if matches:
                r = matches[0]
                print(f"{seq_len:<10} {method:<18} {r.kv_cache_mb:>10.1f}  "
                      f"{r.kv_cache_compressed_mb:>10.1f}  {r.memory_saved_pct:>8.1f}%  "
                      f"{r.compression_ratio:>6.2f}x")

    # ── Table 4: Combined Tradeoff (for paper) ────────────────────────
    print(f"\n{'Table 4: Combined Speed-Memory-Quality Tradeoff':^90}")
    print(f"{'Method':<18} {'Seq':<8} {'Ratio':<7} {'Tok/s':<10} "
          f"{'VRAM↓%':<8} {'Overhead':<10} {'MSE':<10}")
    print("-" * 90)

    for seq_len in seq_lengths:
        for method in methods:
            matches = [r for r in results if r.seq_len == seq_len and r.method == method]
            if matches:
                r = matches[0]
                print(f"{method:<18} {seq_len:<8} {r.compression_ratio:>5.2f}x "
                      f"{r.tokens_per_sec_decode:>8.0f}  {r.memory_saved_pct:>6.1f}%  "
                      f"{r.overhead_pct:>+8.1f}%  {r.reconstruction_mse:>8.4f}")

    # ── Key Findings ───────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("KEY FINDINGS")
    print("=" * 90)

    tiered_results = [r for r in results if r.method == "Tiered (Ours)"]
    h2o_results = [r for r in results if r.method == "H2O"]
    sh_results = [r for r in results if r.method == "ScissorHands"]
    baseline_results = [r for r in results if r.method == "Baseline"]

    if tiered_results:
        avg_overhead = np.mean([r.overhead_pct for r in tiered_results])
        avg_comp = np.mean([r.compression_ratio for r in tiered_results])
        avg_mem_saved = np.mean([r.memory_saved_pct for r in tiered_results])
        avg_decode_tps = np.mean([r.tokens_per_sec_decode for r in tiered_results])

        print(f"\n  Tiered compression (averaged across sequence lengths):")
        print(f"    Avg overhead:     {avg_overhead:+.1f}%")
        print(f"    Avg compression:  {avg_comp:.2f}x")
        print(f"    Avg memory saved: {avg_mem_saved:.1f}%")
        print(f"    Avg decode tok/s: {avg_decode_tps:.0f}")

    if h2o_results and tiered_results:
        # Compare overhead at 8K
        h2o_8k = [r for r in h2o_results if r.seq_len == 8192]
        tiered_8k = [r for r in tiered_results if r.seq_len == 8192]
        if h2o_8k and tiered_8k:
            print(f"\n  At 8K context:")
            print(f"    H2O overhead:     {h2o_8k[0].overhead_pct:+.1f}%")
            print(f"    Tiered overhead:  {tiered_8k[0].overhead_pct:+.1f}%")
            diff = tiered_8k[0].overhead_pct - h2o_8k[0].overhead_pct
            if diff < 5:
                print(f"    Difference:       {diff:+.1f}% — comparable to binary eviction")
            else:
                print(f"    Difference:       {diff:+.1f}% — tradeoff for {tiered_8k[0].compression_ratio:.1f}x compression")

    print("=" * 90)


def save_results(results: List[LatencyResult], filename: str = None):
    if filename is None:
        filename = os.path.join(os.path.dirname(__file__), '..', '..',
                                'results', 'exp_latency_overhead.json')

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    serializable = [
        {
            'method': r.method,
            'seq_len': r.seq_len,
            'tau': r.tau,
            'prefill_time_ms': r.prefill_time * 1000,
            'decode_time_per_token_ms': r.decode_time_per_token * 1000,
            'scorer_time_ms': r.scorer_time * 1000,
            'tokens_per_sec_prefill': r.tokens_per_sec_prefill,
            'tokens_per_sec_decode': r.tokens_per_sec_decode,
            'kv_cache_mb': r.kv_cache_mb,
            'kv_cache_compressed_mb': r.kv_cache_compressed_mb,
            'memory_saved_pct': r.memory_saved_pct,
            'reconstruction_mse': r.reconstruction_mse,
            'compression_ratio': r.compression_ratio,
            'overhead_pct': r.overhead_pct,
        }
        for r in results
    ]

    with open(filename, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {filename}")


def main():
    results = run_latency_benchmark()
    print_latency_table(results)
    save_results(results)
    print("\n" + "=" * 90)
    print("LATENCY BENCHMARK COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
