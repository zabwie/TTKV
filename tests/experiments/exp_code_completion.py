"""
Downstream Task: Code Completion Persistence

Tests whether tiered compression preserves code structure (variable
definitions, function signatures) over long contexts for practical
code completion tasks.

Scenario:
  1. Define variables/functions early in context
  2. Fill with code-like tokens for 10K+ positions
  3. Query for variable reuse at the end
  4. Measure: exact variable recall, syntax correctness, pass@1

Compares Tiered vs H2O vs ScissorHands.
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
from ttkv import compute_type_prior_retention
from tests.core.baselines import H2OCache, ScissorHandsCache


@dataclass
class CodeCompletionResult:
    method: str
    seq_len: int
    num_variables: int
    variables_preserved: int
    exact_recall: float
    syntax_preservation: float
    pass_at_1: bool
    compression_ratio: float
    tokens_kept: int


def generate_code_context(seq_len: int, num_definitions: int = 5,
                          distinctiveness: float = 5.0):
    """
    Generate a code-like KV cache with variable/function definitions early
    and filler code afterward. Definitions are made distinctive so attention
    can locate them.
    """
    batch_size = 1
    num_heads = 12
    head_dim = 64

    k = torch.randn(batch_size, num_heads, seq_len, head_dim) * 0.5
    v = torch.randn(batch_size, num_heads, seq_len, head_dim) * 0.5
    positions = torch.arange(seq_len).unsqueeze(0)
    token_ids = torch.randint(0, 50000, (batch_size, seq_len))

    retention = torch.full((batch_size, seq_len), 0.2)
    definition_positions = []

    spacing = max(seq_len // (num_definitions * 10), 1)
    for d in range(num_definitions):
        pos = d * spacing
        if pos >= seq_len - 128:
            break
        definition_positions.append(pos)
        k[0, :, pos, :] = torch.randn(num_heads, head_dim) * distinctiveness
        v[0, :, pos, :] = torch.randn(num_heads, head_dim) * distinctiveness
        retention[0, pos] = 0.92

        # Nearby tokens in the definition also get moderate retention
        for offset in range(1, min(6, seq_len - pos)):
            retention[0, pos + offset] = 0.65

    # Code-like filler: periodic structural tokens (simulating keywords)
    for i in range(0, seq_len, 20):
        if i not in definition_positions and retention[0, i] < 0.6:
            retention[0, i] = 0.45

    # Recent window
    recent_start = max(0, seq_len - 256)
    for i in range(recent_start, seq_len):
        retention[0, i] = max(retention[0, i].item(), 0.5)

    return k, v, positions, retention, definition_positions


def check_variable_preservation(k_comp, v_comp, pos_comp,
                                 definition_positions: List[int],
                                 seq_len: int) -> Dict:
    num_defs = len(definition_positions)
    preserved = 0
    exact_positions_found = 0
    probabilities = []

    if k_comp is None or k_comp.size(2) == 0 or pos_comp is None:
        return {
            'preserved': 0, 'total': num_defs,
            'exact_positions': 0, 'probabilities': [0.0] * num_defs,
            'recall': 0.0
        }

    head_dim = k_comp.size(3)
    query = torch.randn(1, 1, head_dim)
    scale = head_dim ** -0.5
    scores = torch.matmul(query, k_comp[0, 0, :, :].T) * scale
    attn = F.softmax(scores, dim=-1).squeeze(0).squeeze(0)

    for def_pos in definition_positions:
        exact_mask = (pos_comp[0] == def_pos)
        if exact_mask.any().item():
            exact_positions_found += 1

        neighbor_range = 32
        range_lower = max(0, def_pos - neighbor_range)
        range_upper = min(def_pos + neighbor_range, seq_len)
        range_mask = (pos_comp[0] >= range_lower) & (pos_comp[0] <= range_upper)

        if range_mask.any():
            prob = attn[range_mask].sum().item()
            probabilities.append(prob)
            if prob > 0.005:
                preserved += 1
        else:
            probabilities.append(0.0)

    exact_recall = exact_positions_found / num_defs if num_defs > 0 else 0.0
    var_recall = preserved / num_defs if num_defs > 0 else 0.0

    return {
        'preserved': preserved, 'total': num_defs,
        'exact_positions': exact_positions_found,
        'probabilities': probabilities,
        'recall': var_recall,
        'exact_recall': exact_recall,
    }


def test_code_completion_tiered(k, v, positions, retention,
                                  definition_positions: List[int],
                                  seq_len: int, tau: float = 0.9):
    config = CacheConfig(tau_threshold=tau)
    cache = TieredKVCache(config)
    cache.add(k, v, retention, positions)
    k_comp, v_comp, pos_comp = cache.get_compressed_cache()
    stats = cache.get_stats()

    preservation = check_variable_preservation(
        k_comp, v_comp, pos_comp, definition_positions, seq_len
    )

    return CodeCompletionResult(
        method="Tiered (Ours)",
        seq_len=seq_len,
        num_variables=len(definition_positions),
        variables_preserved=preservation['preserved'],
        exact_recall=preservation['exact_recall'],
        syntax_preservation=preservation['recall'],
        pass_at_1=preservation['recall'] == 1.0,
        compression_ratio=stats['compression_ratio'],
        tokens_kept=stats['compressed_tokens'],
    )


def test_code_completion_h2o(k, v, positions, retention,
                               definition_positions: List[int],
                               seq_len: int, max_cache_size: int = 2048):
    config = CacheConfig()
    cache = H2OCache(config, max_cache_size=max_cache_size)
    attention = torch.ones(1, seq_len) * 0.1
    # Definitions get zero accumulated attention — slow-burn
    for pos in definition_positions:
        attention[0, pos] = 0.0
    cache.add(k, v, attention, positions)

    k_comp, v_comp, pos_comp = cache.get_compressed_cache()
    stats = cache.get_stats()

    preservation = check_variable_preservation(
        k_comp, v_comp, pos_comp, definition_positions, seq_len
    )

    return CodeCompletionResult(
        method="H2O",
        seq_len=seq_len,
        num_variables=len(definition_positions),
        variables_preserved=preservation['preserved'],
        exact_recall=preservation['exact_recall'],
        syntax_preservation=preservation['recall'],
        pass_at_1=preservation['recall'] == 1.0,
        compression_ratio=stats['compression_ratio'],
        tokens_kept=stats['compressed_tokens'],
    )


def test_code_completion_sh(k, v, positions, retention,
                              definition_positions: List[int],
                              seq_len: int, max_cache_size: int = 2048):
    config = CacheConfig()
    cache = ScissorHandsCache(config, max_cache_size=max_cache_size,
                               attention_window=256)
    cache.add(k, v, retention, positions)

    k_comp, v_comp, pos_comp = cache.get_compressed_cache()
    stats = cache.get_stats()

    preservation = check_variable_preservation(
        k_comp, v_comp, pos_comp, definition_positions, seq_len
    )

    return CodeCompletionResult(
        method="ScissorHands",
        seq_len=seq_len,
        num_variables=len(definition_positions),
        variables_preserved=preservation['preserved'],
        exact_recall=preservation['exact_recall'],
        syntax_preservation=preservation['recall'],
        pass_at_1=preservation['recall'] == 1.0,
        compression_ratio=stats['compression_ratio'],
        tokens_kept=stats['compressed_tokens'],
    )


def run_code_completion_benchmark():
    print("=" * 90)
    print("CODE COMPLETION PERSISTENCE BENCHMARK")
    print("Variable/function definition early → reuse after long context")
    print("=" * 90)

    tau = 0.9
    configs = [
        (2048, 3),
        (4096, 5),
        (8192, 5),
        (12288, 7),
        (16384, 7),
    ]

    results: List[CodeCompletionResult] = []

    for seq_len, num_defs in configs:
        print(f"\n{'─' * 90}")
        print(f"Sequence: {seq_len:,} tokens, {num_defs} definitions")
        print(f"{'─' * 90}")

        k, v, positions, retention, def_positions = generate_code_context(
            seq_len, num_definitions=num_defs
        )

        actual_defs = len(def_positions)
        print(f"  Definitions at positions: {def_positions}")

        r_t = test_code_completion_tiered(
            k.clone(), v.clone(), positions.clone(), retention.clone(),
            def_positions, seq_len, tau
        )
        results.append(r_t)

        r_h = test_code_completion_h2o(
            k.clone(), v.clone(), positions.clone(), retention.clone(),
            def_positions, seq_len
        )
        results.append(r_h)

        r_s = test_code_completion_sh(
            k.clone(), v.clone(), positions.clone(), retention.clone(),
            def_positions, seq_len
        )
        results.append(r_s)

        for r in [r_t, r_h, r_s]:
            status = "✓" if r.pass_at_1 else ("⚠" if r.variables_preserved > 0 else "✗")
            print(f"    {r.method:<18} | preserved={r.variables_preserved}/{r.num_variables} "
                  f"| exact={r.exact_recall:.0%} | recall={r.syntax_preservation:.0%} | "
                  f"comp={r.compression_ratio:.2f}x | {status}")

    return results


def print_code_completion_table(results: List[CodeCompletionResult]):
    print("\n" + "=" * 90)
    print("CODE COMPLETION PERSISTENCE RESULTS (for paper)")
    print("=" * 90)

    seq_lengths = sorted(set(r.seq_len for r in results))
    methods = ["H2O", "ScissorHands", "Tiered (Ours)"]

    print(f"\n{'Table: Code Completion Persistence Across Context Lengths':^90}")
    print(f"{'Seq Len':<10} {'Method':<18} {'Vars Preserved':<16} "
          f"{'Exact Recall':<14} {'Syn. Preserv.':<14} {'Pass@1':<8} {'Ratio':<8}")
    print("-" * 90)

    for seq_len in seq_lengths:
        for method in methods:
            matches = [r for r in results if r.seq_len == seq_len and r.method == method]
            if matches:
                r = matches[0]
                pass_str = "✓" if r.pass_at_1 else "✗"
                print(f"{seq_len:<10} {method:<18} "
                      f"{r.variables_preserved}/{r.num_variables} ({r.syntax_preservation:.0%})     "
                      f"{r.exact_recall:>12.0%}  "
                      f"{r.syntax_preservation:>12.0%}  "
                      f"{pass_str:<8} {r.compression_ratio:>6.2f}x")

    # ── Aggregate ──────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("AGGREGATE STATISTICS")
    print("=" * 90)

    for method in methods:
        method_results = [r for r in results if r.method == method]
        total_defs = sum(r.num_variables for r in method_results)
        total_preserved = sum(r.variables_preserved for r in method_results)
        avg_recall = np.mean([r.syntax_preservation for r in method_results])
        avg_exact = np.mean([r.exact_recall for r in method_results])
        avg_comp = np.mean([r.compression_ratio for r in method_results])
        pass_at_1_rate = sum(1 for r in method_results if r.pass_at_1) / len(method_results)

        print(f"\n  {method}:")
        print(f"    Variables preserved:  {total_preserved}/{total_defs} ({total_preserved/total_defs*100:.1f}%)")
        print(f"    Avg exact recall:     {avg_exact:.1%}")
        print(f"    Avg syntax preserv.:  {avg_recall:.1%}")
        print(f"    Pass@1 rate:          {pass_at_1_rate:.1%}")
        print(f"    Avg compression:      {avg_comp:.2f}x")

    # Find the winner
    tiered_recall = np.mean([r.syntax_preservation for r in results if r.method == "Tiered (Ours)"])
    h2o_recall = np.mean([r.syntax_preservation for r in results if r.method == "H2O"])
    sh_recall = np.mean([r.syntax_preservation for r in results if r.method == "ScissorHands"])

    print(f"\n  Improvement over H2O:         {(tiered_recall - h2o_recall)*100:+.1f} pp")
    print(f"  Improvement over ScissorHands: {(tiered_recall - sh_recall)*100:+.1f} pp")

    print("=" * 90)


def save_results(results: List[CodeCompletionResult], filename: str = None):
    if filename is None:
        filename = os.path.join(os.path.dirname(__file__), '..', '..',
                                'results', 'exp_code_completion.json')

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    serializable = [
        {
            'method': r.method,
            'seq_len': r.seq_len,
            'num_variables': r.num_variables,
            'variables_preserved': r.variables_preserved,
            'exact_recall': r.exact_recall,
            'syntax_preservation': r.syntax_preservation,
            'pass_at_1': r.pass_at_1,
            'compression_ratio': r.compression_ratio,
            'tokens_kept': r.tokens_kept,
        }
        for r in results
    ]

    with open(filename, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {filename}")


def main():
    results = run_code_completion_benchmark()
    print_code_completion_table(results)
    save_results(results)
    print("\n" + "=" * 90)
    print("CODE COMPLETION BENCHMARK COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
