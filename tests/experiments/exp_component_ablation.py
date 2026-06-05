"""
Component Ablation Test

Addresses the paper's acknowledged limitation:
"Our method conflates two effects: (1) tier assignment based on salience, 
and (2) mean pooling compression within tiers. Future work should ablate 
these components to isolate their individual contributions." (§6, item 3)

This test isolates:
1. Tier assignment effect: Does tiering by salience beat random assignment?
2. Pooling effect: Does salience-weighted pooling beat uniform pooling?

By comparing four configurations, we can attribute results to specific mechanisms.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ttkv import CacheConfig, TieredKVCache
from ttkv import compute_type_prior_retention


@dataclass
class AblationConfig:
    """Configuration for ablation test component."""
    name: str
    tier_assignment: str  # 'salience' or 'random'
    pooling: str  # 'salience_weighted' or 'uniform'
    description: str


def create_test_data(seq_len: int = 8192, batch_size: int = 1, num_heads: int = 12, head_dim: int = 64):
    """Create standardized test data for fair comparison."""
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    token_ids = torch.randint(0, 50000, (batch_size, seq_len))
    retention = compute_type_prior_retention(token_ids)
    return k, v, positions, token_ids, retention


def apply_tier_assignment(
    k: torch.Tensor,
    v: torch.Tensor,
    retention: torch.Tensor,
    positions: torch.Tensor,
    method: str = 'salience',
    config: CacheConfig = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Apply tier assignment (either salience-based or random).

    Args:
        method: 'salience' or 'random'
        config: CacheConfig with tier thresholds

    Returns:
        Tensors partitioned by tier, plus metadata
    """
    if config is None:
        config = CacheConfig()

    batch_size, num_heads, seq_len, head_dim = k.shape
    device = k.device
    tau = config.tau_threshold

    if method == 'salience':
        # Standard salience-based tiering
        protected_mask = retention > tau
    elif method == 'random':
        # Random tier assignment - same distribution as salience but uncorrelated with importance
        # This tests whether tier assignment itself provides value, or just salience
        torch.manual_seed(42)  # Reproducible randomness
        random_scores = torch.rand_like(retention)
        protected_mask = random_scores > tau
    else:
        raise ValueError(f"Unknown tier assignment method: {method}")

    # Partition into tiers
    unprotected_mask = ~protected_mask

    tier_data = {
        'k_tiers': [],
        'v_tiers': [],
        'pos_tiers': [],
        'tier_assignments': []
    }

    # Tier 0: Protected tokens
    if protected_mask.any():
        for b in range(batch_size):
            mask = protected_mask[b]
            if mask.any():
                tier_data['k_tiers'].append(k[b, :, mask, :])
                tier_data['v_tiers'].append(v[b, :, mask, :])
                tier_data['pos_tiers'].append(positions[b, mask])
                tier_data['tier_assignments'].extend([0] * mask.sum().item())

    # Tier 1 & 2: Position-based
    recent_mask = unprotected_mask.clone()
    for b in range(batch_size):
        recent_mask[b] = unprotected_mask[b] & (torch.arange(seq_len, device=device) < config.tier0_size)

    if recent_mask.any():
        for b in range(batch_size):
            mask = recent_mask[b]
            if mask.any():
                tier_data['k_tiers'].append(k[b, :, mask, :])
                tier_data['v_tiers'].append(v[b, :, mask, :])
                tier_data['pos_tiers'].append(positions[b, mask])
                tier_data['tier_assignments'].extend([0] * mask.sum().item())

    middle_mask = unprotected_mask.clone()
    for b in range(batch_size):
        idx = torch.arange(seq_len, device=device)
        middle_mask[b] = unprotected_mask[b] & (idx >= config.tier0_size) & (idx < config.tier1_size)

    if middle_mask.any():
        k_mid, v_mid, pos_mid = [], [], []
        for b in range(batch_size):
            mask = middle_mask[b]
            if mask.any():
                k_mid.append(k[b, :, mask, :])
                v_mid.append(v[b, :, mask, :])
                pos_mid.append(positions[b, mask])

        if k_mid:
            tier_data['k_tiers'].append(torch.stack(k_mid, dim=0) if len(k_mid) > 1 else k_mid[0].unsqueeze(0))
            tier_data['v_tiers'].append(torch.stack(v_mid, dim=0) if len(v_mid) > 1 else v_mid[0].unsqueeze(0))
            tier_data['pos_tiers'].append(torch.stack(pos_mid, dim=0) if len(pos_mid) > 1 else pos_mid[0].unsqueeze(0))
            tier_data['tier_assignments'].extend([1] * sum(m.sum().item() for m in [middle_mask[b] for b in range(batch_size)]))

    # Tier 2
    old_mask = unprotected_mask.clone()
    for b in range(batch_size):
        old_mask[b] = unprotected_mask[b] & (torch.arange(seq_len, device=device) >= config.tier1_size)

    if old_mask.any():
        k_old, v_old, pos_old = [], [], []
        for b in range(batch_size):
            mask = old_mask[b]
            if mask.any():
                k_old.append(k[b, :, mask, :])
                v_old.append(v[b, :, mask, :])
                pos_old.append(positions[b, mask])

        if k_old:
            tier_data['k_tiers'].append(torch.stack(k_old, dim=0) if len(k_old) > 1 else k_old[0].unsqueeze(0))
            tier_data['v_tiers'].append(torch.stack(v_old, dim=0) if len(v_old) > 1 else v_old[0].unsqueeze(0))
            tier_data['pos_tiers'].append(torch.stack(pos_old, dim=0) if len(pos_old) > 1 else pos_old[0].unsqueeze(0))
            tier_data['tier_assignments'].extend([2] * sum(m.sum().item() for m in [old_mask[b] for b in range(batch_size)]))

    return tier_data


def apply_pooling(
    k_tier: torch.Tensor,
    v_tier: torch.Tensor,
    pos_tier: torch.Tensor,
    compression_ratio: int,
    method: str = 'salience_weighted',
    retention: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply compression pooling (either salience-weighted or uniform).

    Args:
        method: 'salience_weighted' or 'uniform'
        retention: Salience scores for weighted pooling (ignored for uniform)

    Returns:
        Compressed k, v, positions
    """
    batch_size, num_heads, seq_len, head_dim = k_tier.shape

    if seq_len == 0:
        return k_tier, v_tier, pos_tier

    compressed_len = (seq_len + compression_ratio - 1) // compression_ratio
    k_out, v_out, pos_out = [], [], []

    for b in range(batch_size):
        k_batch, v_batch, pos_batch = [], [], []

        for i in range(compressed_len):
            start = i * compression_ratio
            end = min((i + 1) * compression_ratio, seq_len)

            k_chunk = k_tier[b, :, start:end, :]
            v_chunk = v_tier[b, :, start:end, :]
            pos_chunk = pos_tier[b, start:end]

            if method == 'salience_weighted' and retention is not None:
                # Salience-weighted mean pooling
                ret_chunk = retention[b, start:end]
                weights = F.softmax(ret_chunk, dim=0).unsqueeze(0).unsqueeze(-1)
                k_pooled = (k_chunk * weights).sum(dim=1)
                v_pooled = (v_chunk * weights).sum(dim=1)
                pos_pooled = (pos_chunk.float() * weights.squeeze()).sum().long()
            elif method == 'uniform':
                # Uniform mean pooling (no salience guidance)
                weights = torch.ones(end - start, device=k_chunk.device) / (end - start)
                weights = weights.unsqueeze(0).unsqueeze(-1)
                k_pooled = (k_chunk * weights).sum(dim=1)
                v_pooled = (v_chunk * weights).sum(dim=1)
                pos_pooled = (pos_chunk.float() * weights.squeeze()).sum().long()
            else:
                raise ValueError(f"Unknown pooling method: {method}")

            k_batch.append(k_pooled)
            v_batch.append(v_pooled)
            pos_batch.append(pos_pooled)

        k_out.append(torch.stack(k_batch, dim=1))
        v_out.append(torch.stack(v_batch, dim=1))
        pos_out.append(torch.stack(pos_batch))

    return torch.stack(k_out, dim=0), torch.stack(v_out, dim=0), torch.stack(pos_out, dim=0)


class AblationTieredCache:
    """Tiered cache with configurable components for ablation testing."""

    def __init__(self, config: CacheConfig, tier_assignment: str = 'salience', pooling: str = 'salience_weighted'):
        self.config = config
        self.tier_assignment = tier_assignment
        self.pooling = pooling
        self.clear()

    def clear(self):
        self.k_cache = []
        self.v_cache = []
        self.retention_scores = []
        self.positions = []
        self.total_tokens = 0

    def add(self, k: torch.Tensor, v: torch.Tensor, retention: torch.Tensor, positions: torch.Tensor):
        self.k_cache.append(k)
        self.v_cache.append(v)
        self.retention_scores.append(retention)
        self.positions.append(positions)
        self.total_tokens += k.size(2)

    def get_compressed_cache(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.k_cache:
            return None, None, None

        k_all = torch.cat(self.k_cache, dim=2)
        v_all = torch.cat(self.v_cache, dim=2)
        retention_all = torch.cat(self.retention_scores, dim=1)
        positions_all = torch.cat(self.positions, dim=1)

        batch_size, num_heads, total_len, head_dim = k_all.shape
        device = k_all.device
        tau = self.config.tau_threshold

        # Apply tier assignment (salience-based or random)
        if self.tier_assignment == 'salience':
            protected_mask = retention_all > tau
        else:  # random
            torch.manual_seed(42)
            random_scores = torch.rand_like(retention_all)
            protected_mask = random_scores > tau

        unprotected_mask = ~protected_mask

        k_tiers = []
        v_tiers = []
        pos_tiers = []

        # Tier 0: Protected
        if protected_mask.any():
            k_prot_list, v_prot_list, pos_prot_list = [], [], []
            for b in range(batch_size):
                mask = protected_mask[b]
                if mask.any():
                    k_prot_list.append(k_all[b, :, mask, :])
                    v_prot_list.append(v_all[b, :, mask, :])
                    pos_prot_list.append(positions_all[b, mask])

            if k_prot_list:
                k_tiers.append(self._stack_and_pad(k_prot_list))
                v_tiers.append(self._stack_and_pad(v_prot_list))
                pos_tiers.append(self._stack_and_pad_positions(pos_prot_list))

        # Tier 0: Recent
        recent_mask = unprotected_mask.clone()
        for b in range(batch_size):
            recent_mask[b] = unprotected_mask[b] & (torch.arange(total_len, device=device) < self.config.tier0_size)

        if recent_mask.any():
            k_rec_list, v_rec_list, pos_rec_list = [], [], []
            for b in range(batch_size):
                mask = recent_mask[b]
                if mask.any():
                    k_rec_list.append(k_all[b, :, mask, :])
                    v_rec_list.append(v_all[b, :, mask, :])
                    pos_rec_list.append(positions_all[b, mask])

            if k_rec_list:
                k_tiers.append(self._stack_and_pad(k_rec_list))
                v_tiers.append(self._stack_and_pad(v_rec_list))
                pos_tiers.append(self._stack_and_pad_positions(pos_rec_list))

        # Tier 1: Middle with pooling
        middle_mask = unprotected_mask.clone()
        for b in range(batch_size):
            idx = torch.arange(total_len, device=device)
            middle_mask[b] = unprotected_mask[b] & (idx >= self.config.tier0_size) & (idx < self.config.tier1_size)

        if middle_mask.any():
            k_mid_list, v_mid_list, pos_mid_list = [], [], []
            for b in range(batch_size):
                mask = middle_mask[b]
                if mask.any():
                    k_mid_list.append(k_all[b, :, mask, :])
                    v_mid_list.append(v_all[b, :, mask, :])
                    pos_mid_list.append(positions_all[b, mask])

            if k_mid_list:
                k_mid = self._stack_and_pad(k_mid_list)
                v_mid = self._stack_and_pad(v_mid_list)
                pos_mid = self._stack_and_pad_positions(pos_mid_list)
                ret_mid = retention_all[middle_mask].view(batch_size, -1)[:, :k_mid.size(2)]

                k_comp, v_comp, pos_comp = self._compress_tier(
                    k_mid, v_mid, ret_mid, pos_mid, self.config.tier1_compression
                )
                k_tiers.append(k_comp)
                v_tiers.append(v_comp)
                pos_tiers.append(pos_comp)

        # Tier 2: Old with aggressive pooling
        old_mask = unprotected_mask.clone()
        for b in range(batch_size):
            old_mask[b] = unprotected_mask[b] & (torch.arange(total_len, device=device) >= self.config.tier1_size)

        if old_mask.any():
            k_old_list, v_old_list, pos_old_list = [], [], []
            for b in range(batch_size):
                mask = old_mask[b]
                if mask.any():
                    k_old_list.append(k_all[b, :, mask, :])
                    v_old_list.append(v_all[b, :, mask, :])
                    pos_old_list.append(positions_all[b, mask])

            if k_old_list:
                k_old = self._stack_and_pad(k_old_list)
                v_old = self._stack_and_pad(v_old_list)
                pos_old = self._stack_and_pad_positions(pos_old_list)
                ret_old = retention_all[old_mask].view(batch_size, -1)[:, :k_old.size(2)]

                k_comp, v_comp, pos_comp = self._compress_tier(
                    k_old, v_old, ret_old, pos_old, self.config.tier2_compression
                )
                k_tiers.append(k_comp)
                v_tiers.append(v_comp)
                pos_tiers.append(pos_comp)

        if k_tiers:
            return torch.cat(k_tiers, dim=2), torch.cat(v_tiers, dim=2), torch.cat(pos_tiers, dim=1)
        else:
            return (torch.empty(batch_size, num_heads, 0, head_dim, device=device),
                    torch.empty(batch_size, num_heads, 0, head_dim, device=device),
                    torch.empty(batch_size, 0, device=device, dtype=torch.long))

    def _stack_and_pad(self, tensor_list: List[torch.Tensor]) -> torch.Tensor:
        if not tensor_list:
            return None

        if tensor_list[0].dim() == 3:
            max_len = max(t.shape[1] for t in tensor_list)
            padded = []
            for t in tensor_list:
                pad_len = max_len - t.shape[1]
                if pad_len > 0:
                    pad_shape = (t.shape[0], pad_len, t.shape[2])
                    t = torch.cat([t, torch.zeros(*pad_shape, device=t.device, dtype=t.dtype)], dim=1)
                padded.append(t)
            return torch.stack(padded, dim=0) if padded else None
        return None

    def _stack_and_pad_positions(self, tensor_list: List[torch.Tensor]) -> torch.Tensor:
        if not tensor_list:
            return None

        max_len = max(t.shape[0] for t in tensor_list)
        padded = []
        for t in tensor_list:
            pad_len = max_len - t.shape[0]
            if pad_len > 0:
                t = torch.cat([t, torch.zeros(pad_len, device=t.device, dtype=t.dtype)], dim=0)
            padded.append(t)
        return torch.stack(padded, dim=0) if padded else None

    def _compress_tier(self, k, v, retention, positions, ratio):
        if k is None or k.size(2) == 0:
            return k, v, positions

        batch_size, num_heads, seq_len, head_dim = k.shape

        if seq_len == 0:
            return k, v, positions

        compressed_len = (seq_len + ratio - 1) // ratio
        k_out, v_out, pos_out = [], [], []

        for b in range(batch_size):
            k_batch, v_batch, pos_batch = [], [], []

            for i in range(compressed_len):
                start = i * ratio
                end = min((i + 1) * ratio, seq_len)

                k_chunk = k[b, :, start:end, :]
                v_chunk = v[b, :, start:end, :]
                pos_chunk = positions[b, start:end]

                if self.pooling == 'salience_weighted':
                    ret_chunk = retention[b, start:end]
                    weights = F.softmax(ret_chunk, dim=0).unsqueeze(0).unsqueeze(-1)
                else:  # uniform
                    weights = torch.ones(end - start, device=k.device) / (end - start)
                    weights = weights.unsqueeze(0).unsqueeze(-1)

                k_pooled = (k_chunk * weights).sum(dim=1)
                v_pooled = (v_chunk * weights).sum(dim=1)
                pos_pooled = (pos_chunk.float() * weights.squeeze()).sum().long()

                k_batch.append(k_pooled)
                v_batch.append(v_pooled)
                pos_batch.append(pos_pooled)

            k_out.append(torch.stack(k_batch, dim=1))
            v_out.append(torch.stack(v_batch, dim=1))
            pos_out.append(torch.stack(pos_batch))

        return torch.stack(k_out, dim=0), torch.stack(v_out, dim=0), torch.stack(pos_out, dim=0)

    def get_stats(self) -> Dict:
        if not self.k_cache:
            return {'total_tokens': 0, 'compressed_tokens': 0, 'compression_ratio': 1.0}

        k_comp, _, _ = self.get_compressed_cache()
        compressed_len = k_comp.size(2) if k_comp is not None else 0

        return {
            'total_tokens': self.total_tokens,
            'compressed_tokens': compressed_len,
            'compression_ratio': self.total_tokens / max(compressed_len, 1)
        }


def run_ablation_test(seq_len: int = 8192, tau: float = 0.9) -> Dict:
    """
    Run component ablation test with four configurations.

    Configurations:
    1. Full: Salience-based tiering + Salience-weighted pooling
    2. Tier-Only: Salience-based tiering + Uniform pooling
    3. Pool-Only: Random tiering + Salience-weighted pooling
    4. Neither: Random tiering + Uniform pooling
    """
    print("="*80)
    print("COMPONENT ABLATION TEST")
    print("Isolating tier assignment vs pooling contributions")
    print("="*80)

    # Create test data
    k, v, positions, token_ids, retention = create_test_data(seq_len)
    print(f"\nTest data: {seq_len} tokens, τ={tau}")

    config = CacheConfig(tau_threshold=tau)

    # Baseline: Full cache (no compression)
    print("\n--- Baseline (no compression) ---")
    baseline_tokens = seq_len
    print(f"Tokens: {baseline_tokens}")

    # Configuration 1: Full method
    print("\n--- Configuration 1: FULL METHOD ---")
    print("Tier assignment: Salience-based")
    print("Pooling: Salience-weighted")

    cache1 = AblationTieredCache(config, tier_assignment='salience', pooling='salience_weighted')
    cache1.add(k, v, retention, positions)
    stats1 = cache1.get_stats()
    k_comp1, v_comp1, pos_comp1 = cache1.get_compressed_cache()

    print(f"Compression: {stats1['compression_ratio']:.2f}x")
    print(f"Tokens kept: {stats1['compressed_tokens']}")

    # Configuration 2: Tier assignment only
    print("\n--- Configuration 2: TIER ASSIGNMENT ONLY ---")
    print("Tier assignment: Salience-based")
    print("Pooling: Uniform (no salience weighting)")

    cache2 = AblationTieredCache(config, tier_assignment='salience', pooling='uniform')
    cache2.add(k, v, retention, positions)
    stats2 = cache2.get_stats()
    k_comp2, v_comp2, pos_comp2 = cache2.get_compressed_cache()

    print(f"Compression: {stats2['compression_ratio']:.2f}x")
    print(f"Tokens kept: {stats2['compressed_tokens']}")

    # Configuration 3: Pooling only
    print("\n--- Configuration 3: POOLING ONLY ---")
    print("Tier assignment: Random (uncorrelated with importance)")
    print("Pooling: Salience-weighted")

    cache3 = AblationTieredCache(config, tier_assignment='random', pooling='salience_weighted')
    cache3.add(k, v, retention, positions)
    stats3 = cache3.get_stats()
    k_comp3, v_comp3, pos_comp3 = cache3.get_compressed_cache()

    print(f"Compression: {stats3['compression_ratio']:.2f}x")
    print(f"Tokens kept: {stats3['compressed_tokens']}")

    # Configuration 4: Neither component
    print("\n--- Configuration 4: NEITHER COMPONENT ---")
    print("Tier assignment: Random")
    print("Pooling: Uniform")

    cache4 = AblationTieredCache(config, tier_assignment='random', pooling='uniform')
    cache4.add(k, v, retention, positions)
    stats4 = cache4.get_stats()
    k_comp4, v_comp4, pos_comp4 = cache4.get_compressed_cache()

    print(f"Compression: {stats4['compression_ratio']:.2f}x")
    print(f"Tokens kept: {stats4['compressed_tokens']}")

    # Calculate component contributions
    full_compression = stats1['compression_ratio']
    tier_only_compression = stats2['compression_ratio']
    pool_only_compression = stats3['compression_ratio']
    neither_compression = stats4['compression_ratio']

    # Calculate quality metrics (proxy: reconstruction error)
    def compute_quality(k_orig, v_orig, k_comp, v_comp):
        """Compute quality loss as MSE between original and compressed."""
        if k_comp is None or k_comp.size(2) == 0:
            return 1.0  # Max error

        # Downsample original to match compressed for comparison
        comp_len = k_comp.size(2)
        indices = torch.linspace(0, k_orig.size(2) - 1, comp_len).long()
        k_sampled = k_orig[:, :, indices, :]
        v_sampled = v_orig[:, :, indices, :]

        mse_k = F.mse_loss(k_comp, k_sampled).item()
        mse_v = F.mse_loss(v_comp, v_sampled).item()
        return (mse_k + mse_v) / 2

    quality_full = compute_quality(k, v, k_comp1, v_comp1)
    quality_tier_only = compute_quality(k, v, k_comp2, v_comp2)
    quality_pool_only = compute_quality(k, v, k_comp3, v_comp3)
    quality_neither = compute_quality(k, v, k_comp4, v_comp4)

    results = {
        'configurations': [
            {
                'name': 'Full',
                'tier_assignment': 'salience',
                'pooling': 'salience_weighted',
                'compression': full_compression,
                'quality_loss': quality_full
            },
            {
                'name': 'Tier-Only',
                'tier_assignment': 'salience',
                'pooling': 'uniform',
                'compression': tier_only_compression,
                'quality_loss': quality_tier_only
            },
            {
                'name': 'Pool-Only',
                'tier_assignment': 'random',
                'pooling': 'salience_weighted',
                'compression': pool_only_compression,
                'quality_loss': quality_pool_only
            },
            {
                'name': 'Neither',
                'tier_assignment': 'random',
                'pooling': 'uniform',
                'compression': neither_compression,
                'quality_loss': quality_neither
            }
        ],
        'component_contributions': {
            'tier_assignment_effect': tier_only_compression - neither_compression,
            'pooling_effect': pool_only_compression - neither_compression,
            'synergy_effect': full_compression - max(tier_only_compression, pool_only_compression),
            'total_improvement': full_compression - neither_compression
        }
    }

    return results


def analyze_results(results: Dict):
    """Analyze and report component contributions."""
    print("\n" + "="*80)
    print("ABLATION RESULTS TABLE")
    print("="*80)

    configs = results['configurations']
    contributions = results['component_contributions']

    print(f"\n{'Configuration':<20} | {'Tier':<10} | {'Pool':<10} | {'Compression':<12} | {'Quality Loss':<12}")
    print("-" * 80)

    for c in configs:
        tier_str = 'salience' if c['tier_assignment'] == 'salience' else 'random'
        pool_str = 'weighted' if c['pooling'] == 'salience_weighted' else 'uniform'
        print(f"{c['name']:<20} | {tier_str:<10} | {pool_str:<10} | {c['compression']:>10.2f}x | {c['quality_loss']:>10.4f}")

    print("\n" + "="*80)
    print("COMPONENT CONTRIBUTION ANALYSIS")
    print("="*80)

    print(f"\n{'Effect':<30} | {'Contribution':<15} | {'Interpretation'}")
    print("-" * 80)

    tier_effect = contributions['tier_assignment_effect']
    print(f"{'Tier assignment (salience vs random)':<30} | {tier_effect:>+13.2f}x | "
          f"{'Major' if abs(tier_effect) > 1.0 else 'Minor'} benefit of salience-based tiering")

    pool_effect = contributions['pooling_effect']
    print(f"{'Pooling (weighted vs uniform)':<30} | {pool_effect:>+13.2f}x | "
          f"{'Major' if abs(pool_effect) > 1.0 else 'Minor'} benefit of salience-weighted pooling")

    synergy = contributions['synergy_effect']
    print(f"{'Synergy (interaction effect)':<30} | {synergy:>+13.2f}x | "
          f"{'Positive' if synergy > 0 else 'Negative'} interaction between components")

    total = contributions['total_improvement']
    print(f"{'Total improvement over baseline':<30} | {total:>+13.2f}x | "
          f"Overall method contribution")

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    # Determine which component contributes more
    tier_pct = abs(tier_effect) / (abs(tier_effect) + abs(pool_effect)) * 100 if (abs(tier_effect) + abs(pool_effect)) > 0 else 0
    pool_pct = abs(pool_effect) / (abs(tier_effect) + abs(pool_effect)) * 100 if (abs(tier_effect) + abs(pool_effect)) > 0 else 0

    if tier_effect > pool_effect:
        print(f"✓ TIER ASSIGNMENT is the primary contributor ({tier_pct:.0f}% of improvement)")
        print("  Salience-based tiering provides most of the compression benefit")
    elif pool_effect > tier_effect:
        print(f"✓ SALIENCE-WEIGHTED POOLING is the primary contributor ({pool_pct:.0f}% of improvement)")
        print("  How we compress within tiers matters more than which tokens we compress")
    else:
        print("✓ BOTH COMPONENTS contribute equally")

    if synergy > 0.5:
        print(f"✓ STRONG SYNERGY between components (+{synergy:.2f}x additional)")
        print("  Components work better together than in isolation")
    elif synergy < -0.5:
        print(f"⚠ NEGATIVE SYNERGY ({synergy:.2f}x)")
        print("  Components may interfere with each other")
    else:
        print(f"△ Weak synergy ({synergy:.2f}x) - components are largely independent")

    baseline_compression = configs[3]['compression']  # Neither
    full_compression = configs[0]['compression']  # Full

    print(f"\n✓ Full method achieves {full_compression/baseline_compression:.2f}x better compression than baseline")
    print(f"  (Salience tiering + weighted pooling vs random tiering + uniform pooling)")

    print("\n" + "="*80)
    print("PAPER CLAIMS VALIDATION")
    print("="*80)

    if tier_effect > 0.5:
        print("✓ VALIDATED: Tier assignment based on salience provides significant benefit")
        print(f"  Random tiering with uniform pooling: {baseline_compression:.2f}x")
        print(f"  Salience tiering with uniform pooling: {configs[1]['compression']:.2f}x")
    else:
        print("✗ NOT VALIDATED: Tier assignment provides minimal benefit")

    if pool_effect > 0.2:
        print("✓ VALIDATED: Salience-weighted pooling improves over uniform pooling")
        print(f"  Random tiering with weighted pooling: {configs[2]['compression']:.2f}x")
        print(f"  Improvement: +{pool_effect:.2f}x")
    else:
        print("△ WEAK VALIDATION: Pooling method has minor effect")

    return results


def save_results(results: Dict, filename='../results/exp_component_ablation.json'):
    """Save results to JSON."""
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")


def main():
    """Run component ablation test."""
    print("="*80)
    print("COMPONENT ABLATION TEST")
    print("Isolating Tier Assignment vs Pooling Contributions")
    print("="*80)
    print("\nThis test addresses the paper's acknowledged limitation:")
    print('  "Our method conflates two effects: (1) tier assignment based on')
    print('   salience, and (2) mean pooling compression within tiers." (§6)')
    print("\nWe test four configurations to isolate each component's contribution:")
    print("  1. Full: Salience tiering + Weighted pooling")
    print("  2. Tier-Only: Salience tiering + Uniform pooling")
    print("  3. Pool-Only: Random tiering + Weighted pooling")
    print("  4. Neither: Random tiering + Uniform pooling")
    print("="*80)

    # Run ablation test
    results = run_ablation_test(seq_len=8192, tau=0.9)

    # Analyze results
    results = analyze_results(results)

    # Save results
    save_results(results)

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nConclusion: This ablation clarifies which architectural choices")
    print("contribute most to the reported 7.36x compression and 0.12% quality loss.")


if __name__ == "__main__":
    main()
