"""Auto-configuration for TTKV tier geometry.

Given retention scores marking critical tokens, automatically determines
optimal tier sizes and tau to maximize compression while maintaining
retrieval quality through signal-to-noise optimization.
"""
import torch
from ttkv import CacheConfig


def auto_config(
    n_tokens: int,
    protected_indices: torch.Tensor,
    target_signal_pct: float = 3.5,
    min_tier0: int = 8,
    min_tau: float = 0.85,
) -> CacheConfig:
    """Determine optimal tier geometry for a given context and set of protected tokens.

    The key insight: compression improves retrieval by reducing noise (filler)
    while the retention floor preserves signal (critical tokens). The sweet spot
    is where signal is ~3-5% of the compressed cache.

    Args:
        n_tokens: Total tokens in the context
        protected_indices: Boolean mask [n_tokens] — True = protect this token
        target_signal_pct: Desired signal percentage in compressed cache (default 3.5%)
        min_tier0: Minimum Tier 0 size (recent window)
        min_tau: Minimum tau threshold for protection

    Returns:
        CacheConfig with optimized tier0_size, tier1_size, tau_threshold
    """
    n_protected = int(protected_indices.sum().item())

    if n_protected == 0:
        return CacheConfig(
            tau_threshold=min_tau,
            tier0_size=min_tier0,
            tier1_size=2048,
        )

    # Tier 0: protected tokens + recent window
    tier0_size = max(min_tier0, n_protected)
    tau = min_tau

    # Search for tier1_size that gives target signal percentage
    # The compressed token count for a given tier1_size:
    #   tier0: tier0_size + n_protected (some overlap with recent window)
    #   tier1: (tier1_size - tier0_size) / 4
    #   tier2: max(0, n_tokens - tier1_size) / 16
    candidates = [128, 256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048]
    best_t1 = 2048
    best_score = float("inf")

    for t1 in candidates:
        if t1 <= tier0_size:
            continue

        tier0_tokens = tier0_size
        tier1_tokens = max(0, (t1 - tier0_size)) // 4
        tier2_tokens = max(0, (n_tokens - t1)) // 16
        total_compressed = tier0_tokens + tier1_tokens + tier2_tokens

        if total_compressed == 0:
            continue

        signal_pct = n_protected / total_compressed * 100
        score = abs(signal_pct - target_signal_pct)

        if score < best_score:
            best_score = score
            best_t1 = t1

    return CacheConfig(
        tau_threshold=tau,
        tier0_size=tier0_size,
        tier1_size=best_t1,
    )


def build_retention_mask(
    n_tokens: int,
    needle_ranges: list[tuple[int, int]],
    query_ranges: list[tuple[int, int]] | None = None,
    recent_window: int = 16,
    base_score: float = 0.3,
    protect_score: float = 1.0,
) -> torch.Tensor:
    """Build a retention tensor marking critical token ranges.

    Args:
        n_tokens: Total tokens
        needle_ranges: List of (start, end) ranges for needle tokens
        query_ranges: Optional list of (start, end) ranges for query tokens
        recent_window: How many recent tokens to protect
        base_score: Score for non-critical tokens
        protect_score: Score for protected tokens

    Returns:
        Retention tensor [1, n_tokens] with high scores on protected ranges
    """
    ret = torch.ones(1, n_tokens) * base_score

    for start, end in needle_ranges:
        ret[0, start:end] = protect_score

    for start, end in (query_ranges or []):
        ret[0, start:end] = protect_score

    if recent_window > 0:
        ret[0, max(0, n_tokens - recent_window):] = protect_score

    return ret
