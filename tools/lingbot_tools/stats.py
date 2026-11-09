"""Paired statistics used for A/B decisions.

wilcoxon_gate keeps the acceptance semantics of agenticx.rl.acceptance.wilcoxon_gate
(AgenticX main@779a400e) so verdicts stay comparable with the AgenticX flywheel gate:
zero differences dropped, fewer than 2 non-zero pairs -> reject, mean gain required,
default alpha 0.1.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value.

    b: pairs where baseline failed and candidate succeeded.
    c: pairs where baseline succeeded and candidate failed.
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


@dataclass
class AcceptDecision:
    accepted: bool
    statistic: float
    p_value: float
    mean_baseline: float
    mean_candidate: float
    n_pairs: int
    reason: str

    def to_json(self) -> dict:
        return asdict(self)


def wilcoxon_gate(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    alpha: float = 0.1,
    min_improve: float = 0.0,
) -> AcceptDecision:
    from scipy.stats import wilcoxon

    if len(baseline) != len(candidate):
        raise ValueError(f"paired lengths differ: baseline={len(baseline)} candidate={len(candidate)}")
    if len(baseline) == 0:
        raise ValueError("empty samples cannot be gated")
    mb = float(sum(baseline) / len(baseline))
    mc = float(sum(candidate) / len(candidate))

    n_pairs = sum(1 for b, c in zip(baseline, candidate) if c != b)
    if n_pairs < 2:
        return AcceptDecision(False, 0.0, 1.0, mb, mc, n_pairs, "insufficient_nonzero_pairs")
    stat, p = wilcoxon(candidate, baseline, zero_method="wilcox")
    if mc < mb + min_improve:
        return AcceptDecision(False, float(stat), float(p), mb, mc, n_pairs, "no_mean_gain")
    accepted = bool(p < alpha)
    return AcceptDecision(
        accepted, float(stat), float(p), mb, mc, n_pairs,
        "significant_gain" if accepted else "not_significant",
    )


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = 10000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0])
    return float(lo), float(hi)
