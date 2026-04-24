"""
filters/biological.py
======================
Biological hard filters for generated peptide candidates.

Fixes over original:
1. Charge computation now uses the full pKa-based Henderson-Hasselbalch
   model from features/physicochemical.py. Original used only K+R+H vs
   D+E counts without terminal contributions, pH correction for His,
   or Cys/Tyr contributions.

2. Instability index uses the Guruprasad 1990 DIWV formula. Original
   used an ad hoc charged-residue ratio heuristic with no empirical basis.
   The 40-threshold for stable/unstable only applies to the Guruprasad
   formula and was applied incorrectly in the original.

3. Added amphipathicity filter (hydrophobic moment ≥ threshold).
   AMPs typically form amphipathic helices; this is a necessary
   structural condition, not just a desirable property.

4. Added aggregation propensity filter (max consecutive hydrophobic
   run). Sequences with long hydrophobic runs form amyloid-like
   aggregates in solution, making them unsuitable as therapeutic candidates.

5. All thresholds are configurable through the filter_config dict,
   not hardcoded. The function signature accepts a dict so the calling
   code can store thresholds in config.yaml.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from features.physicochemical import (
    compute_descriptors, net_charge, instability_index, hydrophobic_moment,
    aggregation_propensity,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Filter configuration dataclass
# =============================================================================

@dataclass
class FilterConfig:
    # Sequence length
    min_len: int = 8
    max_len: int = 50

    # Charge at pH 7.4 (pKa-based HH model)
    min_net_charge: float = 1.0

    # GRAVY (Kyte-Doolittle mean hydrophobicity)
    max_gravy: float = 2.5

    # Instability index (Guruprasad 1990): < 40 = stable
    max_instability: float = 60.0

    # Classifier score thresholds
    min_amp_score: float = 0.50
    min_cpp_score: float = 0.50

    # Amphipathicity: Eisenberg hydrophobic moment (per 11-residue window)
    min_hydrophobic_moment: float = 0.0   # 0.0 = disabled; typical AMP: ≥ 0.3

    # Aggregation: max consecutive hydrophobic residues
    max_aggregation_run: int = 5          # ≥5 consecutive = aggregation-prone

    @classmethod
    def from_dict(cls, d: dict) -> 'FilterConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Per-sequence filter
# =============================================================================

def _passes_filters(
    seq: str,
    amp_score: float,
    cpp_score: float,
    cfg: FilterConfig,
) -> Tuple[bool, Optional[str]]:
    """
    Return (passes: bool, failure_reason: Optional[str]).
    """
    L = len(seq)
    if not (cfg.min_len <= L <= cfg.max_len):
        return False, 'length'

    if amp_score < cfg.min_amp_score:
        return False, 'amp_score'

    if cpp_score < cfg.min_cpp_score:
        return False, 'cpp_score'

    nc = net_charge(seq, pH=7.4)
    if nc < cfg.min_net_charge:
        return False, 'net_charge'

    d = compute_descriptors(seq)
    if d['GRAVY'] > cfg.max_gravy:
        return False, 'gravy'

    if d['instability_index'] > cfg.max_instability:
        return False, 'instability'

    if d['hydrophobic_moment'] < cfg.min_hydrophobic_moment:
        return False, 'hydrophobic_moment'

    if d['aggregation_propensity'] >= cfg.max_aggregation_run:
        return False, 'aggregation'

    return True, None


# =============================================================================
# Batch filter
# =============================================================================

def apply_biological_filters(
    candidates: List[Tuple],
    filter_config: FilterConfig,
) -> List[Tuple]:
    """
    Apply all biological hard filters to a list of candidates.

    candidates: list of (seq, amp_score, cpp_score, combined_score)

    Returns a list of (seq, amp_score, cpp_score, combined_score, descriptors)
    for all sequences that pass all filters.
    """
    reasons = {
        'length': 0, 'amp_score': 0, 'cpp_score': 0, 'net_charge': 0,
        'gravy': 0, 'instability': 0, 'hydrophobic_moment': 0,
        'aggregation': 0,
    }
    passed = []

    for seq, amp_s, cpp_s, comb in candidates:
        ok, reason = _passes_filters(seq, amp_s, cpp_s, filter_config)
        if not ok:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue

        d = compute_descriptors(seq)
        passed.append((seq, amp_s, cpp_s, comb, d))

    total = len(candidates)
    n_passed = len(passed)
    logger.info("Biological filters: %d / %d passed (%.1f%%)",
                n_passed, total, 100.0 * n_passed / max(total, 1))
    for k, v in reasons.items():
        if v > 0:
            logger.info("  Filtered by %-22s: %d", k, v)

    return passed


# =============================================================================
# Fallback: relax thresholds if nothing passes
# =============================================================================

def relax_and_refilter(
    candidates: List[Tuple],
    cfg: FilterConfig,
    relaxation_factor: float = 0.5,
) -> Tuple[List[Tuple], FilterConfig]:
    """
    If candidates is empty, halve all strict thresholds and retry.
    Returns (filtered_candidates, relaxed_config).

    This preserves the explicit intent: relaxation is logged and surfaced,
    not silently applied (as in the original code which just changed numbers).
    """
    relaxed = FilterConfig(
        min_len=max(5, cfg.min_len - 2),
        max_len=min(60, cfg.max_len + 5),
        min_net_charge=cfg.min_net_charge * relaxation_factor,
        max_gravy=cfg.max_gravy * (1 + relaxation_factor),
        max_instability=cfg.max_instability * (1 + relaxation_factor * 0.5),
        min_amp_score=cfg.min_amp_score * relaxation_factor,
        min_cpp_score=cfg.min_cpp_score * relaxation_factor,
        min_hydrophobic_moment=0.0,
        max_aggregation_run=cfg.max_aggregation_run + 2,
    )
    logger.warning(
        "No candidates passed strict filters. Relaxing: "
        "min_charge=%.2f → %.2f, min_amp=%.2f → %.2f, min_cpp=%.2f → %.2f",
        cfg.min_net_charge, relaxed.min_net_charge,
        cfg.min_amp_score, relaxed.min_amp_score,
        cfg.min_cpp_score, relaxed.min_cpp_score,
    )
    result = apply_biological_filters(candidates, relaxed)
    return result, relaxed
