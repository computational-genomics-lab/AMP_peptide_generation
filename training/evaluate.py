"""
training/evaluate.py
=====================
Evaluation of generated peptides against reference distributions.

Provides:
- Score distribution comparison (generated vs. training positives vs. random)
- Novelty metrics (edit distance to nearest training sequence)
- Baseline: random peptide population at the same length distribution
"""

import logging
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from generation.pipeline import levenshtein
from features.physicochemical import AA_LIST, is_canonical

logger = logging.getLogger(__name__)


# =============================================================================
# Random baseline generation
# =============================================================================

def generate_random_peptides(
    n: int,
    length_distribution: List[int],
    seed: int = 42,
) -> List[str]:
    """
    Generate n random peptides drawn from uniform AA distribution
    at lengths sampled from length_distribution.
    Used as a null-hypothesis baseline.
    """
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(n):
        L = int(rng.choice(length_distribution))
        seqs.append(''.join(rng.choice(AA_LIST, size=L)))
    return seqs


# =============================================================================
# Novelty metrics
# =============================================================================

def nearest_training_distance(
    generated_seqs: List[str],
    training_seqs: List[str],
    n_sample: int = 200,
) -> np.ndarray:
    """
    For each generated sequence, compute the normalised edit distance
    to the nearest training sequence.

    Returns an array of shape (len(generated_seqs),) with values in [0, 1].
    0.0 = identical to a training sequence (memorised)
    1.0 = maximally different

    n_sample: subsample training sequences for speed on large datasets.
    """
    rng = np.random.default_rng(0)
    if len(training_seqs) > n_sample:
        ref = list(rng.choice(len(training_seqs), size=n_sample, replace=False))
        ref_seqs = [training_seqs[i] for i in ref]
    else:
        ref_seqs = training_seqs

    distances = []
    for gen_seq in generated_seqs:
        min_dist = min(
            levenshtein(gen_seq, ref) / max(len(gen_seq), len(ref), 1)
            for ref in ref_seqs
        )
        distances.append(min_dist)
    return np.array(distances)


# =============================================================================
# Score distribution comparison
# =============================================================================

def compare_score_distributions(
    generated_seqs: List[str],
    training_pos_seqs: List[str],
    predict_amp: Callable,
    predict_cpp: Callable,
    output_path: Optional[str] = None,
    n_random: int = 500,
) -> pd.DataFrame:
    """
    Compare AMP and CPP score distributions across three populations:
    1. Generated sequences (from CVAE + optimisation)
    2. Training positives (experimental AMPs used for generator training)
    3. Random baseline (uniform AA composition, matched length distribution)

    Reports summary statistics and optionally saves to CSV.
    """
    train_lengths = [len(s) for s in training_pos_seqs]
    random_seqs   = generate_random_peptides(n_random, train_lengths, seed=0)
    random_seqs   = [s for s in random_seqs if is_canonical(s)]

    rows = []
    for population, seqs in [
        ('generated', generated_seqs),
        ('training_positives', training_pos_seqs[:n_random]),
        ('random_baseline', random_seqs),
    ]:
        if not seqs:
            continue
        amp_s = predict_amp(seqs)
        cpp_s = predict_cpp(seqs)
        comb  = np.sqrt(amp_s * cpp_s)
        rows.append({
            'population': population,
            'n': len(seqs),
            'amp_mean': float(amp_s.mean()),
            'amp_median': float(np.median(amp_s)),
            'amp_std':  float(amp_s.std()),
            'cpp_mean': float(cpp_s.mean()),
            'cpp_median': float(np.median(cpp_s)),
            'cpp_std':  float(cpp_s.std()),
            'combined_mean': float(comb.mean()),
            'combined_median': float(np.median(comb)),
        })
        logger.info(
            "%-22s  n=%4d  AMP=%.3f±%.3f  CPP=%.3f±%.3f  Comb=%.3f",
            population, len(seqs),
            amp_s.mean(), amp_s.std(),
            cpp_s.mean(), cpp_s.std(),
            comb.mean(),
        )

    df = pd.DataFrame(rows)
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info("Score comparison saved: %s", output_path)
    return df


# =============================================================================
# Novelty summary
# =============================================================================

def novelty_report(
    generated_seqs: List[str],
    training_seqs: List[str],
    output_path: Optional[str] = None,
) -> dict:
    """
    Compute novelty statistics for generated sequences.

    A sequence with distance 0.0 to the training set was memorised.
    High novelty (distance > 0.5) with high scores indicates genuine
    generation beyond interpolation.
    """
    distances = nearest_training_distance(generated_seqs, training_seqs)
    memorised = (distances < 0.05).sum()
    novel     = (distances > 0.5).sum()

    report = {
        'n_generated': len(generated_seqs),
        'n_memorised': int(memorised),
        'n_novel':     int(novel),
        'pct_memorised': round(100.0 * memorised / max(len(generated_seqs), 1), 2),
        'pct_novel':     round(100.0 * novel     / max(len(generated_seqs), 1), 2),
        'mean_distance': float(distances.mean()),
        'min_distance':  float(distances.min()),
        'p25_distance':  float(np.percentile(distances, 25)),
        'median_distance': float(np.median(distances)),
    }

    logger.info(
        "Novelty: mean dist=%.3f | memorised=%d (%.1f%%) | novel=%d (%.1f%%)",
        report['mean_distance'],
        report['n_memorised'], report['pct_memorised'],
        report['n_novel'],     report['pct_novel'],
    )

    if memorised / max(len(generated_seqs), 1) > 0.20:
        logger.warning(
            "%.1f%% of generated sequences are near-identical to training data. "
            "Consider increasing CVAE temperature or reducing training epochs.",
            report['pct_memorised'],
        )

    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info("Novelty report saved: %s", output_path)

    return report
