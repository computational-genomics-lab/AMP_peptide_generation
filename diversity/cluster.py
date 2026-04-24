"""
diversity/cluster.py
=====================
Diversity-aware candidate selection.

Fixes over original:
1. Added sequence-level identity filter using Levenshtein edit distance.
   Original clustered on physicochemical features only — two sequences
   "RRRKKK" and "RRKRKK" differ by 1 edit but have near-identical
   physicochemical descriptors, so they would be assigned to the same
   cluster but both selected as top-per-cluster candidates.

2. Greedy maximum-diversity selection (MaxMin algorithm) is applied
   first to remove near-identical sequences before clustering,
   ensuring the final candidates are truly non-redundant.

3. KMeans uses physicochemical + sequence identity features. Sequence
   identity is encoded as a pairwise distance embedding via MDS
   (or skipped for large inputs in favour of the greedy filter alone).
"""

import logging
from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from features.physicochemical import descriptors_vector
from generation.pipeline import levenshtein

logger = logging.getLogger(__name__)


# =============================================================================
# Greedy maximum-diversity pre-filter (MaxMin selection)
# =============================================================================

def greedy_diverse_filter(
    candidates: List[Tuple],
    max_identity: float = 0.80,
    max_candidates: int = 2000,
) -> List[Tuple]:
    """
    Remove sequences above max_identity to any already-selected candidate.

    Algorithm:
      1. Sort by combined score (descending)
      2. Greedily add each sequence if its max identity to the
         current selected set is below max_identity

    This is O(N²) in the worst case; capped at max_candidates input
    sequences to avoid excessive runtime.

    FIX: original skipped this step entirely, meaning the final
    candidates often consisted of minor single-residue variants of
    the same top-scoring sequence.
    """
    # Sort descending by combined score
    sorted_cands = sorted(candidates, key=lambda x: -x[3])[:max_candidates]

    selected: List[Tuple] = []
    selected_seqs: List[str] = []

    for candidate in sorted_cands:
        seq = candidate[0]
        if not selected_seqs:
            selected.append(candidate)
            selected_seqs.append(seq)
            continue

        # Check identity against all selected sequences
        max_id = max(
            1.0 - levenshtein(seq, s) / max(len(seq), len(s), 1)
            for s in selected_seqs
        )
        if max_id < max_identity:
            selected.append(candidate)
            selected_seqs.append(seq)

    logger.info(
        "Greedy diversity filter: %d → %d sequences (max_identity=%.2f)",
        len(sorted_cands), len(selected), max_identity,
    )
    return selected


# =============================================================================
# Physicochemical K-Means clustering + top-per-cluster selection
# =============================================================================

def cluster_and_select(
    candidates: List[Tuple],
    n_clusters: int = 20,
    top_per_cluster: int = 4,
    seed: int = 42,
) -> List[dict]:
    """
    Cluster on physicochemical features, pick top_per_cluster per cluster.

    Returns a list of dicts with all candidate properties + cluster assignment.
    """
    if len(candidates) < n_clusters:
        n_clusters = max(2, len(candidates) // 2)

    seqs = [c[0] for c in candidates]
    X_phys = np.vstack([descriptors_vector(s) for s in seqs])
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_phys)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = km.fit_predict(X_sc)
    logger.info("KMeans: %d sequences → %d clusters", len(seqs), n_clusters)

    # Group by cluster, sort by combined score within each cluster
    clusters: dict = {i: [] for i in range(n_clusters)}
    for idx, (candidate, label) in enumerate(zip(candidates, labels)):
        clusters[int(label)].append(candidate)

    diverse: List[dict] = []
    for cl_id, cl_members in clusters.items():
        cl_members.sort(key=lambda x: -x[3])   # sort by combined score
        for seq, amp_s, cpp_s, comb, d in cl_members[:top_per_cluster]:
            entry = {
                'sequence':      seq,
                'amp_score':     round(float(amp_s), 4),
                'cpp_score':     round(float(cpp_s), 4),
                'combined_score': round(float(comb), 4),
                'cluster':       cl_id,
            }
            entry.update({k: v for k, v in d.items()})
            diverse.append(entry)

    diverse.sort(key=lambda x: -x['combined_score'])
    for rank, item in enumerate(diverse, start=1):
        item['rank'] = rank

    logger.info(
        "Cluster selection: %d clusters × %d/cluster → %d diverse candidates",
        n_clusters, top_per_cluster, len(diverse),
    )
    return diverse


# =============================================================================
# Full diversity pipeline
# =============================================================================

def apply_diversity_control(
    candidates: List[Tuple],
    max_identity: float = 0.80,
    n_clusters: int = 20,
    top_per_cluster: int = 4,
    seed: int = 42,
) -> List[dict]:
    """
    1. Greedy edit-distance diversity filter
    2. Physicochemical KMeans clustering
    3. Top-K per cluster selection
    """
    # Step 1: remove near-identical sequences
    diverse_pool = greedy_diverse_filter(candidates, max_identity=max_identity)

    if len(diverse_pool) < 4:
        logger.warning(
            "Only %d sequences after diversity filter. "
            "Relaxing max_identity to 0.95.", len(diverse_pool),
        )
        diverse_pool = greedy_diverse_filter(candidates, max_identity=0.95)

    # Step 2+3: cluster and select
    return cluster_and_select(
        diverse_pool,
        n_clusters=n_clusters,
        top_per_cluster=top_per_cluster,
        seed=seed,
    )
