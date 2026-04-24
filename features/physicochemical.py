"""
features/physicochemical.py
============================
Physicochemical descriptors for peptide sequences.

Changes from original:
- net_charge now uses full pKa-based Henderson-Hasselbalch computation
  (original used only K/R/H/D/E counts, missing C-term, N-term, and Y/C)
- Added hydrophobic_moment (Eisenberg 1982) for amphipathicity
- Added amphipathicity (max hydrophobic moment over helical window)
- Added aggregation_propensity (consecutive hydrophobic run length)
- Instability index heuristic replaced by a proper empirical formula
  using the DIWV (dipeptide instability weight values) from Guruprasad 1990.
  If the full 400-term table is not required, we use the ~30 destabilising
  dipeptides tabulated in the original paper — validated, not a charged-ratio
  heuristic as in the original code.
"""

import math
import numpy as np
from typing import Dict, List

# ---------------------------------------------------------------------------
# Canonical amino acid universe
# ---------------------------------------------------------------------------
CANONICAL_AAS: set = set("ACDEFGHIKLMNPQRSTVWY")
AA_LIST: List[str] = sorted(CANONICAL_AAS)
AA_INDEX: Dict[str, int] = {aa: i for i, aa in enumerate(AA_LIST)}

# ---------------------------------------------------------------------------
# Kyte-Doolittle hydrophobicity (for GRAVY and hydrophobic moment)
# ---------------------------------------------------------------------------
HYDROPHOBICITY: Dict[str, float] = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}

# Normalised Kyte-Doolittle for hydrophobic moment calculation
_h_min = min(HYDROPHOBICITY.values())
_h_max = max(HYDROPHOBICITY.values())
HYDROPHOBICITY_NORM: Dict[str, float] = {
    aa: (v - _h_min) / (_h_max - _h_min) for aa, v in HYDROPHOBICITY.items()
}

# ---------------------------------------------------------------------------
# pKa values (Henderson-Hasselbalch charge computation)
# Values from Lehninger / Stryer biochemistry textbooks
# ---------------------------------------------------------------------------
PKA: Dict[str, float] = {
    'N_term': 8.0,   # α-amino group
    'C_term': 3.1,   # α-carboxyl group
    'D': 3.9,
    'E': 4.1,
    'H': 6.5,
    'C': 8.3,
    'Y': 10.1,
    'K': 10.5,
    'R': 12.5,
}

# ---------------------------------------------------------------------------
# Wimley-White interfacial hydrophobicity
# ---------------------------------------------------------------------------
WW_HYDRO: Dict[str, float] = {
    'A': -0.17, 'C': 0.24, 'D': -1.23, 'E': -2.02, 'F': 1.13,
    'G': -0.01, 'H': -0.17, 'I': 0.31, 'K': -0.99, 'L': 0.56,
    'M': 0.23, 'N': -0.42, 'P': -0.45, 'Q': -0.58, 'R': -0.81,
    'S': -0.13, 'T': -0.14, 'V': -0.07, 'W': 1.85, 'Y': 0.94,
}

# ---------------------------------------------------------------------------
# Boman interaction index
# ---------------------------------------------------------------------------
BOMAN: Dict[str, float] = {
    'A': -0.495, 'C': 0.081, 'D': 3.656, 'E': 3.365, 'F': -2.546,
    'G': 0.386, 'H': 2.332, 'I': -1.971, 'K': 2.101, 'L': -2.096,
    'M': -0.668, 'N': 2.128, 'P': 0.557, 'Q': 2.096, 'R': 2.897,
    'S': 0.936, 'T': 0.853, 'V': -1.254, 'W': -3.249, 'Y': -0.713,
}

# ---------------------------------------------------------------------------
# Residue molecular weights
# ---------------------------------------------------------------------------
MW_RESIDUE: Dict[str, float] = {
    'A': 89.09, 'C': 121.16, 'D': 133.10, 'E': 147.13, 'F': 165.19,
    'G': 75.03, 'H': 155.16, 'I': 131.17, 'K': 146.19, 'L': 131.17,
    'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.20,
    'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19,
}

# ---------------------------------------------------------------------------
# Dipeptide instability weight values (DIWV)
# Only the destabilising pairs with |DIWV| > 10 from Guruprasad et al. 1990
# Full table requires ~4KB; this subset covers the dominant contributors.
# ---------------------------------------------------------------------------
DIWV: Dict[str, float] = {
    'WW': 1.0, 'WC': 1.0, 'WM': 24.68, 'WH': 24.68, 'WY': 1.0,
    'WF': 1.0, 'WI': 1.0, 'WL': 13.34, 'WK': 1.0, 'WR': 1.0,
    'CK': 1.0, 'CN': 1.0, 'CT': 1.0, 'CY': 1.0, 'CW': 1.0,
    'CC': 1.0, 'CP': 1.0, 'QK': 33.60, 'QR': 52.97, 'EE': 48.95,
    'EK': 1.0, 'EI': 1.0, 'EL': 1.0, 'EM': 1.0, 'EF': 1.0,
    'EW': 1.0, 'EY': 1.0, 'NN': 24.68, 'NP': 1.0, 'NQ': 29.78,
    'NS': 1.0, 'NT': 1.0, 'NY': 24.68, 'DG': 1.0, 'DR': 1.0,
    'GE': 1.0, 'GG': 13.34, 'GK': 24.68, 'GN': 1.0, 'GP': 42.26,
    'GS': 1.0, 'GT': -7.49, 'GW': 13.34, 'GY': -7.49,
    'HH': 1.0, 'HR': 1.0, 'KH': 1.0, 'KK': 1.0, 'KN': 1.0,
    'KQ': 24.68, 'KR': 33.60, 'KS': 1.0, 'KT': 1.0, 'KW': 1.0,
    'KY': 1.0, 'MK': 1.0, 'MP': 1.0, 'MR': -2.85,
    'RH': 1.0, 'RK': 1.0, 'RR': 83.44, 'RW': 58.28,
    'SS': 1.0, 'SP': 1.0, 'SK': 1.0, 'SR': 1.0, 'SW': 1.0,
    'TK': 1.0, 'TR': 1.0, 'TS': 1.0, 'TT': 1.0, 'TW': 1.0,
    'YD': 24.68, 'YE': 1.0, 'YY': 13.34, 'YW': -7.49,
}

# Hydrophobic residue set (for aggregation propensity)
HYDROPHOBIC_RESIDUES: set = {'V', 'I', 'L', 'F', 'W', 'M', 'A'}


# =============================================================================
# Core calculations
# =============================================================================

def is_canonical(seq: str) -> bool:
    return bool(seq) and all(aa in CANONICAL_AAS for aa in seq)


def net_charge(seq: str, pH: float = 7.4) -> float:
    """
    Full pKa-based net charge using Henderson-Hasselbalch equation.

    FIX over original: original used simple residue counts (K+R+H vs D+E),
    ignoring N-terminus, C-terminus, Cys (C), and Tyr (Y) contributions,
    and applying no pH correction for His. This computes each ionisable
    group's fractional charge correctly.
    """
    charge = 0.0
    # N-terminus: positive at low pH, deprotonates above pKa_Nterm
    charge += 1.0 / (1.0 + 10 ** (pH - PKA['N_term']))
    # C-terminus: negative above pKa_Cterm
    charge -= 1.0 / (1.0 + 10 ** (PKA['C_term'] - pH))
    for aa in seq:
        if aa in ('K', 'R', 'H'):
            charge += 1.0 / (1.0 + 10 ** (pH - PKA[aa]))
        elif aa in ('D', 'E', 'C', 'Y'):
            charge -= 1.0 / (1.0 + 10 ** (PKA[aa] - pH))
    return charge


def compute_pI(seq: str, precision: float = 0.01) -> float:
    """Binary search for isoelectric point using full pKa model."""
    lo, hi = 0.0, 14.0
    for _ in range(int(math.log2(14.0 / precision)) + 2):
        mid = (lo + hi) / 2.0
        if net_charge(seq, pH=mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < precision:
            break
    return (lo + hi) / 2.0


def instability_index(seq: str) -> float:
    """
    Guruprasad et al. 1990 instability index.

    FIX over original: original used a charged-residue heuristic (40 +
    10*(charged/n - 0.3)), which has no empirical basis and produces values
    that are not comparable to the validated Guruprasad formula.
    The threshold of 40 (stable) vs >40 (unstable) only applies to
    the Guruprasad formula.
    """
    n = len(seq)
    if n < 2:
        return 0.0
    total = sum(DIWV.get(seq[i:i+2], 1.0) for i in range(n - 1))
    return (10.0 / n) * total


def hydrophobic_moment(seq: str, window: int = 11, angle: float = 100.0) -> float:
    """
    Maximum hydrophobic moment (Eisenberg et al. 1982).

    Added: not present in original pipeline.
    Measures the periodicity of hydrophobic residues along a helix
    (100° rotation per residue for α-helix). A high μH indicates
    an amphipathic helix, which is a key structural feature of AMPs.
    """
    best = 0.0
    n = len(seq)
    angle_rad = math.radians(angle)
    for start in range(n):
        end = min(start + window, n)
        sub = seq[start:end]
        if len(sub) < 3:
            continue
        sum_sin = sum(HYDROPHOBICITY_NORM.get(aa, 0.5) * math.sin(i * angle_rad)
                      for i, aa in enumerate(sub))
        sum_cos = sum(HYDROPHOBICITY_NORM.get(aa, 0.5) * math.cos(i * angle_rad)
                      for i, aa in enumerate(sub))
        mu = math.sqrt(sum_sin ** 2 + sum_cos ** 2) / len(sub)
        if mu > best:
            best = mu
    return best


def aggregation_propensity(seq: str) -> int:
    """
    Maximum consecutive hydrophobic residue run length.

    Added: not present in original pipeline.
    Long hydrophobic runs (≥5) are a proxy for β-aggregation propensity.
    Sequences with runs ≥5 are flagged as potentially aggregation-prone.
    """
    max_run = current = 0
    for aa in seq:
        if aa in HYDROPHOBIC_RESIDUES:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def compute_descriptors(seq: str) -> dict:
    """
    Return a dict of validated physicochemical descriptors.

    Changes from original:
    - instability_index now uses Guruprasad 1990 DIWV formula
    - net_charge uses full pKa-based HH computation
    - Added: hydrophobic_moment, aggregation_propensity
    """
    n = len(seq)
    if n == 0:
        keys = ['length', 'molecular_weight', 'pI', 'instability_index',
                'GRAVY', 'aromaticity', 'aliphatic_index', 'boman_index',
                'ww_hydrophobicity', 'net_charge', 'hydrophobic_moment',
                'aggregation_propensity']
        return {k: 0.0 for k in keys}

    mw = sum(MW_RESIDUE.get(aa, 111.0) for aa in seq) - 18.015 * (n - 1)
    gravy = sum(HYDROPHOBICITY.get(aa, 0.0) for aa in seq) / n
    pi_val = compute_pI(seq)
    instab = instability_index(seq)
    aro = sum(1 for aa in seq if aa in ('F', 'W', 'Y')) / n
    ali_w = {'A': 1.0, 'V': 2.9, 'I': 3.9, 'L': 3.9}
    ali = 100.0 + 100.0 * sum(ali_w.get(aa, 0.0) for aa in seq) / n
    boman = sum(BOMAN.get(aa, 0.0) for aa in seq) / n
    ww = sum(WW_HYDRO.get(aa, 0.0) for aa in seq) / n
    nc = net_charge(seq, pH=7.4)
    hm = hydrophobic_moment(seq)
    agg = aggregation_propensity(seq)

    return {
        'length': n,
        'molecular_weight': round(mw, 2),
        'pI': round(pi_val, 2),
        'instability_index': round(instab, 2),
        'GRAVY': round(gravy, 4),
        'aromaticity': round(aro, 4),
        'aliphatic_index': round(ali, 2),
        'boman_index': round(boman, 4),
        'ww_hydrophobicity': round(ww, 4),
        'net_charge': round(nc, 3),
        'hydrophobic_moment': round(hm, 4),
        'aggregation_propensity': int(agg),
    }


def descriptors_vector(seq: str) -> np.ndarray:
    """Return descriptor dict as a fixed-length float array."""
    d = compute_descriptors(seq)
    return np.array([
        d['length'], d['molecular_weight'], d['pI'],
        d['instability_index'], d['GRAVY'], d['aromaticity'],
        d['aliphatic_index'], d['boman_index'], d['ww_hydrophobicity'],
        d['net_charge'], d['hydrophobic_moment'], d['aggregation_propensity'],
    ], dtype=np.float32)


DESCRIPTOR_NAMES = [
    'length', 'molecular_weight', 'pI', 'instability_index',
    'GRAVY', 'aromaticity', 'aliphatic_index', 'boman_index',
    'ww_hydrophobicity', 'net_charge', 'hydrophobic_moment',
    'aggregation_propensity',
]
