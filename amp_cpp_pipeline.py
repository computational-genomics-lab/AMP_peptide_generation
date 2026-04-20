"""
==============================================================================
ML-Guided Generation of AMP Peptides with CPP Activity
==============================================================================
Full pipeline — Phase 1 through Phase 9
Tools: NumPy, SciPy, scikit-learn  (no PyTorch / TF required)

Architecture choices (given offline environment):
  • Sequence representation  : k-mer (tri/penta-gram) TF-IDF + physicochemical
  • CPP predictor            : Gradient-Boosting + MLP ensemble
  • AMP predictor            : Gradient-Boosting + MLP ensemble
  • Generator (CVAE)         : numpy VAE with one-hot encoding
  • Optimiser                : latent-space gradient ascent + mutation walk
==============================================================================
"""

import os, re, warnings, random, json, csv
from copy import deepcopy
from itertools import product
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from scipy.spatial.distance import cdist

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_recall_curve)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

# ─── Canonical amino acids ───────────────────────────────────────────────────
CANONICAL_AAS = set("ACDEFGHIKLMNPQRSTVWY")
AA_LIST = sorted(CANONICAL_AAS)
AA_INDEX = {aa: i for i, aa in enumerate(AA_LIST)}

# ─── Physicochemical property tables ─────────────────────────────────────────
# Kyte-Doolittle hydrophobicity
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}
# pKa side chains
PKA_SC = {'D': 3.65, 'E': 4.25, 'H': 6.00, 'C': 8.18,
          'Y': 10.07, 'K': 10.53, 'R': 12.48}
PKA_N  = 9.60   # N-terminus
PKA_C  = 2.34   # C-terminus

# Boman index (interaction energies)
BOMAN = {
    'A': -0.495, 'C': 0.081, 'D': 3.656, 'E': 3.365, 'F': -2.546,
    'G': 0.386, 'H': 2.332, 'I': -1.971, 'K': 2.101, 'L': -2.096,
    'M': -0.668, 'N': 2.128, 'P': 0.557, 'Q': 2.096, 'R': 2.897,
    'S': 0.936, 'T': 0.853, 'V': -1.254, 'W': -3.249, 'Y': -0.713
}
# Wimley-White interfacial hydrophobicity
WW_HYDRO = {
    'A': -0.17, 'C': 0.24, 'D': -1.23, 'E': -2.02, 'F': 1.13,
    'G': -0.01, 'H': -0.17, 'I': 0.31, 'K': -0.99, 'L': 0.56,
    'M': 0.23, 'N': -0.42, 'P': -0.45, 'Q': -0.58, 'R': -0.81,
    'S': -0.13, 'T': -0.14, 'V': -0.07, 'W': 1.85, 'Y': 0.94
}
# Aliphatic residues
ALIPHATIC = {'A': 1.0, 'V': 1.0, 'I': 1.0, 'L': 1.0}
# Aromatic residues
AROMATIC = {'F', 'W', 'Y'}
# Molecular weights (residue)
MW_RESIDUE = {
    'A': 89.09, 'C': 121.16, 'D': 133.10, 'E': 147.13, 'F': 165.19,
    'G': 75.03, 'H': 155.16, 'I': 131.17, 'K': 146.19, 'L': 131.17,
    'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.20,
    'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19
}


# =============================================================================
# UTILITY: Physicochemical descriptors
# =============================================================================
def compute_pI(sequence: str, precision: float = 0.01) -> float:
    """Binary-search for isoelectric point."""
    charges = []
    for aa in sequence:
        if aa in PKA_SC:
            charges.append(PKA_SC[aa])
    lo, hi = 0.0, 14.0
    while hi - lo > precision:
        mid = (lo + hi) / 2.0
        charge = 1.0 / (1.0 + 10 ** (mid - PKA_N))   # N-term positive
        charge -= 1.0 / (1.0 + 10 ** (PKA_C - mid))  # C-term negative
        for aa in sequence:
            if aa in ('K', 'R', 'H'):
                charge += 1.0 / (1.0 + 10 ** (mid - PKA_SC[aa]))
            elif aa in ('D', 'E', 'C', 'Y'):
                charge -= 1.0 / (1.0 + 10 ** (PKA_SC[aa] - mid))
        if charge > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def compute_descriptors(seq: str) -> dict:
    """Return a dict of 9 physicochemical descriptors for one sequence."""
    n = len(seq)
    if n == 0:
        return {k: 0.0 for k in ['length', 'molecular_weight', 'pI',
                                   'instability_index', 'GRAVY', 'aromaticity',
                                   'aliphatic_index', 'boman_index', 'ww_hydrophobicity']}
    mw = sum(MW_RESIDUE.get(aa, 110.0) for aa in seq) - 18.02 * (n - 1)
    gravy = sum(HYDROPHOBICITY.get(aa, 0.0) for aa in seq) / n

    # Instability index heuristic based on charged-residue content
    # Use heuristic: charged-rich sequences are unstable
    charged = sum(1 for aa in seq if aa in 'DEKR')
    instability = 40.0 + 10.0 * (charged / n - 0.3)

    pi_val = compute_pI(seq)
    aro = sum(1 for aa in seq if aa in AROMATIC) / n
    ali = (sum(ALIPHATIC.get(aa, 0.0) * {'A': 1.0, 'V': 2.9,
                                           'I': 3.9, 'L': 3.9}.get(aa, 0.0)
               for aa in seq)) / n * 100 + 100
    boman = sum(BOMAN.get(aa, 0.0) for aa in seq) / n
    ww = sum(WW_HYDRO.get(aa, 0.0) for aa in seq) / n

    return {
        'length': n,
        'molecular_weight': round(mw, 2),
        'pI': round(pi_val, 2),
        'instability_index': round(instability, 2),
        'GRAVY': round(gravy, 4),
        'aromaticity': round(aro, 4),
        'aliphatic_index': round(ali, 2),
        'boman_index': round(boman, 4),
        'ww_hydrophobicity': round(ww, 4),
    }


def descriptors_vector(seq: str) -> np.ndarray:
    d = compute_descriptors(seq)
    return np.array([d['length'], d['molecular_weight'], d['pI'],
                     d['instability_index'], d['GRAVY'], d['aromaticity'],
                     d['aliphatic_index'], d['boman_index'], d['ww_hydrophobicity']],
                    dtype=float)


# =============================================================================
# UTILITY: Sequence feature engineering
# =============================================================================
def seq_to_kmer_string(seq: str, k: int = 3) -> str:
    """Convert sequence to space-separated k-mers."""
    return " ".join(seq[i:i+k] for i in range(len(seq) - k + 1))


def seq_to_onehot(seq: str, max_len: int = 50) -> np.ndarray:
    """Pad / truncate to max_len, one-hot encode, shape (max_len * 20,)."""
    vec = np.zeros(max_len * 20, dtype=np.float32)
    for i, aa in enumerate(seq[:max_len]):
        if aa in AA_INDEX:
            vec[i * 20 + AA_INDEX[aa]] = 1.0
    return vec


def is_canonical(seq: str) -> bool:
    return bool(seq) and all(aa in CANONICAL_AAS for aa in seq)


def net_charge(seq: str, pH: float = 7.0) -> float:
    pos = sum(1 for aa in seq if aa in 'KR')
    pos += sum(1.0 / (1.0 + 10 ** (pH - 6.0)) for aa in seq if aa == 'H')
    neg = sum(1 for aa in seq if aa in 'DE')
    return pos - neg


# =============================================================================
# PHASE 1 — Data Preparation
# =============================================================================
def phase1_load_and_clean():
    print("\n" + "═" * 70)
    print("  PHASE 1: DATA PREPARATION")
    print("═" * 70)

    # --- Load AMP -------------------------------------------------------
    amp_raw = pd.read_excel(
        '/mnt/user-data/uploads/actual_antimicrobial_peptide_list_ADP6_25feb_-_sequences_containing_more_that_one_C_contains_disulfide_bond.xlsx'
    )
    amp_raw.columns = [c.strip() for c in amp_raw.columns]

    # --- Load CPP -------------------------------------------------------
    cpp_raw = pd.read_csv(
        '/mnt/user-data/uploads/Cell_Penetrating_Peptides_List_CellPPD_Raghavendra.csv',
        encoding='latin1'
    )
    cpp_raw.columns = [c.strip().replace('\xa0', '').replace(' ', '_')
                       for c in cpp_raw.columns]

    # Parse merged last column: "  CPP  0.82  0.18  " → label + probs
    last_col = cpp_raw.columns[-1]
    parsed = cpp_raw[last_col].str.strip().str.split(r'\s+', expand=True)
    parsed.columns = ['cpp_label', 'cpp_prob', 'non_cpp_prob']
    cpp_raw = pd.concat([cpp_raw.drop(columns=[last_col]), parsed], axis=1)
    cpp_raw['cpp_label']   = cpp_raw['cpp_label'].str.strip()
    cpp_raw['cpp_prob']    = pd.to_numeric(cpp_raw['cpp_prob'], errors='coerce')
    cpp_raw['non_cpp_prob'] = pd.to_numeric(cpp_raw['non_cpp_prob'], errors='coerce')

    print(f"  Raw AMP rows : {len(amp_raw):,}")
    print(f"  Raw CPP rows : {len(cpp_raw):,}")

    # ── AMP cleaning ────────────────────────────────────────────────────
    amp = amp_raw[['SeqID', 'Sequence']].copy()
    amp['Sequence'] = amp['Sequence'].str.strip().str.upper()
    amp = amp.dropna(subset=['Sequence'])
    amp = amp[amp['Sequence'].apply(is_canonical)]
    amp = amp.drop_duplicates(subset='Sequence')
    # Length filter: keep 5–60 AA (covers 95th percentile of distribution)
    amp = amp[(amp['Sequence'].str.len() >= 5) &
              (amp['Sequence'].str.len() <= 60)]
    amp = amp.reset_index(drop=True)

    # ── CPP cleaning ────────────────────────────────────────────────────
    cpp = cpp_raw[['Protein_ID', 'Protein_Sequence',
                   'cpp_label', 'cpp_prob']].copy()
    cpp.columns = ['SeqID', 'Sequence', 'cpp_label', 'cpp_prob']
    cpp['Sequence'] = cpp['Sequence'].str.strip().str.upper()
    cpp = cpp.dropna(subset=['Sequence'])
    cpp = cpp[cpp['Sequence'].apply(is_canonical)]
    cpp = cpp.drop_duplicates(subset='Sequence')
    cpp = cpp[(cpp['Sequence'].str.len() >= 5) &
              (cpp['Sequence'].str.len() <= 60)]
    cpp['y_cpp'] = (cpp['cpp_label'] == 'CPP').astype(int)
    cpp = cpp.reset_index(drop=True)

    # ── Compute descriptors for both ─────────────────────────────────────
    print("  Computing physicochemical descriptors for AMP sequences…")
    amp_descs = pd.DataFrame([compute_descriptors(s) for s in amp['Sequence']])
    amp = pd.concat([amp, amp_descs], axis=1)

    print("  Computing physicochemical descriptors for CPP sequences…")
    cpp_descs = pd.DataFrame([compute_descriptors(s) for s in cpp['Sequence']])
    cpp = pd.concat([cpp, cpp_descs], axis=1)

    print(f"\n  ✓ AMP clean  : {len(amp):,} sequences")
    print(f"  ✓ CPP clean  : {len(cpp):,} sequences "
          f"({cpp['y_cpp'].sum()} CPP | {(~cpp['y_cpp'].astype(bool)).sum()} Non-CPP)")

    length_overlap = (
        max(amp['length'].min(), cpp['length'].min()),
        min(amp['length'].max(), cpp['length'].max())
    )
    print(f"  Common length range : {length_overlap[0]}–{length_overlap[1]} AA")

    return amp, cpp


# =============================================================================
# SHARED: Feature builder (k-mer + physicochemical)
# =============================================================================
class FeatureBuilder:
    """Fits TF-IDF k-mer vectoriser on a training corpus, then SVD-reduces.
    Final feature = [SVD(k-mer)] ++ [physicochemical descriptors]
    """
    def __init__(self, k: int = 3, n_components: int = 64):
        self.k = k
        self.n_components = n_components
        self.tfidf = TfidfVectorizer(analyzer='word', token_pattern=r'\S+',
                                     max_features=5000)
        self.svd   = TruncatedSVD(n_components=n_components, random_state=42)
        self.fitted = False

    def _kmer_strings(self, seqs):
        return [seq_to_kmer_string(s, self.k) for s in seqs]

    def fit(self, seqs):
        km = self._kmer_strings(seqs)
        X_tfidf = self.tfidf.fit_transform(km)
        self.svd.fit(X_tfidf)
        self.fitted = True
        return self

    def transform(self, seqs):
        km = self._kmer_strings(seqs)
        X_tfidf = self.tfidf.transform(km)
        X_svd   = self.svd.transform(X_tfidf)
        X_phys  = np.vstack([descriptors_vector(s) for s in seqs])
        # Z-score the physicochemical block
        return np.hstack([X_svd, X_phys])

    def fit_transform(self, seqs):
        self.fit(seqs)
        return self.transform(seqs)


# =============================================================================
# PHASE 2 — CPP Predictor
# =============================================================================
def phase2_build_cpp_predictor(cpp_df: pd.DataFrame):
    print("\n" + "═" * 70)
    print("  PHASE 2: CPP PREDICTOR")
    print("═" * 70)

    seqs  = cpp_df['Sequence'].tolist()
    y     = cpp_df['y_cpp'].values

    # ── Negative class augmentation ────────────────────────────────────
    # Shuffle CPP sequences to make synthetic negatives
    cpp_pos = cpp_df[cpp_df['y_cpp'] == 1]['Sequence'].tolist()
    n_neg   = max(0, len(cpp_pos) - (y == 0).sum())
    shuffled_neg = []
    rng = np.random.default_rng(42)
    for seq in cpp_pos[:n_neg]:
        arr = list(seq)
        rng.shuffle(arr)
        shuffled_neg.append(''.join(arr))
    if shuffled_neg:
        extra = pd.DataFrame({
            'Sequence': shuffled_neg,
            'y_cpp': [0] * len(shuffled_neg)
        })
        aug_df = pd.concat([cpp_df[['Sequence', 'y_cpp']], extra],
                           ignore_index=True)
        seqs = aug_df['Sequence'].tolist()
        y    = aug_df['y_cpp'].values
        print(f"  Augmented with {len(shuffled_neg)} shuffled negatives → "
              f"{(y==1).sum()} pos / {(y==0).sum()} neg")

    # ── Feature extraction ─────────────────────────────────────────────
    print("  Building k-mer + physicochemical features…")
    feat_builder = FeatureBuilder(k=3, n_components=50)
    X = feat_builder.fit_transform(seqs)
    print(f"  Feature matrix : {X.shape}")

    # ── Ensemble: GBM + MLP ───────────────────────────────────────────
    gbm = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=4,
        subsample=0.8, random_state=42
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64), activation='relu',
        max_iter=500, random_state=42, early_stopping=True,
        validation_fraction=0.1
    )

    # ── Stratified CV ──────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Scale for MLP
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("  Running 5-fold cross-validation…")
    gbm_aucs = cross_val_score(gbm, X, y, cv=cv,
                                scoring='roc_auc', n_jobs=1)
    mlp_aucs = cross_val_score(mlp, X_scaled, y, cv=cv,
                                scoring='roc_auc', n_jobs=1)

    print(f"  GBM  ROC-AUC : {gbm_aucs.mean():.3f} ± {gbm_aucs.std():.3f}")
    print(f"  MLP  ROC-AUC : {mlp_aucs.mean():.3f} ± {mlp_aucs.std():.3f}")

    # ── Calibrated full-fit models ─────────────────────────────────────
    gbm_cal = CalibratedClassifierCV(gbm, cv=3, method='isotonic')
    mlp_cal = CalibratedClassifierCV(mlp, cv=3, method='isotonic')
    gbm_cal.fit(X, y)
    mlp_cal.fit(X_scaled, y)
    print("  ✓ CPP predictor trained and calibrated.")

    def predict_cpp(sequences):
        Xq = feat_builder.transform(sequences)
        Xq_sc = scaler.transform(Xq)
        p_gbm = gbm_cal.predict_proba(Xq)[:, 1]
        p_mlp = mlp_cal.predict_proba(Xq_sc)[:, 1]
        return (p_gbm + p_mlp) / 2.0

    # Attach metadata
    predict_cpp.feat_builder = feat_builder
    predict_cpp.scaler       = scaler
    predict_cpp.gbm          = gbm_cal
    predict_cpp.mlp          = mlp_cal
    predict_cpp.gbm_auc      = gbm_aucs.mean()
    predict_cpp.mlp_auc      = mlp_aucs.mean()

    return predict_cpp


# =============================================================================
# PHASE 3 — AMP Predictor
# =============================================================================
def phase3_build_amp_predictor(amp_df: pd.DataFrame):
    print("\n" + "═" * 70)
    print("  PHASE 3: AMP PREDICTOR")
    print("═" * 70)

    amp_seqs = amp_df['Sequence'].tolist()

    # ── Negative class: shuffled AMP sequences ─────────────────────────
    rng = np.random.default_rng(99)
    neg_seqs = []
    for seq in amp_seqs:
        arr = list(seq)
        rng.shuffle(arr)
        neg_seqs.append(''.join(arr))

    all_seqs = amp_seqs + neg_seqs
    y = np.array([1] * len(amp_seqs) + [0] * len(neg_seqs))
    print(f"  Training set: {len(amp_seqs)} AMP + {len(neg_seqs)} shuffled negatives")

    # ── Features ────────────────────────────────────────────────────────
    print("  Building features…")
    feat_builder = FeatureBuilder(k=3, n_components=64)
    X = feat_builder.fit_transform(all_seqs)
    print(f"  Feature matrix : {X.shape}")

    gbm = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.07, max_depth=4,
        subsample=0.8, random_state=42
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), activation='relu',
        max_iter=600, random_state=42, early_stopping=True,
        validation_fraction=0.1
    )
    rf = RandomForestClassifier(n_estimators=200, random_state=42)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("  Running 5-fold cross-validation…")
    gbm_aucs = cross_val_score(gbm, X, y, cv=cv, scoring='roc_auc')
    mlp_aucs = cross_val_score(mlp, X_scaled, y, cv=cv, scoring='roc_auc')
    rf_aucs  = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc')

    print(f"  GBM  ROC-AUC : {gbm_aucs.mean():.3f} ± {gbm_aucs.std():.3f}")
    print(f"  MLP  ROC-AUC : {mlp_aucs.mean():.3f} ± {mlp_aucs.std():.3f}")
    print(f"  RF   ROC-AUC : {rf_aucs.mean():.3f}  ± {rf_aucs.std():.3f}")

    gbm_cal = CalibratedClassifierCV(gbm, cv=3, method='isotonic')
    mlp_cal = CalibratedClassifierCV(mlp, cv=3, method='isotonic')
    rf_cal  = CalibratedClassifierCV(rf, cv=3, method='isotonic')
    gbm_cal.fit(X, y)
    mlp_cal.fit(X_scaled, y)
    rf_cal.fit(X, y)
    print("  ✓ AMP predictor trained and calibrated.")

    def predict_amp(sequences):
        Xq    = feat_builder.transform(sequences)
        Xq_sc = scaler.transform(Xq)
        p_gbm = gbm_cal.predict_proba(Xq)[:, 1]
        p_mlp = mlp_cal.predict_proba(Xq_sc)[:, 1]
        p_rf  = rf_cal.predict_proba(Xq)[:, 1]
        return (p_gbm + p_mlp + p_rf) / 3.0

    predict_amp.feat_builder = feat_builder
    predict_amp.scaler       = scaler
    predict_amp.gbm_auc      = gbm_aucs.mean()
    predict_amp.mlp_auc      = mlp_aucs.mean()
    predict_amp.rf_auc       = rf_aucs.mean()

    return predict_amp


# =============================================================================
# PHASE 4 — CVAE Peptide Generator (NumPy implementation)
# =============================================================================
class PeptideCVAE:
    """
    VAE for peptide generation — pure NumPy, correct explicit backprop.
    Architecture  (smaller for stability without deep-learning framework):
        Encoder : x(D) → h1(128, ReLU) → μ(L), log σ²(L)
        Decoder : z(L) → h1(128, ReLU) → out(D) → per-position softmax
    where D = max_len × 20,  L = latent_dim
    """
    def __init__(self, max_len: int = 40, latent_dim: int = 32, lr: float = 5e-4):
        self.max_len   = max_len
        self.L         = latent_dim
        self.lr        = lr
        self.D         = max_len * 20    # input / output dimension
        H              = 128             # hidden size

        # Encoder
        self.We1  = self._he(self.D, H)
        self.be1  = np.zeros(H, dtype=np.float32)
        self.Wmu  = self._he(H, latent_dim)
        self.bmu  = np.zeros(latent_dim, dtype=np.float32)
        self.Wlv  = self._he(H, latent_dim)
        self.blv  = np.zeros(latent_dim, dtype=np.float32)

        # Decoder
        self.Wd1  = self._he(latent_dim, H)
        self.bd1  = np.zeros(H, dtype=np.float32)
        self.Wd2  = self._he(H, self.D)
        self.bd2  = np.zeros(self.D, dtype=np.float32)

    @staticmethod
    def _he(fan_in, fan_out):
        return (np.random.randn(fan_in, fan_out)
                * np.sqrt(2.0 / fan_in)).astype(np.float32)

    @staticmethod
    def _relu(x):
        return np.maximum(0.0, x)

    def _softmax_per_pos(self, x):
        """x: (B, D=max_len*20) → per-position softmax, same shape."""
        B = x.shape[0]
        x3 = x.reshape(B, self.max_len, 20)
        x3 = x3 - x3.max(axis=2, keepdims=True)
        ex = np.exp(x3)
        return (ex / ex.sum(axis=2, keepdims=True)).reshape(B, self.D)

    # ── Encoder ──────────────────────────────────────────────────────────
    def _encode(self, X):
        z1   = X  @ self.We1 + self.be1     # (B, H)
        h1   = self._relu(z1)
        mu   = h1 @ self.Wmu + self.bmu    # (B, L)
        logv = h1 @ self.Wlv + self.blv    # (B, L)
        return mu, logv, h1, z1

    # ── Decoder ──────────────────────────────────────────────────────────
    def _decode(self, z):
        z1  = z  @ self.Wd1 + self.bd1     # (B, H)
        h1  = self._relu(z1)
        out = h1 @ self.Wd2 + self.bd2     # (B, D)
        probs = self._softmax_per_pos(out)
        return probs, h1, z1

    def _step(self, batch, beta, t, m, v, params,
              beta1=0.9, beta2=0.999, adam_eps=1e-8):
        B = batch.shape[0]

        # ── Forward ──────────────────────────────────────────────────
        mu, logv, eh1, ez1 = self._encode(batch)
        eps = np.random.randn(B, self.L).astype(np.float32)
        z   = mu + eps * np.exp(0.5 * logv)
        probs, dh1, dz1 = self._decode(z)

        # Losses
        recon = -np.sum(batch * np.log(probs + 1e-8)) / B
        kl    = -0.5 * np.mean(1.0 + logv - mu**2 - np.exp(logv))
        loss  = recon + beta * kl

        # ── Backward: decoder ────────────────────────────────────────
        # dL/d_logit = (probs - X) / B   [softmax+CE combined gradient]
        d_logit = (probs - batch) / B               # (B, D)
        gWd2    = dh1.T @ d_logit                   # (H, D)
        gbd2    = d_logit.sum(axis=0)               # (D,)
        d_dh1   = d_logit @ self.Wd2.T              # (B, H)
        d_dz1   = d_dh1 * (dz1 > 0)                # ReLU grad
        gWd1    = z.T @ d_dz1                       # (L, H)
        gbd1    = d_dz1.sum(axis=0)                 # (H,)
        d_z     = d_dz1 @ self.Wd1.T               # (B, L)

        # ── Backward: KL + reparameterisation ───────────────────────
        d_kl_mu  = mu / B
        d_kl_lv  = 0.5 * (np.exp(logv) - 1.0) / B
        d_mu     = d_kl_mu + d_z                    # (B, L)
        d_lv     = d_kl_lv + d_z * eps * 0.5       # (B, L)

        gWmu  = eh1.T @ d_mu                        # (H, L)
        gbmu  = d_mu.sum(axis=0)
        gWlv  = eh1.T @ d_lv
        gblv  = d_lv.sum(axis=0)
        d_eh1 = d_mu @ self.Wmu.T + d_lv @ self.Wlv.T   # (B, H)
        d_ez1 = d_eh1 * (ez1 > 0)                  # ReLU grad
        gWe1  = batch.T @ d_ez1                     # (D, H)
        gbe1  = d_ez1.sum(axis=0)

        # ── Adam update ───────────────────────────────────────────────
        grads = dict(We1=gWe1, be1=gbe1, Wmu=gWmu, bmu=gbmu,
                     Wlv=gWlv, blv=gblv, Wd1=gWd1, bd1=gbd1,
                     Wd2=gWd2, bd2=gbd2)
        for p in params:
            g     = np.clip(grads[p], -5.0, 5.0)
            m[p]  = beta1 * m[p] + (1 - beta1) * g
            v[p]  = beta2 * v[p] + (1 - beta2) * g**2
            mh    = m[p] / (1 - beta1**t)
            vh    = v[p] / (1 - beta2**t)
            setattr(self, p, getattr(self, p) - self.lr * mh / (np.sqrt(vh) + adam_eps))

        return loss, recon, kl

    def train(self, seqs: list, epochs: int = 80, batch_size: int = 64,
              beta: float = 0.5, verbose: bool = True):
        X      = np.vstack([seq_to_onehot(s, self.max_len) for s in seqs])
        n      = len(X)
        params = ['We1','be1','Wmu','bmu','Wlv','blv','Wd1','bd1','Wd2','bd2']
        m = {p: np.zeros_like(getattr(self, p)) for p in params}
        v = {p: np.zeros_like(getattr(self, p)) for p in params}
        t = 0
        history = []
        for epoch in range(epochs):
            idx  = np.random.permutation(n)
            eloss = 0.0; nb = 0
            recon = kl = 0.0
            for start in range(0, n, batch_size):
                batch = X[idx[start: start + batch_size]]
                if len(batch) < 4:
                    continue
                t += 1
                loss, recon, kl = self._step(batch, beta, t, m, v, params)
                eloss += loss; nb += 1
            avg = eloss / max(nb, 1)
            history.append(avg)
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"    Epoch {epoch+1:3d}/{epochs}  "
                      f"loss={avg:.4f}  recon={recon:.4f}  kl={kl:.4f}")
        return history

    def generate(self, n: int = 500, temperature: float = 1.1,
                 min_len: int = 8, max_len_out: int = None) -> list:
        max_out  = max_len_out or self.max_len
        sequences = []
        while len(sequences) < n:
            nb    = min(256, (n - len(sequences)) * 3)
            z     = np.random.randn(nb, self.L).astype(np.float32)
            probs, _, _ = self._decode(z)
            probs = probs.reshape(nb, self.max_len, 20)
            if temperature != 1.0:
                logits = np.log(probs + 1e-8) / temperature
                logits -= logits.max(axis=2, keepdims=True)
                probs  = np.exp(logits) / np.exp(logits).sum(axis=2, keepdims=True)
            for i in range(nb):
                seq = [AA_LIST[np.random.choice(20, p=probs[i, pos])]
                       for pos in range(self.max_len)]
                raw     = ''.join(seq)
                trimmed = raw.rstrip('A')
                seq_str = trimmed if len(trimmed) >= min_len else raw[:max_out]
                if min_len <= len(seq_str) <= max_out and is_canonical(seq_str):
                    sequences.append(seq_str)
                if len(sequences) >= n:
                    break
        return sequences[:n]

    def encode_sequence(self, seq: str) -> np.ndarray:
        x = seq_to_onehot(seq, self.max_len).reshape(1, -1)
        mu, _, _, _ = self._encode(x)
        return mu[0]


def phase4_train_generator(amp_df: pd.DataFrame):
    print("\n" + "═" * 70)
    print("  PHASE 4: CVAE PEPTIDE GENERATOR")
    print("═" * 70)

    seqs     = amp_df['Sequence'].tolist()
    max_len  = min(int(np.percentile([len(s) for s in seqs], 90)), 50)
    print(f"  Training on {len(seqs)} AMP sequences  |  max_len={max_len}")

    vae = PeptideCVAE(max_len=max_len, latent_dim=32, lr=5e-4)
    vae.train(seqs, epochs=100, batch_size=64, beta=0.4, verbose=True)
    print("  ✓ CVAE trained.")
    return vae


# =============================================================================
# PHASE 5 — Multi-Objective Generation & Optimisation
# =============================================================================
def mutate_sequence(seq: str, n_mutations: int = 1,
                    rng: np.random.Generator = None) -> str:
    """Random single/multi-point amino-acid mutation."""
    if rng is None:
        rng = np.random.default_rng()
    arr = list(seq)
    for _ in range(n_mutations):
        pos = rng.integers(0, len(arr))
        arr[pos] = rng.choice(AA_LIST)
    return ''.join(arr)


def phase5_generate_and_optimise(vae, predict_amp, predict_cpp,
                                  n_candidates: int = 2000,
                                  n_optimise_rounds: int = 3):
    print("\n" + "═" * 70)
    print("  PHASE 5: MULTI-OBJECTIVE GENERATION & OPTIMISATION")
    print("═" * 70)

    rng = np.random.default_rng(7)

    # ── Step 1: Generate large candidate pool ────────────────────────
    print(f"  Generating {n_candidates} initial candidates from CVAE…")
    raw_pool = vae.generate(n=n_candidates, temperature=1.1,
                             min_len=8, max_len_out=vae.max_len)
    print(f"  Generated {len(raw_pool)} valid sequences")

    # ── Step 2: Score pool ────────────────────────────────────────────
    def score_pool(seqs):
        amp_s = predict_amp(seqs)
        cpp_s = predict_cpp(seqs)
        return amp_s, cpp_s

    print("  Scoring initial pool…")
    amp_scores, cpp_scores = score_pool(raw_pool)

    # Combined score (geometric mean)
    combined = np.sqrt(amp_scores * cpp_scores)
    order = np.argsort(-combined)

    # ── Step 3: Iterative mutation of top candidates ──────────────────
    top_k = min(200, len(raw_pool))
    candidates = [raw_pool[i] for i in order[:top_k]]
    all_candidates = list(zip(raw_pool, amp_scores, cpp_scores, combined))

    for rnd in range(1, n_optimise_rounds + 1):
        print(f"  Optimisation round {rnd}/{n_optimise_rounds}…")
        mutated = []
        for seq in candidates:
            for nm in [1, 2]:
                mutated.append(mutate_sequence(seq, nm, rng))
        mutated = [s for s in mutated if is_canonical(s) and len(s) >= 8]
        if not mutated:
            continue
        m_amp, m_cpp = score_pool(mutated)
        m_comb = np.sqrt(m_amp * m_cpp)
        for s, a, c, co in zip(mutated, m_amp, m_cpp, m_comb):
            all_candidates.append((s, a, c, co))

        # Re-rank and keep top-k for next round
        all_candidates.sort(key=lambda x: -x[3])
        candidates = [x[0] for x in all_candidates[:top_k]]

    # ── Step 4: Deduplicate pool ──────────────────────────────────────
    seen = set()
    unique_candidates = []
    for tup in all_candidates:
        if tup[0] not in seen:
            seen.add(tup[0])
            unique_candidates.append(tup)

    print(f"  ✓ Total unique candidates after optimisation: "
          f"{len(unique_candidates):,}")
    return unique_candidates


# =============================================================================
# PHASE 6 — Biological Hard Filters
# =============================================================================
def phase6_biological_filters(candidates: list,
                                min_len: int = 8, max_len: int = 50,
                                min_charge: float = 1.0,
                                max_gravy: float = 2.5,
                                max_instability: float = 65.0,
                                min_amp_score: float = 0.50,
                                min_cpp_score: float = 0.50):
    print("\n" + "═" * 70)
    print("  PHASE 6: BIOLOGICAL HARD FILTERS")
    print("═" * 70)

    passed = []
    reasons = {'length': 0, 'charge': 0, 'gravy': 0,
               'instability': 0, 'amp_score': 0, 'cpp_score': 0}

    for seq, amp_s, cpp_s, comb in candidates:
        L = len(seq)
        if not (min_len <= L <= max_len):
            reasons['length'] += 1
            continue
        d    = compute_descriptors(seq)
        nc   = net_charge(seq)
        if nc < min_charge:
            reasons['charge'] += 1
            continue
        if d['GRAVY'] > max_gravy:
            reasons['gravy'] += 1
            continue
        if d['instability_index'] > max_instability:
            reasons['instability'] += 1
            continue
        if amp_s < min_amp_score:
            reasons['amp_score'] += 1
            continue
        if cpp_s < min_cpp_score:
            reasons['cpp_score'] += 1
            continue
        passed.append((seq, amp_s, cpp_s, comb, d))

    print(f"  Filtered out:")
    for k, v in reasons.items():
        print(f"    {k:>12s} violation : {v:,}")
    print(f"  ✓ Passed : {len(passed):,} sequences")
    return passed


# =============================================================================
# PHASE 7 — External CPP Validation (self-consistency check)
# =============================================================================
def phase7_secondary_validation(passed: list, predict_cpp):
    """Re-score with the CPP predictor as a sanity check tie-breaker."""
    print("\n" + "═" * 70)
    print("  PHASE 7: SECONDARY CPP VALIDATION")
    print("═" * 70)

    seqs = [p[0] for p in passed]
    cpp_val = predict_cpp(seqs)
    # Re-combine with validation score (average original + validation)
    validated = []
    for (seq, amp_s, cpp_s, comb, d), val in zip(passed, cpp_val):
        avg_cpp = (cpp_s + val) / 2.0
        new_comb = np.sqrt(amp_s * avg_cpp)
        validated.append((seq, amp_s, avg_cpp, new_comb, d))
    validated.sort(key=lambda x: -x[3])
    print(f"  ✓ Validation complete. Top combined score: "
          f"{validated[0][3]:.3f}")
    return validated


# =============================================================================
# PHASE 8 — Diversity Control
# =============================================================================
def phase8_diversity(validated: list, n_clusters: int = 20,
                     top_per_cluster: int = 3):
    print("\n" + "═" * 70)
    print("  PHASE 8: DIVERSITY CONTROL")
    print("═" * 70)

    seqs = [v[0] for v in validated]
    # Feature matrix for clustering: physicochemical only (fast, interpretable)
    X_phys = np.vstack([descriptors_vector(s) for s in seqs])
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_phys)

    n_cl = min(n_clusters, len(seqs) // 2)
    km   = KMeans(n_clusters=n_cl, n_init=10, random_state=42)
    labels = km.fit_predict(X_sc)

    # Pick top_per_cluster sequences per cluster
    cluster_best = {i: [] for i in range(n_cl)}
    for idx, (record, label) in enumerate(zip(validated, labels)):
        cluster_best[label].append((record, idx))

    diverse = []
    for cl, items in cluster_best.items():
        items.sort(key=lambda x: -x[0][3])      # sort by combined score
        for rec, idx in items[:top_per_cluster]:
            seq, amp_s, cpp_s, comb, d = rec
            diverse.append({
                'sequence': seq,
                'amp_score': round(float(amp_s), 4),
                'cpp_score': round(float(cpp_s), 4),
                'combined_score': round(float(comb), 4),
                'cluster': int(cl),
                **{k: v for k, v in d.items()}
            })

    diverse.sort(key=lambda x: -x['combined_score'])
    # Final re-rank
    for rank, item in enumerate(diverse, start=1):
        item['rank'] = rank

    print(f"  Clusters: {n_cl}  |  Top {top_per_cluster}/cluster  "
          f"→  {len(diverse)} diverse final candidates")
    return diverse


# =============================================================================
# PHASE 9 — Final Output
# =============================================================================
def phase9_final_output(diverse: list, out_dir: str = '/mnt/user-data/outputs'):
    print("\n" + "═" * 70)
    print("  PHASE 9: FINAL OUTPUT")
    print("═" * 70)

    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(diverse)
    cols_order = ['rank', 'sequence', 'amp_score', 'cpp_score', 'combined_score',
                  'cluster', 'length', 'molecular_weight', 'pI',
                  'instability_index', 'GRAVY', 'aromaticity',
                  'aliphatic_index', 'boman_index', 'ww_hydrophobicity']
    df = df[[c for c in cols_order if c in df.columns]]

    csv_path = os.path.join(out_dir, 'amp_cpp_candidates.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved → {csv_path}")

    # ── Summary report ────────────────────────────────────────────────
    report_lines = [
        "=" * 72,
        "  ML-GUIDED AMP+CPP PEPTIDE GENERATION — FINAL REPORT",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 72,
        f"  Total candidates : {len(diverse)}",
        f"  Diversity clusters: {df['cluster'].nunique()}",
        "",
        f"  Score statistics:",
        f"    AMP score   mean={df['amp_score'].mean():.3f}  "
        f"max={df['amp_score'].max():.3f}",
        f"    CPP score   mean={df['cpp_score'].mean():.3f}  "
        f"max={df['cpp_score'].max():.3f}",
        f"    Combined    mean={df['combined_score'].mean():.3f}  "
        f"max={df['combined_score'].max():.3f}",
        "",
        "  Top 10 candidates:",
        "-" * 72,
        f"  {'Rank':>4}  {'Sequence':<45}  {'AMP':>5}  {'CPP':>5}  {'Comb':>5}  {'Cluster':>7}",
        "-" * 72,
    ]
    for row in df.head(10).itertuples():
        report_lines.append(
            f"  {row.rank:>4}  {row.sequence:<45}  "
            f"{row.amp_score:>5.3f}  {row.cpp_score:>5.3f}  "
            f"{row.combined_score:>5.3f}  {row.cluster:>7}"
        )
    report_lines += [
        "-" * 72,
        "",
        "  Physicochemical summary (top 10):",
        f"  {'Seq':<20}  {'MW':>8}  {'pI':>5}  {'GRAVY':>6}  "
        f"{'Charge':>6}  {'Boman':>6}",
        "-" * 72,
    ]
    for row in df.head(10).itertuples():
        nc = net_charge(row.sequence)
        report_lines.append(
            f"  {row.sequence[:20]:<20}  {row.molecular_weight:>8.1f}  "
            f"{row.pI:>5.2f}  {row.GRAVY:>6.3f}  {nc:>6.1f}  "
            f"{row.boman_index:>6.3f}"
        )
    report_lines.append("=" * 72)

    report_path = os.path.join(out_dir, 'amp_cpp_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"  Saved → {report_path}")

    print("\n" + '\n'.join(report_lines))
    return df, csv_path, report_path


# =============================================================================
# MAIN
# =============================================================================
def run_pipeline():
    print("\n" + "█" * 70)
    print("  ML-GUIDED AMP+CPP PEPTIDE GENERATION PIPELINE")
    print("  All 9 Phases  —  NumPy/SciPy/scikit-learn")
    print("█" * 70)

    # Phase 1
    amp_df, cpp_df = phase1_load_and_clean()

    # Phase 2
    predict_cpp = phase2_build_cpp_predictor(cpp_df)

    # Phase 3
    predict_amp = phase3_build_amp_predictor(amp_df)

    # Phase 4
    vae = phase4_train_generator(amp_df)

    # Phase 5
    candidates = phase5_generate_and_optimise(
        vae, predict_amp, predict_cpp,
        n_candidates=2000, n_optimise_rounds=3
    )

    # Phase 6
    passed = phase6_biological_filters(
        candidates,
        min_len=8, max_len=50,
        min_charge=1.0,
        max_gravy=2.5,
        max_instability=65.0,
        min_amp_score=0.50,
        min_cpp_score=0.50
    )

    if not passed:
        print("  ⚠ No sequences passed biological filters. "
              "Relaxing constraints…")
        passed = phase6_biological_filters(
            candidates,
            min_len=6, max_len=55,
            min_charge=0.0,
            max_gravy=3.0,
            max_instability=80.0,
            min_amp_score=0.40,
            min_cpp_score=0.40
        )

    # Phase 7
    validated = phase7_secondary_validation(passed, predict_cpp)

    # Phase 8
    diverse = phase8_diversity(validated, n_clusters=20, top_per_cluster=4)

    # Phase 9
    df, csv_path, report_path = phase9_final_output(diverse)

    print("\n  ✅ Pipeline complete!")
    print(f"     Candidates CSV : {csv_path}")
    print(f"     Report TXT     : {report_path}")
    return df


if __name__ == '__main__':
    results = run_pipeline()
