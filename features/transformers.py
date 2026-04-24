"""
features/transformers.py
=========================
sklearn-compatible feature transformers for peptide sequences.

CRITICAL FIX (data leakage):
  Original code called FeatureBuilder.fit_transform(all_seqs) BEFORE
  cross-validation, meaning TF-IDF vocabulary, SVD projection, and
  StandardScaler statistics were all estimated from the full dataset —
  including held-out folds. This is a textbook leakage bug.

  Fix: wrap all feature extraction steps in a single sklearn
  BaseEstimator + TransformerMixin. When this transformer is placed
  inside a sklearn Pipeline and passed to cross_val_score, sklearn
  ensures fit() is called exclusively on training fold data for each
  split. Held-out fold data only sees transform() with the training
  statistics.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from features.physicochemical import descriptors_vector


def _kmer_string(seq: str, k: int) -> str:
    """Tokenise a sequence into overlapping k-mers separated by spaces."""
    return " ".join(seq[i:i + k] for i in range(len(seq) - k + 1))


class PhyschemTransformer(BaseEstimator, TransformerMixin):
    """
    Converts a list of sequences to a scaled physicochemical feature matrix.

    StandardScaler is fit inside this transformer, so when wrapped in a
    Pipeline the scaler statistics come only from the training fold.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        # X: list[str] of amino acid sequences
        mat = np.vstack([descriptors_vector(s) for s in X])
        self.scaler_ = StandardScaler()
        self.scaler_.fit(mat)
        return self

    def transform(self, X):
        check_is_fitted(self, 'scaler_')
        mat = np.vstack([descriptors_vector(s) for s in X])
        return self.scaler_.transform(mat).astype(np.float32)

    def fit_transform(self, X, y=None):
        mat = np.vstack([descriptors_vector(s) for s in X])
        self.scaler_ = StandardScaler()
        return self.scaler_.fit_transform(mat).astype(np.float32)


class KmerTfidfSvdTransformer(BaseEstimator, TransformerMixin):
    """
    k-mer TF-IDF → TruncatedSVD representation.

    Both the TF-IDF vocabulary and the SVD projection are fit inside
    this transformer. The n_components should be smaller than the
    vocabulary size; setting max_features limits memory.
    """

    def __init__(
        self,
        k: int = 3,
        n_components: int = 64,
        max_features: int = 5000,
    ):
        self.k = k
        self.n_components = n_components
        self.max_features = max_features

    def _to_kmer_corpus(self, X):
        return [_kmer_string(s, self.k) for s in X]

    def fit(self, X, y=None):
        corpus = self._to_kmer_corpus(X)
        self.tfidf_ = TfidfVectorizer(
            analyzer='word',
            token_pattern=r'\S+',
            max_features=self.max_features,
        )
        tfidf_mat = self.tfidf_.fit_transform(corpus)
        actual_components = min(self.n_components, tfidf_mat.shape[1] - 1)
        self.svd_ = TruncatedSVD(
            n_components=actual_components, random_state=42
        )
        self.svd_.fit(tfidf_mat)
        # Actual output dims may be < actual_components if n_samples is
        # the binding constraint (TruncatedSVD caps at min(n, p))
        self.n_components_ = self.svd_.components_.shape[0]
        return self

    def transform(self, X):
        check_is_fitted(self, ['tfidf_', 'svd_'])
        corpus = self._to_kmer_corpus(X)
        tfidf_mat = self.tfidf_.transform(corpus)
        return self.svd_.transform(tfidf_mat).astype(np.float32)

    def fit_transform(self, X, y=None):
        corpus = self._to_kmer_corpus(X)
        self.tfidf_ = TfidfVectorizer(
            analyzer='word',
            token_pattern=r'\S+',
            max_features=self.max_features,
        )
        tfidf_mat = self.tfidf_.fit_transform(corpus)
        actual_components = min(self.n_components, tfidf_mat.shape[1] - 1)
        self.svd_ = TruncatedSVD(
            n_components=actual_components, random_state=42
        )
        result = self.svd_.fit_transform(tfidf_mat)
        self.n_components_ = self.svd_.components_.shape[0]
        return result.astype(np.float32)


class CombinedSequenceTransformer(BaseEstimator, TransformerMixin):
    """
    Concatenates KmerTfidfSvd features with scaled physicochemical features.

    This is the primary feature extractor used in the sklearn Pipelines.
    Both sub-transformers fit on training data only — no leakage.
    """

    def __init__(
        self,
        k: int = 3,
        n_components: int = 64,
        max_features: int = 5000,
    ):
        self.k = k
        self.n_components = n_components
        self.max_features = max_features

    def fit(self, X, y=None):
        self.kmer_transformer_ = KmerTfidfSvdTransformer(
            k=self.k,
            n_components=self.n_components,
            max_features=self.max_features,
        )
        self.kmer_transformer_.fit(X)
        self.phys_transformer_ = PhyschemTransformer()
        self.phys_transformer_.fit(X)
        return self

    def transform(self, X):
        check_is_fitted(self, ['kmer_transformer_', 'phys_transformer_'])
        kmer_feats = self.kmer_transformer_.transform(X)
        phys_feats = self.phys_transformer_.transform(X)
        return np.hstack([kmer_feats, phys_feats])

    def fit_transform(self, X, y=None):
        self.kmer_transformer_ = KmerTfidfSvdTransformer(
            k=self.k,
            n_components=self.n_components,
            max_features=self.max_features,
        )
        kmer_feats = self.kmer_transformer_.fit_transform(X)
        self.phys_transformer_ = PhyschemTransformer()
        phys_feats = self.phys_transformer_.fit_transform(X)
        return np.hstack([kmer_feats, phys_feats])

    @property
    def n_features_out_(self) -> int:
        check_is_fitted(self, ['kmer_transformer_', 'phys_transformer_'])
        return (self.kmer_transformer_.n_components_ +
                len(self.phys_transformer_.scaler_.mean_))


# ---------------------------------------------------------------------------
# Utility: one-hot encoding (for CVAE)
# ---------------------------------------------------------------------------
from features.physicochemical import AA_LIST, AA_INDEX


def seq_to_onehot(seq: str, max_len: int) -> np.ndarray:
    """
    One-hot encode a sequence into shape (max_len, 20).

    Positions beyond len(seq) are left as all-zeros (padding token).
    The padding mask should be derived from the actual sequence length,
    not from detecting all-zero columns here.
    """
    mat = np.zeros((max_len, 20), dtype=np.float32)
    for i, aa in enumerate(seq[:max_len]):
        if aa in AA_INDEX:
            mat[i, AA_INDEX[aa]] = 1.0
    return mat


def seq_length_to_mask(length: int, max_len: int) -> np.ndarray:
    """
    Return a boolean mask of shape (max_len,):
      True  → position is a real residue
      False → position is padding
    """
    mask = np.zeros(max_len, dtype=bool)
    mask[:length] = True
    return mask
