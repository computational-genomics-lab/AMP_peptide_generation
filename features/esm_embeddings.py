"""
features/esm_embeddings.py
===========================
Optional ESM-2 protein language model embeddings.

When use_esm=true in config.yaml, this replaces the k-mer TF-IDF + SVD
representation with per-sequence mean-pooled embeddings from ESM-2.

Why ESM-2 is preferred over k-mer TF-IDF + SVD:
  - ESM-2 is pre-trained on 250M protein sequences and captures
    evolutionary, structural, and functional context that k-mer
    frequency statistics cannot represent.
  - Embeddings are contextual: the representation of 'K' in 'KKKK'
    differs from 'K' in 'RLLK', capturing position-dependent meaning.
  - Dimensionality reduction via SVD discards variance; ESM-2 embeddings
    are already compressed into a dense, semantically rich space.

Setup (CPU, no GPU required):
  pip install fair-esm torch
  # Weights (~700MB for esm2_t6_8M_UR50D) download automatically on first run.
  # For best accuracy: esm2_t33_650M_UR50D (~2.5GB)

IMPORTANT: fit() of this transformer does NOTHING (ESM-2 weights are fixed).
The transformer is still wrapped as a sklearn BaseEstimator to allow it to
be used inside a sklearn Pipeline. No data leakage concern because there
are no learned statistics to estimate from training data.
"""

import logging
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

try:
    import torch
    import esm
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False


class ESM2Transformer(BaseEstimator, TransformerMixin):
    """
    Converts a list of amino acid sequences to ESM-2 mean-pooled embeddings.

    Parameters
    ----------
    model_name : str
        ESM-2 model identifier, e.g. 'esm2_t6_8M_UR50D' (smallest/fastest)
        or 'esm2_t33_650M_UR50D' (best quality).
    repr_layer : int
        Transformer layer from which to extract representations.
        Defaults to the last layer of the selected model.
    batch_size : int
        Number of sequences per forward pass. Reduce if OOM on CPU.
    """

    # Map model name → number of transformer layers
    _MODEL_LAYERS = {
        'esm2_t6_8M_UR50D':    6,
        'esm2_t12_35M_UR50D':  12,
        'esm2_t30_150M_UR50D': 30,
        'esm2_t33_650M_UR50D': 33,
        'esm2_t36_3B_UR50D':   36,
    }

    def __init__(
        self,
        model_name: str = 'esm2_t6_8M_UR50D',
        repr_layer: Optional[int] = None,
        batch_size: int = 64,
    ):
        self.model_name  = model_name
        self.repr_layer  = repr_layer
        self.batch_size  = batch_size
        self._model      = None
        self._alphabet   = None
        self._batch_converter = None

    def _load_model(self):
        if not ESM_AVAILABLE:
            raise ImportError(
                "ESM-2 not available. Install with:\n"
                "  pip install fair-esm torch\n"
                "Weights (~700MB for esm2_t6_8M_UR50D) download on first use."
            )
        if self._model is not None:
            return
        logger.info("Loading ESM-2 model: %s (CPU)", self.model_name)
        load_fn = getattr(esm.pretrained, self.model_name, None)
        if load_fn is None:
            raise ValueError(
                f"Unknown ESM-2 model: '{self.model_name}'. "
                f"Available: {list(self._MODEL_LAYERS.keys())}"
            )
        self._model, self._alphabet = load_fn()
        self._model.eval()
        self._batch_converter = self._alphabet.get_batch_converter()
        logger.info("ESM-2 model loaded.")

    @property
    def _layer(self) -> int:
        if self.repr_layer is not None:
            return self.repr_layer
        return self._MODEL_LAYERS.get(self.model_name, 6)

    def fit(self, X, y=None):
        # ESM-2 is a fixed pre-trained model; no fitting is required.
        # We still load it here so the transformer is ready after fit().
        self._load_model()
        return self

    @torch.no_grad()
    def transform(self, X: List[str]) -> np.ndarray:
        """
        X: list of amino acid strings (canonical only)
        Returns: ndarray of shape (N, embedding_dim)
        """
        self._load_model()
        layer = self._layer
        all_embeddings = []

        for start in range(0, len(X), self.batch_size):
            batch_seqs = X[start: start + self.batch_size]
            batch_data = [(f"seq_{i}", seq) for i, seq in enumerate(batch_seqs)]

            _, _, tokens = self._batch_converter(batch_data)
            results = self._model(
                tokens,
                repr_layers=[layer],
                return_contacts=False,
            )
            # token_representations: (B, seq_len+2, D)  (+2 for BOS/EOS)
            token_repr = results['representations'][layer]

            for i, (_, seq) in enumerate(batch_data):
                # Mean-pool over real residue positions (exclude BOS at 0, EOS at -1)
                L = len(seq)
                mean_repr = token_repr[i, 1:L + 1].mean(dim=0).numpy()
                all_embeddings.append(mean_repr)

        return np.vstack(all_embeddings).astype(np.float32)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    @property
    def embedding_dim(self) -> int:
        """Return output dimensionality (requires model to be loaded)."""
        if self._model is None:
            self._load_model()
        # ESM-2 embed_dim is accessible via model.embed_dim
        return self._model.embed_dim
