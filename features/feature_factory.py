"""
features/feature_factory.py
============================
Factory that builds the correct feature transformer based on config.

If use_esm=True (and fair-esm is installed), returns an ESM2Transformer.
Otherwise falls back to CombinedSequenceTransformer (k-mer TF-IDF + SVD
+ scaled physicochemical descriptors).

The returned transformer has a consistent sklearn interface:
  .fit(seqs)  .transform(seqs)  .fit_transform(seqs)
"""

import logging
from typing import Optional

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

logger = logging.getLogger(__name__)


def build_feature_transformer(
    use_esm: bool = False,
    esm_model: str = 'esm2_t6_8M_UR50D',
    esm_layer: Optional[int] = None,
    k: int = 3,
    n_components: int = 64,
    max_features: int = 5000,
) -> BaseEstimator:
    """
    Return the appropriate feature transformer.

    ESM-2 path:
      - No fit-time statistics (pure forward pass through frozen weights)
      - Higher quality representations; ~700MB model weights
      - Requires: pip install fair-esm torch

    k-mer + physicochemical path:
      - Must be fit inside CV folds (handled by sklearn Pipeline)
      - No external model downloads
      - Suitable for offline/air-gapped environments
    """
    if use_esm:
        try:
            from features.esm_embeddings import ESM2Transformer
            logger.info(
                "Feature extractor: ESM-2 (%s, layer %s)",
                esm_model, esm_layer or 'last',
            )
            return ESM2Transformer(
                model_name=esm_model,
                repr_layer=esm_layer,
            )
        except ImportError:
            logger.warning(
                "use_esm=True but 'fair-esm' is not installed. "
                "Falling back to k-mer + physicochemical features.\n"
                "To enable ESM-2: pip install fair-esm torch"
            )

    logger.info(
        "Feature extractor: k-mer TF-IDF (k=%d, SVD=%d) + physicochemical",
        k, n_components,
    )
    from features.transformers import CombinedSequenceTransformer
    return CombinedSequenceTransformer(
        k=k,
        n_components=n_components,
        max_features=max_features,
    )
