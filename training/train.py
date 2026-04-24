"""
training/train.py
==================
Top-level training orchestration for AMP and CPP classifiers + CVAE generator.

Evaluation reporting:
- CV ROC-AUC (training fold only)
- Validation ROC-AUC, PR-AUC, Brier score (calibration quality)
- Test ROC-AUC, PR-AUC, Brier score (final, call once)
- Calibration curves saved to output directory
"""

import json
import logging
import os
from pathlib import Path
from typing import Tuple, Optional, Callable

import numpy as np
import pandas as pd

from data.loader import (
    build_amp_dataset, build_cpp_dataset, stratified_split
)
from models.classifier import train_predictor, evaluate_on_test

logger = logging.getLogger(__name__)


# =============================================================================
# Helper: save evaluation metrics
# =============================================================================

def _save_metrics(metrics: dict, path: str):
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", path)


# =============================================================================
# AMP classifier
# =============================================================================

def train_amp_classifier(
    amp_pos_path: str,
    amp_neg_path: Optional[str],
    output_dir: str,
    data_cfg: dict,
    model_cfg: dict,
    seed: int = 42,
) -> Tuple[Callable, pd.DataFrame]:
    """
    Train, calibrate, and evaluate the AMP classifier.

    Returns (predict_fn, amp_train_df) where amp_train_df is the training
    split used to train the CVAE generator (positives only).
    """
    logger.info("=" * 60)
    logger.info("AMP CLASSIFIER TRAINING")
    logger.info("=" * 60)

    dataset = build_amp_dataset(
        pos_path=amp_pos_path,
        neg_path=amp_neg_path,
        min_len=data_cfg.get('min_len', 5),
        max_len=data_cfg.get('max_len', 60),
    )

    train_df, val_df, test_df = stratified_split(
        dataset,
        val_frac=data_cfg.get('val_frac', 0.15),
        test_frac=data_cfg.get('test_frac', 0.15),
        seed=seed,
    )

    predict_amp, val_metrics = train_predictor(
        train_seqs=train_df['sequence'].tolist(),
        train_labels=train_df['label'].values,
        val_seqs=val_df['sequence'].tolist(),
        val_labels=val_df['label'].values,
        cv_folds=model_cfg.get('cv_folds', 5),
        seed=seed,
        calibration_method=model_cfg.get('calibration_method', 'isotonic'),
    )

    # Final test evaluation (called ONCE here)
    test_metrics = evaluate_on_test(
        predict_amp,
        test_df['sequence'].tolist(),
        test_df['label'].values,
    )

    all_metrics = {**val_metrics, **test_metrics}
    _save_metrics(all_metrics, os.path.join(output_dir, 'amp_metrics.json'))

    logger.info("AMP TEST ROC-AUC: %.3f  PR-AUC: %.3f  Brier: %.4f",
                test_metrics['test_roc_auc'],
                test_metrics['test_pr_auc'],
                test_metrics['test_brier'])

    # Return training positives for CVAE
    amp_train_pos = train_df[train_df['label'] == 1].reset_index(drop=True)
    return predict_amp, amp_train_pos


# =============================================================================
# CPP classifier
# =============================================================================

def train_cpp_classifier(
    cpp_path: str,
    cpp_neg_path: Optional[str],
    output_dir: str,
    data_cfg: dict,
    model_cfg: dict,
    seed: int = 42,
) -> Callable:
    """
    Train, calibrate, and evaluate the CPP classifier.
    """
    logger.info("=" * 60)
    logger.info("CPP CLASSIFIER TRAINING")
    logger.info("=" * 60)

    dataset = build_cpp_dataset(
        cpp_path=cpp_path,
        neg_path=cpp_neg_path,
        min_len=data_cfg.get('min_len', 5),
        max_len=data_cfg.get('max_len', 60),
    )

    train_df, val_df, test_df = stratified_split(
        dataset,
        val_frac=data_cfg.get('val_frac', 0.15),
        test_frac=data_cfg.get('test_frac', 0.15),
        seed=seed,
    )

    predict_cpp, val_metrics = train_predictor(
        train_seqs=train_df['sequence'].tolist(),
        train_labels=train_df['label'].values,
        val_seqs=val_df['sequence'].tolist(),
        val_labels=val_df['label'].values,
        cv_folds=model_cfg.get('cv_folds', 5),
        seed=seed,
        calibration_method=model_cfg.get('calibration_method', 'isotonic'),
    )

    test_metrics = evaluate_on_test(
        predict_cpp,
        test_df['sequence'].tolist(),
        test_df['label'].values,
    )

    all_metrics = {**val_metrics, **test_metrics}
    _save_metrics(all_metrics, os.path.join(output_dir, 'cpp_metrics.json'))

    logger.info("CPP TEST ROC-AUC: %.3f  PR-AUC: %.3f  Brier: %.4f",
                test_metrics['test_roc_auc'],
                test_metrics['test_pr_auc'],
                test_metrics['test_brier'])

    return predict_cpp


# =============================================================================
# CVAE generator
# =============================================================================

def train_generator(
    amp_train_positives: pd.DataFrame,
    generator_cfg: dict,
    seed: int = 42,
):
    """
    Train the CVAE on AMP positive training sequences.

    Selects PyTorch backend if available; falls back to NumPy.
    Note: generator is trained only on training-set positives, never on
    validation or test sequences.
    """
    logger.info("=" * 60)
    logger.info("CVAE GENERATOR TRAINING")
    logger.info("=" * 60)

    seqs    = amp_train_positives['sequence'].tolist()
    lengths = [len(s) for s in seqs]
    max_len = int(np.percentile(lengths, 90))
    max_len = min(max(max_len, 15), generator_cfg.get('max_len', 50))

    logger.info("Training CVAE on %d sequences | max_len=%d", len(seqs), max_len)

    backend = generator_cfg.get('backend', 'torch')
    use_torch = (backend == 'torch')

    if use_torch:
        try:
            import torch  # noqa: F401 — just verify it's available
            from models.cvae_torch import train_cvae
            logger.info("Using PyTorch CVAE backend")
            model = train_cvae(
                seqs=seqs,
                max_len=max_len,
                latent_dim=generator_cfg.get('latent_dim', 64),
                hidden_dim=generator_cfg.get('hidden_dim', 256),
                epochs=generator_cfg.get('epochs', 150),
                batch_size=generator_cfg.get('batch_size', 64),
                lr=generator_cfg.get('lr', 1e-3),
                grad_clip=generator_cfg.get('grad_clip', 5.0),
                kl_beta=generator_cfg.get('kl_beta', 1.0),
                kl_warmup_epochs=generator_cfg.get('kl_warmup_epochs', 60),
                checkpoint_dir=generator_cfg.get('checkpoint_dir', None),
                checkpoint_every=generator_cfg.get('checkpoint_every', 25),
                resume_from=generator_cfg.get('resume_from', None),
                device='cpu',
            )
            return model
        except ImportError:
            logger.warning("PyTorch not available; falling back to NumPy CVAE")

    logger.info("Using NumPy CVAE backend")
    from models.cvae_numpy import PeptideCVAENP
    np.random.seed(seed)
    model = PeptideCVAENP(
        max_len=max_len,
        latent_dim=generator_cfg.get('latent_dim', 64),
        hidden_dim=generator_cfg.get('hidden_dim', 256),
        lr=generator_cfg.get('lr', 1e-3),
        seed=seed,
    )
    model.train(
        seqs=seqs,
        epochs=generator_cfg.get('epochs', 150),
        batch_size=generator_cfg.get('batch_size', 64),
        kl_beta=generator_cfg.get('kl_beta', 1.0),
        kl_warmup_epochs=generator_cfg.get('kl_warmup_epochs', 60),
        grad_clip=generator_cfg.get('grad_clip', 5.0),
        checkpoint_dir=generator_cfg.get('checkpoint_dir', None),
        checkpoint_every=generator_cfg.get('checkpoint_every', 25),
        resume_from=generator_cfg.get('resume_from', None),
    )
    return model
