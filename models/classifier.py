"""
models/classifier.py
=====================
Ensemble classifier (GBM + MLP + RF) built inside a sklearn Pipeline.

CRITICAL FIX (data leakage):
  Original code:
    1. Fit FeatureBuilder on ALL data
    2. Extract features for ALL data
    3. Run cross_val_score on the pre-extracted feature matrix
  Step 1+2 before step 3 is the leakage: TF-IDF vocabulary, SVD
  directions, and StandardScaler statistics all carry information from
  held-out fold data.

  Fix: wrap the feature extractor in a sklearn Pipeline so that
  fit() is only called on training fold data within each CV iteration.

CRITICAL FIX (calibration):
  Original code passed the full training set to CalibratedClassifierCV,
  which internally re-fits the base classifier and calibrator on the same
  data — no held-out calibration set.

  Fix: calibrate explicitly on the validation split using
  CalibratedClassifierCV(cv='prefit') which expects a pre-fitted
  base estimator and calibrates on the provided data alone.

CRITICAL FIX (class imbalance):
  Original code generated shuffled negatives. Now we use
  class_weight='balanced' in GBM, RF, and sample_weight in MLP
  training to handle any residual class imbalance in real data.
"""

import logging
from typing import Tuple, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from features.transformers import CombinedSequenceTransformer

logger = logging.getLogger(__name__)


# =============================================================================
# Base estimators
# =============================================================================

def _make_gbm(seed: int = 42) -> GradientBoostingClassifier:
    # GBM does not support class_weight natively; handle via sample_weight
    # in cross_validate or via subsample
    return GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.07,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=seed,
        validation_fraction=0.1,
        n_iter_no_change=20,
        tol=1e-4,
    )


def _make_mlp(seed: int = 42) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        max_iter=500,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        tol=1e-4,
    )


def _make_rf(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=seed,
        n_jobs=-1,
    )


# =============================================================================
# Pipeline builder
# =============================================================================

def build_pipeline(
    estimator,
    k: int = 3,
    n_components: int = 64,
    max_features: int = 5000,
) -> Pipeline:
    """
    Wrap an estimator in a Pipeline that includes feature extraction.

    The CombinedSequenceTransformer fits TF-IDF, SVD, and StandardScaler
    from scratch on each training fold — preventing leakage.

    Input X to this Pipeline is a list/array of amino acid strings.
    """
    return Pipeline([
        ('features', CombinedSequenceTransformer(
            k=k,
            n_components=n_components,
            max_features=max_features,
        )),
        ('clf', estimator),
    ])


# =============================================================================
# Class-balanced sample weights
# =============================================================================

def _balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Compute per-sample weights so each class contributes equally to loss.
    Used for GBM and MLP which lack built-in class_weight='balanced'.
    """
    classes, counts = np.unique(y, return_counts=True)
    weight_per_class = len(y) / (len(classes) * counts)
    class_weight_map = dict(zip(classes, weight_per_class))
    return np.array([class_weight_map[yi] for yi in y])


# =============================================================================
# Cross-validation with sample weights
# =============================================================================

def _cv_with_weights(pipeline, X, y, cv, scoring='roc_auc'):
    """
    Run StratifiedKFold CV passing balanced sample_weights for GBM/MLP.
    RF already uses class_weight='balanced' internally.
    """
    clf_step = pipeline.named_steps['clf']
    use_weights = isinstance(
        clf_step, (GradientBoostingClassifier, MLPClassifier)
    )

    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr = [X[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_tr = y[train_idx]
        y_val = y[val_idx]

        fit_params = {}
        if use_weights:
            sw = _balanced_sample_weights(y_tr)
            fit_params['clf__sample_weight'] = sw

        pipeline_clone = _clone_pipeline(pipeline)
        pipeline_clone.fit(X_tr, y_tr, **fit_params)
        prob = pipeline_clone.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, prob))

    return np.array(scores)


def _clone_pipeline(pipeline: Pipeline) -> Pipeline:
    from sklearn.base import clone
    return clone(pipeline)


# =============================================================================
# Main training function
# =============================================================================

def train_predictor(
    train_seqs,
    train_labels,
    val_seqs,
    val_labels,
    cv_folds: int = 5,
    seed: int = 42,
    k: int = 3,
    n_components: int = 64,
    calibration_method: str = 'isotonic',
) -> Tuple[object, dict]:
    """
    Train an ensemble classifier and calibrate on the validation set.

    Returns:
      predict_fn  — callable(seqs: list[str]) → np.ndarray of probabilities
      metrics     — dict with CV ROC-AUC, val ROC-AUC, PR-AUC, Brier

    Protocol:
      1. 5-fold stratified CV on training set → CV ROC-AUC
      2. Refit each base model on the full training set
      3. Calibrate on validation set using CalibratedClassifierCV(cv='prefit')
      4. Report test-like metrics on validation set (for model selection)
      NOTE: test set is never touched here.
    """
    X_tr = list(train_seqs)
    y_tr = np.array(train_labels)
    X_val = list(val_seqs)
    y_val = np.array(val_labels)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    gbm_pipe = build_pipeline(_make_gbm(seed), k=k, n_components=n_components)
    mlp_pipe = build_pipeline(_make_mlp(seed), k=k, n_components=n_components)
    rf_pipe = build_pipeline(_make_rf(seed), k=k, n_components=n_components)

    logger.info("Running %d-fold CV for GBM...", cv_folds)
    gbm_cv_scores = _cv_with_weights(gbm_pipe, X_tr, y_tr, cv)
    logger.info("  GBM CV ROC-AUC: %.3f ± %.3f",
                gbm_cv_scores.mean(), gbm_cv_scores.std())

    logger.info("Running %d-fold CV for MLP...", cv_folds)
    mlp_cv_scores = _cv_with_weights(mlp_pipe, X_tr, y_tr, cv)
    logger.info("  MLP CV ROC-AUC: %.3f ± %.3f",
                mlp_cv_scores.mean(), mlp_cv_scores.std())

    logger.info("Running %d-fold CV for RF...", cv_folds)
    rf_cv_scores = _cv_with_weights(rf_pipe, X_tr, y_tr, cv)
    logger.info("  RF  CV ROC-AUC: %.3f ± %.3f",
                rf_cv_scores.mean(), rf_cv_scores.std())

    # -- Refit on full training set ----------------------------------------
    sw_tr = _balanced_sample_weights(y_tr)

    logger.info("Refitting GBM on full training set...")
    gbm_pipe.fit(X_tr, y_tr, clf__sample_weight=sw_tr)

    logger.info("Refitting MLP on full training set...")
    mlp_pipe.fit(X_tr, y_tr, clf__sample_weight=sw_tr)

    logger.info("Refitting RF on full training set...")
    rf_pipe.fit(X_tr, y_tr)   # RF uses class_weight='balanced' internally

    # -- Calibration on VALIDATION set only --------------------------------
    # FIX: original used CalibratedClassifierCV(cv=3) on training data,
    # which re-fits base models on nested CV folds of training data.
    # This mixes calibration with training data.
    # Correct approach: prefit base model, calibrate on independent val set.
    logger.info("Calibrating on validation set...")

    def _calibrate(pipe, X_val, y_val, method):
        cal = CalibratedClassifierCV(
            estimator=pipe,
            cv='prefit',
            method=method,
        )
        cal.fit(X_val, y_val)
        return cal

    gbm_cal = _calibrate(gbm_pipe, X_val, y_val, calibration_method)
    mlp_cal = _calibrate(mlp_pipe, X_val, y_val, calibration_method)
    rf_cal  = _calibrate(rf_pipe,  X_val, y_val, calibration_method)

    # -- Evaluate on validation set ----------------------------------------
    def _ensemble_proba(seqs):
        p_gbm = gbm_cal.predict_proba(seqs)[:, 1]
        p_mlp = mlp_cal.predict_proba(seqs)[:, 1]
        p_rf  = rf_cal.predict_proba(seqs)[:, 1]
        return (p_gbm + p_mlp + p_rf) / 3.0

    p_val = _ensemble_proba(X_val)
    roc_auc = roc_auc_score(y_val, p_val)
    pr_auc  = average_precision_score(y_val, p_val)
    brier   = brier_score_loss(y_val, p_val)
    logger.info(
        "Validation — ROC-AUC: %.3f  PR-AUC: %.3f  Brier: %.4f",
        roc_auc, pr_auc, brier,
    )

    metrics = {
        'gbm_cv_roc_auc': float(gbm_cv_scores.mean()),
        'mlp_cv_roc_auc': float(mlp_cv_scores.mean()),
        'rf_cv_roc_auc':  float(rf_cv_scores.mean()),
        'val_roc_auc':    float(roc_auc),
        'val_pr_auc':     float(pr_auc),
        'val_brier':      float(brier),
    }

    return _ensemble_proba, metrics


# =============================================================================
# Final test-set evaluation (call ONCE, at the very end)
# =============================================================================

def evaluate_on_test(
    predict_fn,
    test_seqs,
    test_labels,
) -> dict:
    """
    Compute ROC-AUC, PR-AUC, calibration metrics on the held-out test set.

    This function should be called exactly ONCE after all model selection
    and hyperparameter tuning is complete.
    """
    X_test = list(test_seqs)
    y_test = np.array(test_labels)
    p_test = predict_fn(X_test)

    roc_auc  = roc_auc_score(y_test, p_test)
    pr_auc   = average_precision_score(y_test, p_test)
    brier    = brier_score_loss(y_test, p_test)

    fpr, tpr, _ = roc_curve(y_test, p_test)
    prec, rec, _ = precision_recall_curve(y_test, p_test)

    # Calibration curve (10 bins)
    frac_pos, mean_pred = calibration_curve(y_test, p_test, n_bins=10)

    logger.info(
        "TEST SET — ROC-AUC: %.3f  PR-AUC: %.3f  Brier: %.4f",
        roc_auc, pr_auc, brier,
    )

    return {
        'test_roc_auc': float(roc_auc),
        'test_pr_auc':  float(pr_auc),
        'test_brier':   float(brier),
        'roc_fpr':      fpr.tolist(),
        'roc_tpr':      tpr.tolist(),
        'pr_precision': prec.tolist(),
        'pr_recall':    rec.tolist(),
        'cal_frac_pos': frac_pos.tolist(),
        'cal_mean_pred': mean_pred.tolist(),
    }
