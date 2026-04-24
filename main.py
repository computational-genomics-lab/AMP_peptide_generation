"""
main.py
========
Pipeline entry point with CLI argument support.

Usage:
  python main.py --config config.yaml
  python main.py --config config.yaml --set generator.epochs=200
  python main.py --config config.yaml --set data.amp_pos_path=/abs/path.xlsx

Reproducibility:
  All random state is controlled by cfg['seed']. NumPy, Python random,
  and PyTorch seeds are set centrally here before any other imports execute.
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='ML-guided AMP+CPP peptide generation pipeline'
    )
    p.add_argument('--config', default='config.yaml',
                   help='Path to YAML configuration file')
    p.add_argument('--set', nargs='*', default=[],
                   metavar='KEY=VALUE',
                   help='Override config values, e.g. --set generator.epochs=200')
    p.add_argument('--skip-training', action='store_true',
                   help='Skip classifier and generator training (use cached models)')
    p.add_argument('--amp-model', default=None,
                   help='Path to saved AMP predict function (pickle)')
    p.add_argument('--cpp-model', default=None,
                   help='Path to saved CPP predict function (pickle)')
    p.add_argument('--generator', default=None,
                   help='Path to saved CVAE checkpoint')
    return p.parse_args()


def _parse_overrides(override_list):
    overrides = {}
    for item in override_list:
        if '=' not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: '{item}'")
        key, val = item.split('=', 1)
        # Try numeric conversion
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                if val.lower() in ('true', 'yes'):
                    val = True
                elif val.lower() in ('false', 'no'):
                    val = False
                # else keep as string
        overrides[key] = val
    return overrides


# ---------------------------------------------------------------------------
# Seed control
# ---------------------------------------------------------------------------

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Final output
# ---------------------------------------------------------------------------

def save_outputs(
    diverse: list,
    metrics_amp: Optional[dict],
    metrics_cpp: Optional[dict],
    novelty_report: Optional[dict],
    score_comparison_df: Optional[pd.DataFrame],
    output_dir: str,
) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Candidates CSV
    df = pd.DataFrame(diverse)
    col_order = [
        'rank', 'sequence', 'amp_score', 'cpp_score', 'combined_score',
        'cluster', 'length', 'molecular_weight', 'pI', 'instability_index',
        'GRAVY', 'aromaticity', 'aliphatic_index', 'boman_index',
        'ww_hydrophobicity', 'net_charge', 'hydrophobic_moment',
        'aggregation_propensity',
    ]
    df = df[[c for c in col_order if c in df.columns]]
    candidates_path = os.path.join(output_dir, 'candidates.csv')
    df.to_csv(candidates_path, index=False)

    # Score comparison
    if score_comparison_df is not None:
        score_comparison_df.to_csv(
            os.path.join(output_dir, 'score_comparison.csv'), index=False
        )

    # Text report
    _write_report(df, metrics_amp, metrics_cpp, novelty_report, output_dir)

    return candidates_path


def _write_report(df, metrics_amp, metrics_cpp, novelty, output_dir):
    lines = [
        '=' * 72,
        '  ML-GUIDED AMP+CPP PEPTIDE GENERATION — SUMMARY REPORT',
        f'  Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        '=' * 72,
        '',
        f'  Total candidates  : {len(df)}',
        f'  Diversity clusters: {df["cluster"].nunique() if "cluster" in df.columns else "N/A"}',
        '',
        '  Classifier performance (test set):',
    ]
    if metrics_amp:
        lines.append(
            f'    AMP  ROC-AUC={metrics_amp.get("test_roc_auc", "?"):.3f}  '
            f'PR-AUC={metrics_amp.get("test_pr_auc", "?"):.3f}  '
            f'Brier={metrics_amp.get("test_brier", "?"):.4f}'
        )
    if metrics_cpp:
        lines.append(
            f'    CPP  ROC-AUC={metrics_cpp.get("test_roc_auc", "?"):.3f}  '
            f'PR-AUC={metrics_cpp.get("test_pr_auc", "?"):.3f}  '
            f'Brier={metrics_cpp.get("test_brier", "?"):.4f}'
        )
    lines += [
        '',
        '  Candidate score statistics:',
        f'    AMP  mean={df["amp_score"].mean():.3f}  max={df["amp_score"].max():.3f}',
        f'    CPP  mean={df["cpp_score"].mean():.3f}  max={df["cpp_score"].max():.3f}',
        f'    Comb mean={df["combined_score"].mean():.3f}  max={df["combined_score"].max():.3f}',
    ]
    if novelty:
        lines += [
            '',
            '  Novelty (vs. training set):',
            f'    Mean distance : {novelty.get("mean_distance", "?"):.3f}',
            f'    Memorised     : {novelty.get("pct_memorised", "?")}%',
            f'    Truly novel   : {novelty.get("pct_novel", "?")}%',
        ]
    lines += [
        '',
        '  Top 10 candidates:',
        '-' * 72,
        f'  {"Rank":>4}  {"Sequence":<45}  {"AMP":>5}  {"CPP":>5}  {"Comb":>5}',
        '-' * 72,
    ]
    for row in df.head(10).itertuples():
        lines.append(
            f'  {row.rank:>4}  {row.sequence:<45}  '
            f'{row.amp_score:>5.3f}  {row.cpp_score:>5.3f}  '
            f'{row.combined_score:>5.3f}'
        )
    lines.append('=' * 72)

    report_path = os.path.join(output_dir, 'report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    logging.getLogger(__name__).info("Report saved: %s", report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    overrides = _parse_overrides(args.set)

    from utils.logging_utils import setup_logging, load_config, versioned_output_dir
    logger = setup_logging(log_dir='logs')

    logger.info("Loading config from %s", args.config)
    cfg = load_config(args.config, overrides)

    seed = cfg.get('seed', 42)
    set_global_seed(seed)
    logger.info("Global seed: %d", seed)

    out_base = cfg.get('output', {}).get('dir', 'outputs')
    if cfg.get('output', {}).get('versioned', True):
        output_dir = versioned_output_dir(out_base)
    else:
        output_dir = out_base
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save resolved config to output dir for reproducibility
    import yaml
    with open(os.path.join(output_dir, 'config_used.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    data_cfg  = cfg.get('data', {})
    model_cfg = cfg.get('model', {})
    gen_cfg   = cfg.get('generator', {})
    filt_cfg  = cfg.get('filters', {})
    div_cfg   = cfg.get('diversity', {})
    gen_run   = cfg.get('generation', {})

    metrics_amp = metrics_cpp = None

    # ── Phase 1-3: Train classifiers ──────────────────────────────────────
    if not args.skip_training:
        from training.train import (
            train_amp_classifier, train_cpp_classifier, train_generator
        )

        predict_amp, amp_train_pos = train_amp_classifier(
            amp_pos_path=data_cfg['amp_pos_path'],
            amp_neg_path=data_cfg.get('amp_neg_path'),
            output_dir=output_dir,
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            seed=seed,
        )
        with open(os.path.join(output_dir, 'amp_metrics.json')) as f:
            metrics_amp = json.load(f)

        predict_cpp = train_cpp_classifier(
            cpp_path=data_cfg['cpp_path'],
            cpp_neg_path=data_cfg.get('cpp_neg_path'),
            output_dir=output_dir,
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            seed=seed,
        )
        with open(os.path.join(output_dir, 'cpp_metrics.json')) as f:
            metrics_cpp = json.load(f)

        # ── Phase 4: Train CVAE ───────────────────────────────────────────
        generator = train_generator(
            amp_train_positives=amp_train_pos,
            generator_cfg=gen_cfg,
            seed=seed,
        )
        amp_training_seqs = amp_train_pos['sequence'].tolist()

    else:
        # Load pre-trained models (implement serialisation as needed)
        raise NotImplementedError(
            "--skip-training requires pre-saved model objects. "
            "Implement pickle serialisation or save/load logic."
        )

    # ── Phase 5: Generate and optimise ────────────────────────────────────
    from generation.pipeline import generate_and_optimise
    candidates = generate_and_optimise(
        generator=generator,
        predict_amp=predict_amp,
        predict_cpp=predict_cpp,
        n_initial=gen_run.get('n_initial', 3000),
        n_optimise_rounds=gen_run.get('n_optimise_rounds', 5),
        top_k=gen_run.get('top_k', 300),
        temperature=gen_run.get('temperature', 1.1),
        mutation_ns=gen_run.get('mutation_ns', [1, 2, 3]),
        softmax_temperature=gen_run.get('softmax_temperature', 0.3),
        seed=seed,
    )

    # ── Phase 6: Biological filters ───────────────────────────────────────
    from filters.biological import FilterConfig, apply_biological_filters, relax_and_refilter
    filter_config = FilterConfig.from_dict(filt_cfg)
    passed = apply_biological_filters(candidates, filter_config)

    if not passed:
        passed, _ = relax_and_refilter(candidates, filter_config)

    if not passed:
        logger.error("No candidates passed biological filters. "
                     "Check CVAE quality and predictor scores.")
        sys.exit(1)

    # ── Phase 7: Diversity control ────────────────────────────────────────
    from diversity.cluster import apply_diversity_control
    diverse = apply_diversity_control(
        candidates=passed,
        max_identity=div_cfg.get('max_pairwise_identity', 0.80),
        n_clusters=div_cfg.get('n_clusters', 20),
        top_per_cluster=div_cfg.get('top_per_cluster', 4),
        seed=seed,
    )

    # ── Phase 8: Evaluation ───────────────────────────────────────────────
    from training.evaluate import compare_score_distributions, novelty_report
    generated_seqs_all = list(set(c[0] for c in candidates))

    score_df = compare_score_distributions(
        generated_seqs=generated_seqs_all[:500],
        training_pos_seqs=amp_training_seqs,
        predict_amp=predict_amp,
        predict_cpp=predict_cpp,
        output_path=os.path.join(output_dir, 'score_comparison.csv'),
    )

    novelty = novelty_report(
        generated_seqs=generated_seqs_all[:200],
        training_seqs=amp_training_seqs,
        output_path=os.path.join(output_dir, 'novelty_report.json'),
    )

    # ── Phase 9: Save outputs ─────────────────────────────────────────────
    candidates_path = save_outputs(
        diverse=diverse,
        metrics_amp=metrics_amp,
        metrics_cpp=metrics_cpp,
        novelty_report=novelty,
        score_comparison_df=score_df,
        output_dir=output_dir,
    )

    logger.info("Pipeline complete.")
    logger.info("Candidates CSV : %s", candidates_path)
    logger.info("Output dir     : %s", output_dir)


if __name__ == '__main__':
    main()
