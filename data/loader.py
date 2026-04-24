"""
data/loader.py
===============
Data loading, cleaning, and stratified splitting.

CRITICAL FIX (synthetic negatives):
  Original code generated negative class samples by shuffling positive
  sequences. This is biologically invalid: shuffled sequences maintain
  amino acid composition but destroy structural context, and the
  classifier learns to separate by composition rather than by true
  biological signal. Any classifier trained this way will be unable to
  distinguish real non-AMPs/non-CPPs from real AMPs/CPPs with different
  compositions.

  Fix:
  - For CPP: the CellPPD file (CellPPD / Raghavendra) contains ONLY
    experimentally validated CPP sequences. The columns 'type / CPP /
    Non-CPP' are predictions from an external predictor — they are NOT
    experimental labels and must not be used for classification targets.
    Every row in this file is a positive (label=1). A separate file of
    real non-CPP sequences is required, same as for AMP negatives.
  - For AMP: requires a separate file of non-AMP peptides (e.g., from
    UniProt cytosolic/non-secretory proteins). If not provided, the
    pipeline raises an informative error directing the user to obtain
    real negatives. As an interim workaround (explicit, not silent),
    class_weight='balanced' can be passed to classifiers instead of
    generating fake negatives.

CRITICAL FIX (data split):
  Original code never created a held-out test set. Cross-validation
  was the only evaluation, and calibration was performed on the full
  training data. Fixed by:
  - Stratified 3-way split: train / validation / test
  - Calibration performed on validation set only
  - Test set used exclusively for final metric reporting
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from features.physicochemical import is_canonical

logger = logging.getLogger(__name__)


# =============================================================================
# Column name constants (adapt these if your file has different headers)
# =============================================================================
AMP_SEQ_COL = 'Sequence'
AMP_ID_COL = 'SeqID'
CPP_SEQ_COL = 'Protein_Sequence'
CPP_ID_COL = 'Protein_ID'


# =============================================================================
# Low-level file readers
# =============================================================================

def _read_amp_positives(path: str) -> pd.DataFrame:
    """
    Load the AMP positive dataset.
    Supports .xlsx and .csv.
    Expected columns: SeqID, Sequence (+ optional score columns).
    """
    p = Path(path)
    if p.suffix in ('.xlsx', '.xls'):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if AMP_SEQ_COL not in df.columns:
        raise ValueError(
            f"AMP file must contain a '{AMP_SEQ_COL}' column. "
            f"Found: {df.columns.tolist()}"
        )
    return df


def _read_amp_negatives(path: str) -> pd.DataFrame:
    """
    Load non-AMP negative sequences.
    Minimum requirement: a 'Sequence' column.
    Recommended source: UniProt non-secretory cytosolic proteins,
    filtered to 5–60 AA length and canonical residues only.
    """
    p = Path(path)
    if p.suffix in ('.xlsx', '.xls'):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding='latin1')
    df.columns = [c.strip() for c in df.columns]
    if AMP_SEQ_COL not in df.columns:
        # Try common alternatives
        for alt in ['sequence', 'seq', 'SEQUENCE', 'Seq']:
            if alt in df.columns:
                df = df.rename(columns={alt: AMP_SEQ_COL})
                break
        else:
            raise ValueError(
                f"AMP negatives file must contain a 'Sequence' column. "
                f"Found: {df.columns.tolist()}"
            )
    return df


def _read_cpp_dataset(path: str) -> pd.DataFrame:
    """
    Load the CellPPD-format CPP dataset.

    All rows are experimentally validated CPP positives.
    The trailing merged column ('type  CPP  Non-CPP') contains predictions
    from the CellPPD predictor tool — these are NOT experimental labels
    and are dropped entirely. Only the sequence column is retained.
    """
    df = pd.read_csv(path, encoding='latin1')
    df.columns = [
        c.strip().replace('\xa0', '').replace(' ', '_')
        for c in df.columns
    ]
    # Drop the predictor-output column (last column, merged format)
    df = df.drop(columns=[df.columns[-1]])
    return df


# =============================================================================
# Cleaning
# =============================================================================

def _clean_sequences(
    df: pd.DataFrame,
    seq_col: str,
    min_len: int,
    max_len: int,
) -> pd.DataFrame:
    df = df.copy()
    df[seq_col] = df[seq_col].astype(str).str.strip().str.upper()
    df = df.dropna(subset=[seq_col])
    df = df[df[seq_col].apply(is_canonical)]
    df = df[df[seq_col].str.len().between(min_len, max_len)]
    df = df.drop_duplicates(subset=seq_col)
    return df.reset_index(drop=True)


# =============================================================================
# Dataset assemblers
# =============================================================================

def build_amp_dataset(
    pos_path: str,
    neg_path: Optional[str],
    min_len: int = 5,
    max_len: int = 60,
) -> pd.DataFrame:
    """
    Build a labelled AMP dataset.

    Returns a DataFrame with columns ['sequence', 'label'] where
    label=1 → AMP positive, label=0 → non-AMP negative.

    If neg_path is None or the file doesn't exist, the function raises
    a clear error instead of generating synthetic negatives silently.
    """
    logger.info("Loading AMP positives from %s", pos_path)
    pos_df = _read_amp_positives(pos_path)
    pos_df = _clean_sequences(pos_df, AMP_SEQ_COL, min_len, max_len)
    pos_df = pos_df[[AMP_SEQ_COL]].rename(columns={AMP_SEQ_COL: 'sequence'})
    pos_df['label'] = 1
    logger.info("AMP positives after cleaning: %d", len(pos_df))

    if neg_path is None or not Path(neg_path).exists():
        raise FileNotFoundError(
            "No real AMP negative dataset found at path: "
            f"'{neg_path}'.\n"
            "You MUST provide experimentally validated non-AMP sequences.\n"
            "Recommended source: UniProt cytosolic proteins (reviewed),\n"
            "filtered to 5-60 AA, no signal peptides, no antimicrobial\n"
            "annotations. Download via UniProt REST API:\n"
            "  https://rest.uniprot.org/uniprotkb/search?"
            "query=reviewed:true+AND+length:[5+TO+60]"
            "+AND+NOT+keyword:KW-0929&format=fasta\n"
            "If you intentionally want to skip negatives and rely on\n"
            "class_weight='balanced', set amp_neg_path to 'BALANCE_ONLY'\n"
            "in config.yaml."
        )

    logger.info("Loading AMP negatives from %s", neg_path)
    neg_df = _read_amp_negatives(neg_path)
    neg_df = _clean_sequences(neg_df, AMP_SEQ_COL, min_len, max_len)
    neg_df = neg_df[[AMP_SEQ_COL]].rename(columns={AMP_SEQ_COL: 'sequence'})
    neg_df['label'] = 0
    logger.info("AMP negatives after cleaning: %d", len(neg_df))

    # Remove any negatives that appear in positives (sequence overlap)
    pos_seqs = set(pos_df['sequence'])
    neg_df = neg_df[~neg_df['sequence'].isin(pos_seqs)]
    logger.info(
        "AMP negatives after removing positives overlap: %d", len(neg_df)
    )

    dataset = pd.concat([pos_df, neg_df], ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    n_pos = dataset['label'].sum()
    n_neg = (dataset['label'] == 0).sum()
    ratio = n_pos / n_neg if n_neg > 0 else float('inf')
    logger.info(
        "AMP dataset: %d positive, %d negative  (ratio %.2f)",
        n_pos, n_neg, ratio,
    )
    if ratio > 5 or ratio < 0.2:
        logger.warning(
            "Class imbalance ratio %.2f is severe. Ensure classifiers use "
            "class_weight='balanced' or apply undersampling.", ratio
        )

    return dataset


def build_cpp_dataset(
    cpp_path: str,
    neg_path: Optional[str],
    min_len: int = 5,
    max_len: int = 60,
) -> pd.DataFrame:
    """
    Build a labelled CPP dataset.

    All sequences in the CellPPD file are experimentally validated CPPs
    (label=1). The predictor-output columns are discarded. A separate
    file of real non-CPP sequences is required for the negative class,
    for the same reason as AMP negatives: using predicted labels or
    shuffled sequences as negatives produces a biologically invalid model.

    Recommended non-CPP sources:
      - CPPsite 2.0 non-CPP set: https://crdd.osdd.net/raghava/cppsite/
      - UniProt cytosolic short peptides with no CPP annotation
    """
    logger.info("Loading CPP positives from %s", cpp_path)
    df = _read_cpp_dataset(cpp_path)
    df = df.rename(columns={CPP_SEQ_COL: 'sequence'})
    df = _clean_sequences(df, 'sequence', min_len, max_len)
    df = df[['sequence']].copy()
    df['label'] = 1
    logger.info("CPP positives after cleaning: %d", len(df))

    if neg_path is None or not Path(neg_path).exists():
        raise FileNotFoundError(
            "No real CPP negative dataset found at path: "
            f"'{neg_path}'.\n"
            "You MUST provide experimentally validated non-CPP sequences.\n"
            "The CellPPD predictor output columns ('type / CPP / Non-CPP')\n"
            "are NOT experimental labels and cannot be used as negatives.\n"
            "Recommended sources:\n"
            "  - CPPsite 2.0 non-CPP set: https://crdd.osdd.net/raghava/cppsite/\n"
            "  - UniProt cytosolic peptides with no CPP keyword (KW-0985)"
        )

    logger.info("Loading CPP negatives from %s", neg_path)
    neg_df = _read_amp_negatives(neg_path)   # reuse reader (expects 'Sequence')
    neg_df = _clean_sequences(neg_df, AMP_SEQ_COL, min_len, max_len)
    neg_df = neg_df[[AMP_SEQ_COL]].rename(columns={AMP_SEQ_COL: 'sequence'})
    neg_df['label'] = 0
    logger.info("CPP negatives after cleaning: %d", len(neg_df))

    # Remove overlap
    pos_seqs = set(df['sequence'])
    neg_df = neg_df[~neg_df['sequence'].isin(pos_seqs)]
    logger.info("CPP negatives after overlap removal: %d", len(neg_df))

    dataset = pd.concat([df, neg_df], ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    n_pos = dataset['label'].sum()
    n_neg = (dataset['label'] == 0).sum()
    ratio = n_pos / n_neg if n_neg > 0 else float('inf')
    logger.info("CPP dataset: %d positive, %d negative  (ratio %.2f)",
                n_pos, n_neg, ratio)
    if ratio > 5 or ratio < 0.2:
        logger.warning(
            "Class imbalance ratio %.2f is severe. Classifiers will use "
            "class_weight='balanced'.", ratio
        )
    return dataset


# =============================================================================
# Stratified 3-way split
# =============================================================================

def stratified_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / validation / test split.

    CRITICAL FIX:
    Original pipeline had NO test set — cross-validation was the only
    evaluation. Calibration was performed on training data, not a held-out
    validation set. This function creates a proper 3-way split:
      - Train: used for model fitting and CV
      - Validation: used for calibration and hyperparameter selection
      - Test: held-out, used ONLY for final metric reporting

    The test set is never touched during training or calibration.
    """
    y = df['label'].values

    # First split off the test set
    train_val, test = train_test_split(
        df, test_size=test_frac, stratify=y, random_state=seed
    )
    # Then split validation from the remaining train+val
    val_relative = val_frac / (1.0 - test_frac)
    train, val = train_test_split(
        train_val,
        test_size=val_relative,
        stratify=train_val['label'].values,
        random_state=seed,
    )

    logger.info(
        "Split → train: %d  val: %d  test: %d",
        len(train), len(val), len(test),
    )

    for split_name, split_df in [('train', train), ('val', val), ('test', test)]:
        n_pos = split_df['label'].sum()
        n_neg = (split_df['label'] == 0).sum()
        logger.info("  %s: %d pos / %d neg", split_name, n_pos, n_neg)

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )
