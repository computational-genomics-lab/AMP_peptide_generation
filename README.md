# AMP+CPP Peptide Generation Pipeline

ML-guided generation of antimicrobial peptides (AMPs) with cell-penetrating
peptide (CPP) activity. Refactored from scratch for scientific validity,
production robustness, and reproducibility.

---

## Project structure

```
amp_cpp_pipeline/
├── config.yaml                  # All thresholds and paths — edit this
├── main.py                      # Entry point
├── requirements.txt
│
├── data/
│   └── loader.py                # Data loading, cleaning, train/val/test split
│
├── features/
│   ├── physicochemical.py       # pKa-based charge, DIWV instability, hydrophobic moment
│   ├── transformers.py          # sklearn-compatible feature transformers (leak-free)
│   ├── esm_embeddings.py        # Optional ESM-2 embeddings
│   └── feature_factory.py       # Selects correct transformer from config
│
├── models/
│   ├── classifier.py            # Ensemble GBM+MLP+RF with correct calibration
│   ├── cvae_torch.py            # PyTorch CVAE (preferred backend)
│   └── cvae_numpy.py            # NumPy CVAE fallback
│
├── training/
│   ├── train.py                 # Training orchestration
│   └── evaluate.py              # Score distributions, novelty metrics
│
├── generation/
│   └── pipeline.py              # Score-guided mutation + softmax sampling
│
├── filters/
│   └── biological.py            # Configurable biological hard filters
│
├── diversity/
│   └── cluster.py               # Edit-distance + physicochemical diversity
│
└── utils/
    └── logging_utils.py         # Logging setup, config loader
```

---

## Setup

### 1. Conda environment (recommended)

```bash
conda create -n amp_cpp python=3.10 -y
conda activate amp_cpp
pip install -r requirements.txt
```

### 2. PyTorch (CPU) — optional but strongly recommended for CVAE

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. ESM-2 embeddings — optional, highest quality features

```bash
pip install fair-esm
# Weights (~700MB) download automatically on first run
```

---

## Data requirements

### AMP positives (provided)
Your `actual_antimicrobial_peptide_list_ADP6_*.xlsx` file.
Set `data.amp_pos_path` in `config.yaml`.

### AMP negatives (REQUIRED — you must obtain this)

The pipeline **will not generate synthetic negatives**. You must supply
real non-AMP sequences. Recommended sources:

**Option A — UniProt cytosolic proteins (best)**
```bash
# Download non-secretory, non-AMP reviewed human proteins 5-60 AA
curl "https://rest.uniprot.org/uniprotkb/search?\
query=reviewed:true+AND+length:[5+TO+60]\
+AND+NOT+keyword:KW-0929+AND+cc_subcellular_location:cytoplasm\
&format=fasta&size=5000" \
  > uniprot_cytosolic_negatives.fasta

# Convert FASTA to CSV (one column 'Sequence')
python - << 'EOF'
from Bio import SeqIO
import pandas as pd
seqs = [str(r.seq) for r in SeqIO.parse("uniprot_cytosolic_negatives.fasta", "fasta")]
pd.DataFrame({'Sequence': seqs}).to_csv("amp_negatives.csv", index=False)
EOF
```

**Option B — APD3 non-AMP peptides**
Download from https://aps.unmc.edu/AP/main.php (Negative dataset tab).

Set `data.amp_neg_path` in `config.yaml` to your negatives CSV path.

### CPP dataset (provided)
Your `Cell_Penetrating_Peptides_List_CellPPD_*.csv` file already contains
both CPP and Non-CPP rows. No separate negative file needed.
Set `data.cpp_path` in `config.yaml`.

---

## Configuration

Edit `config.yaml` before running. Key settings:

```yaml
data:
  amp_pos_path: /absolute/path/to/amp_positives.xlsx
  amp_neg_path: /absolute/path/to/amp_negatives.csv
  cpp_path:     /absolute/path/to/cpp_dataset.csv

features:
  use_esm: false          # set true if fair-esm is installed

generator:
  backend: torch          # torch | numpy
  epochs: 150
  kl_warmup_epochs: 60    # KL annealing warmup

filters:
  min_net_charge: 1.0
  min_amp_score: 0.50
  min_cpp_score: 0.50
```

---

## Running

```bash
cd amp_cpp_pipeline

# Basic run
python main.py --config config.yaml

# Override config values at CLI
python main.py --config config.yaml --set generator.epochs=200 --set filters.min_amp_score=0.60

# Specify absolute data paths at CLI (no need to edit config.yaml)
python main.py --config config.yaml \
  --set data.amp_pos_path=/data/amp.xlsx \
  --set data.amp_neg_path=/data/negatives.csv \
  --set data.cpp_path=/data/cpp.csv
```

Outputs are written to `outputs/<timestamp>/`:
```
outputs/20250101_120000/
├── config_used.yaml        # exact config used (reproducibility)
├── candidates.csv          # ranked final peptide candidates
├── amp_metrics.json        # AMP classifier metrics (CV + val + test)
├── cpp_metrics.json        # CPP classifier metrics (CV + val + test)
├── score_comparison.csv    # generated vs. training vs. random scores
├── novelty_report.json     # edit-distance novelty statistics
└── report.txt              # human-readable summary
```

---

## What was fixed and why

| Issue | Original | Fixed |
|---|---|---|
| Synthetic negatives | Shuffled positive sequences | Requires real validated negatives |
| Data leakage | TF-IDF/SVD fit before CV | All transformers inside sklearn Pipeline |
| Calibration | On training data | On held-out validation set only |
| No test set | Only CV | Stratified train/val/test split |
| CVAE KL annealing | Missing | Linear β schedule over warmup epochs |
| Padding masking | Missing | Loss masked to real positions only |
| Sequence truncation | `.rstrip('A')` | Explicit length conditioning |
| Instability index | Charged-ratio heuristic | Guruprasad 1990 DIWV formula |
| Net charge | K+R+H count | Full pKa Henderson-Hasselbalch |
| Mutation strategy | Uniform random | Softmax-weighted seed selection |
| Diversity control | Physicochemical only | Edit distance + clustering |
| Evaluation | None | Score distributions + novelty metrics |
| Hardcoded paths | Yes | config.yaml + CLI overrides |
| Logging | print() statements | Python logging with file handler |
| Reproducibility | Partial | Central seed, versioned outputs |
| Aggregation filter | Missing | Max consecutive hydrophobic run |
| Amphipathicity filter | Missing | Eisenberg hydrophobic moment |

---

## Extending the pipeline

### Adding a toxicity filter
1. Install a toxicity prediction tool (e.g., ToxinPred or CAMPR3 API)
2. Add a `predict_toxicity(seqs) -> np.ndarray` function
3. Add threshold to `FilterConfig` in `filters/biological.py`
4. Call it in `_passes_filters()`

### Switching to autoregressive generation
Replace `MLPDecoder` in `models/cvae_torch.py` with an LSTM decoder and
implement teacher forcing in the training loop. The `generate_sequences()`
function would then run beam search or greedy sampling token-by-token.

### Using a different classifier
The `train_predictor()` function in `models/classifier.py` accepts any
sklearn-compatible estimator. Replace `_make_gbm()` / `_make_mlp()` with
your preferred model and it will automatically be wrapped in the leak-free
Pipeline.
