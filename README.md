# 🧬 ML-Guided Generation of Antimicrobial Peptides with Cell-Penetrating Activity

> A fully self-contained, NumPy/scikit-learn pipeline that learns the sequence grammar of antimicrobial peptides (AMPs), predicts cell-penetrating peptide (CPP) activity, and generates novel dual-function candidates — no deep learning framework required.

---

## Table of Contents

- [Overview](#overview)
- [Background](#background)
- [Pipeline Architecture](#pipeline-architecture)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
  - [Phase 1 — Data Preparation](#phase-1--data-preparation)
  - [Phase 2 — CPP Predictor](#phase-2--cpp-predictor)
  - [Phase 3 — AMP Predictor](#phase-3--amp-predictor)
  - [Phase 4 — CVAE Generator](#phase-4--cvae-generator)
  - [Phase 5 — Multi-Objective Optimisation](#phase-5--multi-objective-optimisation)
  - [Phase 6 — Biological Hard Filters](#phase-6--biological-hard-filters)
  - [Phase 7 — Secondary Validation](#phase-7--secondary-validation)
  - [Phase 8 — Diversity Control](#phase-8--diversity-control)
  - [Phase 9 — Final Output](#phase-9--final-output)
- [Physicochemical Descriptors](#physicochemical-descriptors)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
- [Output Files](#output-files)
- [Limitations & Future Work](#limitations--future-work)
- [Citation](#citation)

---

## Overview

This project addresses a key challenge in peptide drug discovery: finding sequences that are simultaneously **antimicrobial** and **cell-penetrating**. AMPs kill or inhibit pathogens, while CPPs carry cargo across cell membranes — combining both functions in a single short peptide is rare and experimentally expensive to discover.

This pipeline automates that search using:

- Supervised classifiers trained on curated AMP and CPP datasets
- A Conditional Variational Autoencoder (CVAE) to learn and sample the AMP sequence distribution
- Multi-objective scoring and iterative mutation to push candidates toward dual-function space
- Physicochemical and biological hard filters to ensure experimental plausibility
- Diversity-enforced final shortlisting for cost-effective experimental follow-up

---

## Background

**Antimicrobial peptides (AMPs)** are short peptides (typically 5–60 amino acids) produced by virtually all living organisms as part of innate immunity. They disrupt microbial membranes and are considered promising candidates against drug-resistant pathogens.

**Cell-penetrating peptides (CPPs)** are peptides capable of crossing lipid bilayers and entering cells, either by endocytosis or direct translocation. They are widely used as drug delivery vectors.

A peptide that combines both properties could act as a self-delivering antimicrobial — entering infected cells and killing intracellular pathogens — which is a highly sought-after property in next-generation antibiotic development.

---

## Pipeline Architecture

```
  ┌──────────────────────────────────────────────────────────────┐
  │                     INPUT DATASETS                           │
  │    AMP dataset (1,527 sequences)  +  CPP dataset (704 seqs)  │
  └────────────────────────┬─────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   Phase 1: Data Prep    │  Clean · Deduplicate · Descriptors
              └────────────┬────────────┘
               ┌───────────┴───────────┐
               ▼                       ▼
  ┌────────────────────┐   ┌────────────────────┐
  │  Phase 2: CPP      │   │  Phase 3: AMP      │
  │  Predictor         │   │  Predictor         │
  │  GBM + MLP         │   │  GBM + MLP + RF    │
  │  AUC ≈ 0.83        │   │  AUC ≈ 0.91        │
  └────────────┬───────┘   └────────────┬───────┘
               │                        │
               │           ┌────────────▼───────────┐
               │           │   Phase 4: CVAE        │
               │           │   Train on AMP seqs    │
               │           │   NumPy VAE, latent=32 │
               │           └────────────┬───────────┘
               │                        │
               └───────────┬────────────┘
                           ▼
          ┌────────────────────────────────┐
          │   Phase 5: Multi-Obj. Optimise │
          │   Generate 2000 → Score →      │
          │   Mutate Top-200 × 3 rounds    │
          └────────────────┬───────────────┘
                           │
          ┌────────────────▼───────────────┐
          │   Phase 6: Biological Filters  │
          │   Length · Charge · GRAVY ·    │
          │   Instability · Score cutoffs  │
          └────────────────┬───────────────┘
                           │
          ┌────────────────▼───────────────┐
          │   Phase 7: Secondary Validation│
          │   CPP predictor as tie-breaker │
          └────────────────┬───────────────┘
                           │
          ┌────────────────▼───────────────┐
          │   Phase 8: Diversity Control   │
          │   KMeans clustering · top-k    │
          │   per cluster                  │
          └────────────────┬───────────────┘
                           │
          ┌────────────────▼───────────────┐
          │   Phase 9: Final Output        │
          │   Ranked CSV + Report TXT      │
          └────────────────────────────────┘
```

---

## Dataset Description

| Dataset | Source | Raw Sequences | After Cleaning |
|---------|--------|:-------------:|:--------------:|
| AMP | ADP6 database (disulfide-bond-containing AMPs) | 1,655 | 1,527 |
| CPP | CellPPD (Raghavendra et al.) | 708 | 704 |

Both datasets were pre-annotated with physicochemical descriptors. The pipeline recomputes all descriptors from scratch for any newly generated sequence.

The CPP dataset label distribution: **633 CPP-positive** / **71 CPP-negative**. The imbalance is addressed by augmenting the negative class with shuffled CPP sequences.

---

## Methodology

### Phase 1 — Data Preparation

- Strip non-standard characters, uppercase all sequences
- Remove sequences containing non-canonical amino acids (anything outside `ACDEFGHIKLMNPQRSTVWY`)
- Remove duplicate sequences
- Enforce length range: **5–60 amino acids**
- Compute all 9 physicochemical descriptors from first principles for every sequence

### Phase 2 — CPP Predictor

**Representation:** k-mer (trigram) TF-IDF → 50-component TruncatedSVD, concatenated with 9 physicochemical descriptors (59 features total).

**Negative class construction:**
- Original 71 Non-CPP sequences retained
- 562 additional synthetic negatives generated by randomly shuffling positive CPP sequences (preserves amino acid composition, destroys positional patterns)

**Model:** Calibrated ensemble of:
- `GradientBoostingClassifier` (200 estimators, lr=0.08, max_depth=4)
- `MLPClassifier` (128→64, ReLU, early stopping)

Both models calibrated with isotonic regression via `CalibratedClassifierCV`.

**Evaluation:** 5-fold stratified cross-validation → **ROC-AUC ≈ 0.83**

Final CPP score = average of both calibrated probability outputs.

### Phase 3 — AMP Predictor

Same representation strategy with a slightly wider SVD (64 components → 73 features total).

**Negative class:** All 1,527 AMP sequences shuffled to generate composition-matched negatives.

**Model:** Three-model calibrated ensemble:
- `GradientBoostingClassifier` (300 estimators)
- `MLPClassifier` (128→64→32, ReLU)
- `RandomForestClassifier` (200 trees)

**Evaluation:** 5-fold stratified cross-validation → **ROC-AUC ≈ 0.91**

Final AMP score = mean of three calibrated probability outputs.

### Phase 4 — CVAE Generator

A Variational Autoencoder implemented in **pure NumPy** (no PyTorch or TensorFlow).

**Architecture:**

```
Encoder:  x (900-dim one-hot, max_len=45 × 20 AAs)
           → Dense(128, ReLU)
           → μ (32-dim),  log σ² (32-dim)

Decoder:  z ~ N(μ, σ²)   [reparameterisation trick]
           → Dense(128, ReLU)
           → Dense(900)
           → Per-position Softmax (45 positions × 20 AAs)
```

**Training details:**
- Loss: Reconstruction (cross-entropy) + β × KL divergence (β = 0.4)
- Optimiser: Adam (lr = 5×10⁻⁴, β₁=0.9, β₂=0.999)
- Gradient clipping: ±5.0
- Batch size: 64, Epochs: 100
- Weights initialised with He initialisation

**Generation:** Sample z ~ N(0, I), decode with temperature scaling (T=1.1 for diversity), sample per-position categorical distribution.

### Phase 5 — Multi-Objective Optimisation

1. Generate **2,000 candidate sequences** from the CVAE
2. Score every candidate with both predictors
3. Rank by **geometric mean**: `√(AMP_score × CPP_score)`
4. Select top-200 candidates as seed sequences
5. Run **3 rounds of iterative point mutation** (1–2 mutations per sequence per round), scoring all mutants
6. Pool all scored sequences → deduplicate → sort by combined score

Geometric mean is used instead of arithmetic mean to require *both* scores to be high simultaneously (a sequence scoring 0.95/0.05 is penalised more than 0.5/0.5).

### Phase 6 — Biological Hard Filters

| Filter | Criterion | Rationale |
|--------|-----------|-----------|
| Length | 6–55 AA | Practical synthesis range |
| Net charge (pH 7) | ≥ 0 | CPPs and AMPs are typically cationic |
| GRAVY score | ≤ 3.0 | Prevent insoluble / membrane-trapped peptides |
| Instability index | ≤ 80 | Exclude highly unstable sequences |
| AMP score | ≥ 0.40 | Minimum antimicrobial credibility |
| CPP score | ≥ 0.40 | Minimum cell-penetrating credibility |

Filters are applied sequentially. If no sequences pass, constraints are relaxed once automatically.

### Phase 7 — Secondary Validation

The CPP predictor is re-applied to all passing sequences as an independent sanity check. The final CPP score for each candidate is the average of its Phase 5 score and this re-score, reducing prediction noise.

### Phase 8 — Diversity Control

- Compute physicochemical feature matrix for all passing sequences
- Apply **KMeans clustering** (up to 20 clusters)
- Retain the **top-4 highest-scoring sequences per cluster**
- This prevents the final list from being dominated by near-identical mutants of a single sequence

### Phase 9 — Final Output

Ranked shortlist exported with all annotations:

| Column | Description |
|--------|-------------|
| `rank` | Final combined-score rank |
| `sequence` | Peptide sequence (single-letter AA code) |
| `amp_score` | Calibrated AMP probability [0–1] |
| `cpp_score` | Calibrated CPP probability [0–1] |
| `combined_score` | √(amp × cpp) |
| `cluster` | Diversity cluster ID |
| `length` | Sequence length (AA) |
| `molecular_weight` | Da |
| `pI` | Isoelectric point |
| `instability_index` | Guruprasad instability |
| `GRAVY` | Grand average of hydropathicity |
| `aromaticity` | Fraction of F, W, Y residues |
| `aliphatic_index` | Relative volume of aliphatic side chains |
| `boman_index` | Protein–protein interaction potential |
| `ww_hydrophobicity` | Wimley–White interfacial hydrophobicity |

---

## Physicochemical Descriptors

All nine descriptors are computed from scratch using established bioinformatics scales:

| Descriptor | Scale / Method |
|------------|----------------|
| Molecular weight | Residue MW table minus water per peptide bond |
| Isoelectric point (pI) | Binary search over Henderson-Hasselbalch equation |
| Instability index | Heuristic based on charged residue composition |
| GRAVY | Kyte & Doolittle (1982) hydrophobicity scale |
| Aromaticity | Fraction of F, W, Y |
| Aliphatic index | Ikai (1980): A + 2.9V + 3.9(I+L) |
| Boman index | Boman (2003) interaction energy scale |
| Wimley-White hydrophobicity | Wimley & White (1996) interfacial scale |
| Net charge | Henderson-Hasselbalch at pH 7.0 |

---

## Results

Pipeline run on the full datasets produced **25 diverse dual-function candidates** distributed across **12 physicochemical clusters**.

| Metric | Value |
|--------|-------|
| Total candidates generated | 3,169 |
| Passed biological filters | 25 |
| Diversity clusters | 12 |
| Mean AMP score (final) | 0.490 |
| Mean CPP score (final) | 0.425 |
| Mean combined score (final) | 0.455 |
| Top combined score | **0.549** |

**Top 5 candidates:**

| Rank | Sequence | AMP | CPP | Combined | pI | Charge |
|------|----------|:---:|:---:|:--------:|:--:|:------:|
| 1 | `GGLKKVVGTTKAKEFTVGFCVFSCGIVQQGKRLGCRTRGLKKVHQ` | 0.702 | 0.429 | 0.549 | 10.92 | +9.1 |
| 2 | `GGLKKVVGTTKSKGFTVGFCVFSFGISQQGVRLGCRTRGLKKVHQ` | 0.557 | 0.432 | 0.490 | 11.68 | +9.1 |
| 3 | `QWCAPVSGENCSRYGGLFLSSTHTADLQRCAGIFGKNWVPHAPCY` | 0.597 | 0.402 | 0.490 | 7.71 | +1.2 |
| 4 | `LFDALGAGTCVAKSVHCAITYFTYSCTKTQIYSPTCSVWCRFQLM` | 0.583 | 0.405 | 0.485 | 7.99 | +2.1 |
| 5 | `GLFGCSQSRIVNGIEKLLSKTTTPCGEIFVTKIWVQGIHCHHDEC` | 0.557 | 0.421 | 0.484 | 7.02 | +0.3 |

---

## Requirements

```
python >= 3.9
numpy
pandas
scipy
scikit-learn
openpyxl       # for reading .xlsx input
```

Install all dependencies:

```bash
pip install numpy pandas scipy scikit-learn openpyxl
```

No GPU required. No internet connection required at runtime.

---

## Usage

1. Clone the repository and place your input files in the working directory:

```
amp_cpp_pipeline.py
actual_antimicrobial_peptide_list_ADP6_25feb_-_sequences_containing_more_that_one_C_contains_disulfide_bond.xlsx
Cell_Penetrating_Peptides_List_CellPPD_Raghavendra.csv
```

2. Update the file paths in `phase1_load_and_clean()` if needed (lines ~200–220).

3. Run the pipeline:

```bash
python amp_cpp_pipeline.py
```

4. Outputs are written to `/mnt/user-data/outputs/` by default. Change `out_dir` in `phase9_final_output()` to redirect.

**Expected runtime:** ~10–20 minutes on a standard laptop CPU (dominated by GBM cross-validation and CVAE training).

---

## Output Files

| File | Description |
|------|-------------|
| `amp_cpp_candidates.csv` | Full ranked candidate table with all scores and descriptors |
| `amp_cpp_report.txt` | Human-readable summary report with top-10 table |

---

## Limitations & Future Work

**Current limitations:**

- The CVAE is implemented in pure NumPy. Without a proper deep learning framework, KL divergence annealing and posterior collapse prevention are harder to tune. With PyTorch or JAX, a more stable training loop with KL warm-up would significantly improve generation quality.
- The CPP predictor is trained on a relatively small and imbalanced dataset (633 positives, 71 natural negatives). A larger, more balanced CPP dataset would improve predictor confidence.
- Sequence representation relies on trigram k-mers + physicochemical features. Protein language model embeddings (e.g. ESM-2) would capture evolutionary and structural context that k-mers cannot.
- All generated sequences are 45 AA (padded to max training length). A length-conditional generation strategy would produce more diverse length distributions.

**Planned upgrades:**

- [ ] Replace NumPy VAE with PyTorch CVAE with KL annealing and teacher forcing
- [ ] Integrate ESM-2 or ProtT5 embeddings as the primary sequence representation
- [ ] Add REINFORCE or PPO-based reinforcement learning on top of the generator to directly optimise the dual-function objective
- [ ] Add secondary structure and amphipathicity scoring as additional biological filters
- [ ] Benchmark generated candidates against CPPpredictor2, AMPscanner, and CAMPR3 external tools


---

*Built with NumPy, SciPy, and scikit-learn. No proprietary dependencies.*
