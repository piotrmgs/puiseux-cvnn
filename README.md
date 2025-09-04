---

<details>
<summary><strong>Table of contents</strong></summary>

- [About this repo](#about-this-repo)
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Datasets ⚠️](#datasets)
  - [MIT-BIH Arrhythmia (PhysioNet)](#1-mitbih-arrhythmia-physionet)
  - [RadioML 2016.10A (`RML2016.10a_dict.pkl`)](#2-radioml-201610a-rml201610a_dictpkl)
- [Installation](#installation)
- [Quick Start (synthetic benchmark)](#quick-start-synthetic-benchmark)
- [Usage](#usage)
  - [1. Synthetic uncertain points](#1-synthetic-uncertain-points)
  - [2. MIT-BIH preprocessing](#2-mit-bih-preprocessing)
  - [3. MIT‑BIH uncertain points](#3-mit-bih-uncertain-points)
  - [4. RadioML 2016.10A — uncertain points](#4-radioml-201610a--uncertain-points)
  - [5. Puiseux Test](#5-puiseux-test)
  - [6. Local Analysis test](#6-local-analysis-test)
  - [7. Post-Processing Synthetic Data](#7-post-processing-synthetic-data)
  - [8. Post-Processing Real Data (MIT-BIH)](#8-post-processing-real-data-mit-bih)
  - [9. Post-Processing Radio Data (RadioML 2016.10A)](#9-post-processing-radio-data-radioml-201610a)
  - [10. Newton–Puiseux Evidence & Triage (MIT-BIH)](#10-newtonpuiseux-evidence--triage-mit-bih)
  - [11. Newton–Puiseux Evidence & Triage (RadioML 2016.10A)](#11-newtonpuiseux-evidence--triage-radioml-201610a)

- [License](#license)
- [Contact](#contact)

</details>


## About this repo
This repository contains the complete, reproducible codebase that accompanies the paper *“Newton-Puiseux Analysis for Interpretability and Calibration of Complex-Valued Neural Networks”*. 

[![arXiv](https://img.shields.io/badge/arXiv-2504.19176-b31b1b.svg)](https://arxiv.org/abs/2504.19176)


https://arxiv.org/pdf/2504.19176

It implements our end-to-end pipeline – from data preprocessing through CVNN training to Newton–Puiseux-based local analysis – across three settings: (1) a controlled synthetic benchmark, (2) the MIT-BIH arrhythmia corpus, and (3) the RadioML 2016.10A wireless modulation dataset.

---


# Newton‑Puiseux Analysis for Interpretability and Calibration of CVNNs

## Overview
This repository provides the full codebase and reproducible scripts for our Newton–Puiseux framework, which enhances interpretability and calibration of complex-valued neural networks (CVNNs). We demonstrate the approach on a controlled synthetic dataset and two real-data settings: the MIT-BIH arrhythmia corpus and the RadioML 2016.10A wireless modulation dataset.

---

## Project Structure
```
├── mit‑bih/                      # Raw MIT‑BIH dataset
├── radio-data/                   # External RadioML files
├── mit_bih_pre/                  # Preprocessing scripts for MIT‑BIH
│   └── pre_pro.py                # Signal filtering and feature extraction
├── src/                          # Core library modules
│   ├── post_processing.py        # Post‑processing (common)
│   ├── find_up_synthetic.py      # Uncertainty mining on synthetic data
│   ├── find_up_real.py           # Uncertainty mining on MIT‑BIH data
│   ├── find_up_radio.py          # Uncertainty mining on RadioML data
│   ├── local_analysis.py         # Local surrogate + Puiseux wrapper
│   └── puiseux.py                # Newton‑Puiseux solver
├── up_synth/                     # Synthetic dataset training and evaluation
│   └── up_synthetic.py
├── local_analysis_synth_test/    # Tests for local analysis (synthetic)
│   └── local_analysis_synth_test.py
├── puiseux_test/                 # Tests for Puiseux solver
│   └── puiseux_test.py
├── post_processing_synth/        # Post‑processing for synthetic data
│   └── post_processing_synth.py
├── up_real/                      # MIT‑BIH CVNN training and evaluation
│   └── up_real.py
├── up_radio/                     # RadioML 2016.10A CVNN training and evaluation
│   └── up_radio.py
├── post_processing_real/         # Post‑processing for MIT‑BIH data
│   └── post_processing_real.py
├── post_processing_radio/        # Post‑processing for RadioML 2016.10A
│   └── post_processing_radio.py
├── NP-analysis_real/             # Newton–Puiseux evidence & triage (MIT-BIH)
│   └── NP-analysis_real.py
├── NP-analysis_radio/            # Newton–Puiseux evidence & triage (RadioML 2016.10A)
│   └── NP-analysis_radio.py
└── README.md                     # This file
```

<!-- Fix anchor: Datasets -->
<a id="datasets"></a>
## Datasets ⚠️


### 1) MIT‑BIH Arrhythmia (PhysioNet)

**MIT-BIH Arrhythmia Database is *not* stored in this repository.**

**Manual download**  
   * Go to <https://physionet.org/content/mitdb/1.0.0/>  
   * Download all files and unzip them into `mit-bih/`.

**Licence notice:**  
MIT-BIH data are released under the PhysioNet open-access license.  
By downloading the files you agree to its terms.

### 2) RadioML 2016.10A (`RML2016.10a_dict.pkl`)
**Dataset is *not* stored in this repository.**

**Manual download**  
  * Go to the DeepSig datasets page (RADIOML 2016.10A) and download the archive `RML2016.10a.tar.bz2` (it contains `RML2016.10a_dict.pkl`) or the `.pkl` file directly.  
  * After download, extract (if needed) and **place the file** into `radio-data/`. Expected path: the code will look for: `radio-data/RML2016.10a_dict.pkl`.

**License notice (RadioML 2016.10A):**  
Released by DeepSig under **CC BY‑NC‑SA 4.0** (Attribution‑NonCommercial‑ShareAlike).  
By downloading/using the dataset you agree to these terms. See the dataset page and license for details.

-----

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/piotrmgs/puiseux-cvnn.git
   cd puiseux-cvnn
   ```   
2. Create a virtual environment and install dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate puiseux-cvnn
   ```

------
## Quick Start (synthetic benchmark)
```bash
# 1  Run the complete synthetic pipeline (≃ 30 s on CPU)
python -m up_synth.up_synthetic

# 2  Generate Newton-Puiseux analysis for uncertain points
python -m post_processing_synth.post_processing_synth
```

----------------------------------------------------------
## Usage
### 1. Synthetic uncertain points

To generate synthetic data, train a CVNN on it, and identify uncertain points, run:

```bash
python -m up_synthetic.up_synthetic
```

This script performs the following steps:

1. **Data generation**: Creates a controlled synthetic dataset in `$\mathbb{C}^2$` (default: 200 samples per class).
2. **Train/Test split**: Splits data 80/20 into training and test sets.
3. **Model construction**: Builds a `SimpleComplexNet` with 2 complex input features, 16 hidden units, and 2 output classes.
4. **Training**: Optimizes the model for 50 epochs using Adam (learning rate $10^{-3}$).
5. **Evaluation**: Computes and prints test accuracy.
6. **Uncertainty mining**: Flags test points where the maximum softmax probability falls below the threshold (default: 0.1).
7. **CSV export**: Saves uncertain samples to `up_synthetic/uncertain_synthetic.csv`, including feature vectors, true labels, and model probabilities.
8. **Model checkpoint**: Saves trained model parameters to `up_synthetic/model_parameters.pt`.
9. **Visualization**: Generates and saves a PCA scatter plot showing all test samples colored by confidence, with uncertain points highlighted (e.g., `up_synthetic/uncertainty_plot.png`).

**Outputs:**

- `up_synthetic/uncertain_synthetic.csv` — CSV file listing indices, complex inputs, labels, and softmax probabilities of uncertain points.
- `up_synthetic/model_parameters.pt` — PyTorch state dict of the trained network.
- `up_synthetic/uncertainty_plot.png` — PCA visualization of test-set confidence.


### 2. MIT-BIH preprocessing

#### Data Preprocessing (`pre_pro.py`)

To load, filter, segment, and visualize ECG recordings from the MIT-BIH Arrhythmia Database, run:

```bash
python -m mit_bih_pre.pre_pro
```

This script performs the following steps:

- **Record loading**: Reads raw signal and annotation files (`.dat`, `.hea`, `.atr`) from `mit-bih/` for records 100–107.
- **Bandpass filtering**: Applies a zero-phase Butterworth filter (0.5–40 Hz, 2nd order) to each channel to remove baseline wander and high‑frequency noise.
- **Segmentation**: Extracts 128‑sample windows around each R peak (50 samples before peak) for beats labeled 'N' (normal) and 'V' (PVC).
- **Hilbert transform**: Computes analytic signals per channel and concatenates real and imaginary parts into feature vectors.
- **Normalization**: Standardizes all features using `StandardScaler`.
- **Train/Test split**: Splits the data 80/20 (stratified by label).
- **Tensor conversion**: Converts the full normalized dataset to PyTorch tensors (optional for downstream ML pipelines).
- **Visualizations**: Generates and saves plots in `mit_bih_pre/`:
  - `class_distribution.png` — histogram of 'N' vs. 'V' samples
  - `signal_comparison_<record>_class_<N|V>.png` — raw vs. filtered signal with R‑peak marker
  - `hilbert_transform_<record>_class_<N|V>.png` — filtered signal and Hilbert envelope
  - `spectrogram_<record>_class_<N|V>.png` — time–frequency spectrogram
  - `correlation_matrix.png` — feature correlation heatmap (first 50 features)
  - `tsne_visualization.png` — 2D t‑SNE embedding of the feature space

**Outputs:**

- All figures listed above saved to `mit_bih_pre/`
- In-memory variables `X`, `y`, `X_norm`, `X_train`, `X_test`, `y_train`, `y_test` (accessible within the script)


### 3. MIT-BIH uncertain points

To train a CVNN with cross‑patient K‑Fold CV, calibrate it, and automatically flag uncertain predictions on the held‑out **TEST** split, run:

```bash
# GPU if available
python -m up_real.up_real

# Force CPU (optional)
python -m up_real.up_real --cpu
```

**What this does**

1. **Cross‑patient K‑Fold training** (`--folds`, default 10): splits patient records so no subject appears in both TRAIN and TEST in a fold; trains `SimpleComplexNet` on complex features (`complex_stats`).
2. **Per‑fold calibration** (controlled by `--calibration`: `temperature` [default], `isotonic`, or `none`): applies calibration on the **validation** split of each fold and evaluates on the fold’s TEST.
   - If `temperature`, saves `T_calib_fold{fold}.pt`.
3. **Multi‑calibration evaluation** (optional, via `--calibs`): runs a panel of calibrators on the same fold logits (e.g., `temperature,isotonic,platt,beta,vector,none`) and writes comparative metrics per fold.
4. **Global reliability curves**: writes baseline **RAW** and **calibrated** reliability diagrams aggregated across folds.
5. **Automatic threshold selection on VALIDATION (full‑model stage)**:
   - Builds a **(τ, δ)** grid where **τ** is the minimum confidence (max probability) to accept a prediction and **δ** is the minimum **margin** between the top‑2 class probabilities.
   - If `--sensitivity` (default **on**), saves the grid to `sens_grid.csv` and heatmaps `sens_full_*`; also computes a "knee" score proxy.
   - Selects `(τ*, δ*)` using an **exact review budget** (`--review_budget`, default 10 samples) via `select_thresholds_budget_count`. The chosen pair and stats are written to `sens_full.csv`.
6. **Uncertain‑point detection on TEST**: flags samples with low confidence and/or small margin (i.e., `max_prob < τ*` and/or `margin < δ*`) and saves them to `uncertain_full.csv`.
7. **Artifacts**: per‑fold training curves, confusion matrix + ROC, overall training history across folds, reliability diagrams (RAW vs calibrated), and uncertainty histograms (confidence and margin).

**Outputs (files)**

- **Logs & meta**
  - `run.log` — full run log (folds, timings, resources).
  - `run_meta.json` — versions and device info for reproducibility.
- **Per‑fold artifacts** (filenames include the fold index):
  - training curves (e.g., `training_history_fold{fold}.png`),
  - confusion matrix & ROC (e.g., `confusion_fold{fold}.png`, `roc_fold{fold}.png`),
  - `scaler_fold{fold}.pkl`,
  - (if `--calibration temperature`) `T_calib_fold{fold}.pt`.
- **Cross‑fold (CV) metrics**
  - `cv_metrics_per_fold.csv` — RAW vs calibrated (ECE, NLL, Brier, Acc, AUC) per fold.
  - `cv_metrics_summary.csv` — mean ±95% CI for ECE/NLL/Brier across folds.
  - `predictions_all_folds.csv` — row‑wise predictions with calibrated probs and the top‑2 **margin**.
- **Multi‑calibration panel** (if `--calibs` is non‑empty; by default several methods are evaluated):
  - `cv_metrics_per_fold_multi.csv` — per‑fold metrics for each method in `--calibs`.
  - `cv_metrics_summary_multi.csv` — cross‑fold means and 95% CIs by method.
- **Global visualizations**
  - `calibration_curve_RAW.png` and `calibration_curve_{TS|ISO|CAL}.png` — reliability diagrams pre/post calibration.
  - `uncertainty_histogram.png` — distribution of `max_prob` on TEST with τ* marker.
  - `uncertainty_margin_hist.png` — distribution of the top‑2 **margin** with δ* marker.
  - `complex_pca_scatter.png` — PCA of complex features over all records.
- **Full‑model stage**
  - `best_model_full.pt` — best weights after full retraining.
  - `scaler_full.pkl` — StandardScaler for the full‑model pipeline.
  - (if `--calibration temperature`) `T_calib.pt` — temperature for the full model.
  - `sens_grid.csv`, `sens_full.csv`, and sensitivity heatmaps `sens_full_*`.
- **Uncertain points**
  - `uncertain_full.csv` — flagged TEST samples with columns:
    - `index` (row id in TEST), `X` (feature vector), `true_label`, `p1`, `p2`.

**Key CLI switches**

- `--calibration {temperature,isotonic,none}` — per‑fold and full‑model calibration (**default: temperature**).
- `--calibs` — comma‑separated list of calibration methods to **evaluate** in addition to the operational one (default: `temperature,isotonic,platt,beta,vector,none`).
- `--sensitivity` / `--no-sensitivity` — enable/disable τ/δ grid & heatmaps (**on by default**).
- `--review_budget <int>` — **exact count** of validation samples allowed for manual review; used to pick `(τ*, δ*)` (**default: 10**).
- `--select_mode {capture,budget,risk,knee}` + `--capture_target`, `--max_abstain`, `--target_risk` — configure how the grid is scored/summarized (the **final selection** still uses `--review_budget`).
- `--epochs`, `--lr`, `--batch_size`, `--folds`, `--cpu`, `--seed` — standard training controls.

**Example**

```bash
# 10-fold CV on GPU (if available), temperature scaling, exact review budget of 20 samples:
python -m up_real.up_real \
  --data_folder mit-bih \
  --output_folder up_real \
  --epochs 10 --lr 1e-3 --batch_size 128 --folds 10 \
  --calibration temperature \
  --review_budget 20 \
  --sensitivity
```

### 4. RadioML 2016.10A — uncertain points

This section mirrors the MIT‑BIH pipeline but runs on **RadioML 2016.10A**.  
Make sure you have downloaded the dataset as described in **Datasets ⚠️** and that:
```
radio-data/RML2016.10a_dict.pkl
```
is present.

**Run**

```bash
# GPU if available
python -m up_radio.up_radio

# Force CPU (optional)
python -m up_radio.up_radio --cpu
```

**What this does**

1. **Subset selection (RadioML 2016.10A)**: filters by modulation classes (`--mods`, default: `BPSK QPSK`) and SNR range (`--snr_low/--snr_high`, default: 5…15 dB).
2. **Complex‑aware features**: converts raw IQ windows into compact **STFT‑based statistics** via `prepare_complex_input(method='stft_stats')`.
3. **Stratified K‑Fold on samples** (`--folds`, default 10): trains `SimpleComplexNet` per fold with a train/val split inside each fold; standardization fitted on TRAIN only and saved per fold.
4. **Probability calibration on VALIDATION** (operational method chosen by `--calibration`; default here: **platt** for binary tasks). The same calibrator is applied on TEST for metrics and plots.
5. **Multi‑calibration sweep** (optional; `--calibs`): evaluates a panel (`temperature,isotonic,platt,beta,vector,none`) on the same TEST logits and logs per‑method metrics.
6. **Cross‑fold aggregation**: reliability diagrams for **RAW** vs **CAL** (the chosen method), learning curves across folds, confusion matrices & ROC curves per fold.
7. **Full‑model stage**: retrains on a fresh Train/Val/Test split; performs **(τ, δ)** sensitivity analysis on VALIDATION (if `--sensitivity`), selects **(τ\*, δ\*)** with an **exact review budget** (`--review_budget`), then plots TEST histograms for `p_max` (with τ\*) and top‑2 **margin** (with δ\*).
8. **Uncertainty export**: flags TEST samples with `(p_max < τ*)` OR `(margin < δ*)` and writes them to `uncertain_full.csv`.

**Outputs**

- **Logs & metadata**
  - `run.log` — run log (fold splits, timings, memory).
  - `run_meta.json` — Python/NumPy/PyTorch versions and device.
- **Per‑fold artifacts** (file names are suffixed with fold index):
  - Training curves from `save_plots` (e.g., `training_history_fold{fold}.png`),
  - Confusion matrix & ROC: `confusion_fold{fold}.png`, `roc_fold{fold}.png`,
  - `scaler_fold{fold}.pkl` — StandardScaler fitted on TRAIN,
  - If `--calibration temperature`: `T_calib_fold{fold}.pt`.
- **Cross‑fold (CV) metrics**
  - `cv_metrics_per_fold.csv` — per‑fold **RAW vs CAL** metrics: ECE, NLL, Brier, Acc, AUC (CAL columns are tagged with the chosen method, e.g., `CAL_PLATT`).
  - `cv_metrics_summary.csv` — mean and 95% CI half‑widths for ECE/NLL/Brier across folds.
  - `predictions_all_folds.csv` — per‑sample calibrated probabilities with **pmax** and top‑2 **margin**; column keys are suffixed with the CAL method (e.g., `p1_CAL_PLATT`, `margin_CAL_PLATT`).
- **Multi‑calibration panel** (if `--calibs` non‑empty)
  - `cv_metrics_per_fold_multi.csv` — per‑fold metrics for each method in `--calibs`.
  - `cv_metrics_summary_multi.csv` — cross‑fold means and 95% CIs by method.
- **Global visualizations**
  - `calibration_curve_RAW.png` and `calibration_curve_<CAL>.png` (e.g., `calibration_curve_PLATT.png`),
  - `complex_pca_scatter.png` — PCA on `stft_stats` features,
  - `uncertainty_histogram.png` — TEST distribution of `p_max` with τ\* marker,
  - `uncertainty_margin_hist.png` — TEST distribution of top‑2 margin with δ\* marker.
- **Full‑model artifacts**
  - `best_model_full.pt`, `scaler_full.pkl`,
  - If `--calibration temperature`: `T_calib.pt`,
  - Sensitivity products: `sens_grid.csv`, `sens_full.csv`, and heatmaps `sens_full_*`.
- **Uncertain points**
  - `uncertain_full.csv` with columns: `index`, `X` (scaled feature vector), `true_label`, `p1`, `p2`.

**Key CLI switches**

- **Dataset filters**: `--mods <list>` (e.g., `BPSK QPSK 8PSK`), `--snr_low <int>`, `--snr_high <int>`.
- **Calibration (operational)**: `--calibration {temperature,isotonic,platt,beta,vector,none}` (**default: platt**).
- **Calibration (evaluation panel)**: `--calibs` — comma‑separated methods to compare on TEST (default includes several).
- **Sensitivity & thresholds**: `--sensitivity/--no-sensitivity`, `--review_budget <int>` (exact‑count selection of (τ\*, δ\*); default 10), and scoring knobs: `--select_mode {capture,budget,risk,knee}`, `--capture_target`, `--max_abstain`, `--target_risk`.
- **Training**: `--epochs`, `--lr`, `--batch_size`, `--folds`, `--seed`, `--cpu`.


### 5. Puiseux Test

To compute and save Newton-Puiseux series expansions for a sample polynomial, run:

```bash
python -m puiseux_test.puiseux_test
```

This script performs the following steps:

1. **Polynomial definition**: Specifies a symbolic two-variable polynomial `f(x,y)` in SymPy (default example included, editable in code).
2. **Initial branch computation**: Uses `initial_branches` and polygon utilities to identify starting terms.
3. **Puiseux expansion**: Invokes `puiseux_expansions(f, x, y, max_terms=5)` to compute fractional-power series up to 5 terms.
4. **Console output**: Prints each expansion to `stdout` with separators.
5. **File export**: Writes formatted expansions into `puiseux_test/puiseux_expansions.txt`, including headers, individual expansions, and a footer.

**Outputs:**

- `puiseux_test/puiseux_expansions.txt` — text file containing the list of computed Puiseux series, complete with numeric evaluation of each term.

### 6. Local Analysis test

To perform local polynomial approximation and Newton-Puiseux analysis on uncertain synthetic points, run:

```bash
python -m local_analysis_synth_test.local_analysis_synth_test
```

This script executes the following pipeline:

1. **Model & data loading**: Reads `up_synth/uncertain_synthetic.csv` for uncertain points and loads the pretrained `SimpleComplexNet` weights from `up_synth/model_parameters.pt`.
2. **Point iteration**: Loops over each uncertain point (`xstar`) from the CSV.
3. **Local surrogate fitting**: Constructs a degree-4 polynomial approximation `F̂` of the logit difference around `xstar` using `local_poly_approx_complex`, sampling `n_samples=200` within a cube of radius `delta=0.1` in ℝ⁴.
4. **Puiseux analysis**:
   - **benchmark_local_poly_approx_and_puiseux**: Times and factors the surrogate, returning its symbolic expression and initial Puiseux expansions (around the origin).
   - **puiseux_uncertain_point**: Computes full Puiseux series anchored back at `xstar` with precision 4.
5. **Quality evaluation**: Measures approximation fidelity with `evaluate_poly_approx_quality` (RMSE, MAE, correlation ρ, sign-agreement) on 300 fresh perturbations.
6. **Result export**: For each point, writes a report `local_analysis_synth_test/benchmark_point<idx>.txt` containing:
   - Timing breakdown
   - Final polynomial expression
   - Puiseux expansions at origin and at `xstar`
   - Approximation metrics
7. **Console logging**: Prints progress, loaded point data, timings, metrics, and final Puiseux expansions to `stdout`.

**Outputs:**

- `local_analysis_synth_test/benchmark_point<idx>.txt` — detailed report per uncertain point.



### 7. Post-Processing Synthetic Data

To analyze uncertain synthetic points, generate local explanations, and robustness analyses, run:

```bash
python -m post_processing_synth.post_processing_synth
```

This script performs the following steps:

1. **Model & data loading**: Loads the pretrained `SimpleComplexNet` and its weights (`up_synth/model_parameters.pt`) and reads uncertain points from `up_synth/uncertain_synthetic.csv`.
2. **Local polynomial fitting**: Computes a degree-4 surrogate `F̂` at each uncertain point using `local_poly_approx_complex` (delta=0.05, n_samples=300).
3. **Approximation quality**: Evaluates RMSE, MAE, Pearson correlation, and sign-agreement via `evaluate_poly_approx_quality`.
4. **Puiseux expansions**: Calculates local Puiseux series at each point (`puiseux_uncertain_point`) and interprets them with `interpret_puiseux_expansions`.
5. **Adversarial robustness**:
   - Identifies promising directions via `find_adversarial_directions`.
   - Measures class-flip radii with `test_adversarial_impact` and plots robustness curves (`plot_robustness_curve`).
6. **Local explanations**:
   - Computes LIME explanations (`compute_lime_explanation`).
   - Computes SHAP values (`compute_shap_explanation`).
7. **2D decision contours**: Generates contour plots fixing pairs of dimensions with `plot_local_contour_2d`.
8. **Report generation**: For each point, produces `post_processing_synth/benchmark_point<idx>.txt` with:
   - Approximation metrics
   - Puiseux expressions & interpretations
   - Robustness analysis table
   - LIME & SHAP feature attributions
   - Paths to saved contour and robustness plots

**Outputs:**

- `post_processing_synth/benchmark_point<idx>.txt` — comprehensive local analysis report per point.
- `post_processing_synth/robustness_curves_point<idx>.png` — robustness plots.
- `post_processing_synth/contour_point<idx>_fix_dim=[...].png` — local decision boundary visualizations.

### 8. Post-Processing Real Data (MIT-BIH)

This step consumes the artifacts produced by **Section 3 (MIT‑BIH uncertain points)** and performs local Newton–Puiseux analysis, robustness probes, LIME/SHAP explanations, calibration comparisons with confidence intervals, and sensitivity summaries.  

**Prerequisites**

- You have already run the MIT‑BIH pipeline (Section 3) and produced at least:
  - `up_real/best_model_full.pt`
  - `up_real/scaler_full.pkl`
  - `up_real/uncertain_full.csv`
  - *(optional)* `up_real/T_calib.pt` (if temperature scaling was used)
  - *(optional)* `up_real/sens_grid.csv`, `up_real/cv_metrics_per_fold_multi.csv`
- The MIT‑BIH Arrhythmia data are available locally (see **Datasets ⚠️**).

**Run**

```bash
# From the repo root
python -m post_processing_real.post_processing_real
```

**What this does**

1. **Load model & artifacts**
   - Loads `best_model_full.pt`, optional `T_calib.pt` (temperature), and `scaler_full.pkl`.
   - Loads uncertain anchors from `uncertain_full.csv`.
   - Builds a background set (up to 512 windows) for LIME/SHAP and ensures consistent scaling via the saved scaler.

2. **(τ, δ) Sensitivity summary (if `sens_grid.csv` present)**
   - Parses the grid and writes detailed and summarized reports:
     - `sensitivity_detailed.csv` (full grid values),
     - `sensitivity_summary.txt` and `sensitivity_extra.txt` (aggregates incl. correlations, medians, budget‑constrained bests).

3. **Calibration comparison with 95% CI**
   - If `cv_metrics_per_fold_multi.csv` exists, compiles per‑method summaries:
     - `comparative_table.csv` (mean ± CI for ECE/NLL/Brier/Acc/AUC),
     - `calibration_ci_report.txt` (readable CI table),
     - `calibration_stats_tests.csv` (pairwise Wilcoxon tests; skips if SciPy not installed),
     - `calibration_winrate_vs_none.csv` (fold‑wise win‑rate vs **NONE**).

4. **Per‑anchor local analysis**
   For each uncertain ECG window:
   - **Kink diagnostics**: non‑holomorphicity probe around the anchor (`kink_score`) and fraction of modReLU “kinks”; saves sweeps over `kink_eps`.
   - **Robust local surrogate**: degree‑4 complex polynomial fit with outlier/weighting safeguards; reports condition, rank, kept ratio.
   - **Quality metrics**: RMSE, MAE, Pearson correlation, sign‑agreement against the CVNN.
   - **Newton–Puiseux expansions** + **interpretation** for local branches.
   - **Robustness** along phase‑selected directions (class change & flip radius) with plots.
   - **LIME & SHAP** explanations on consistently scaled C² features.
   - **2D local contours** of the decision boundary for fixed dim pairs (1,3) and (0,2).
   - **Resource benchmark**: timing/memory of Puiseux pipeline vs. gradient saliency.

5. **Aggregate reports & calibration CI table**
   - Builds 5‑fold CI tables for multiple calibrators (`none`, `platt`, `isotonic`, `beta`, `vector`, `temperature`), plus an ablation **none_T0** (uncalibrated at inference) if a temperature file is present.
   - Summarizes kink prevalence and its effect on fit quality and residual statistics.
   - Optional sweep of **ECE sensitivity to branch‑multiplicity mis‑estimation**.

6. **Dominant-ratio summary**
   - Extracts dominant Puiseux coefficients per anchor (max |c₂| and |c₄| across branches) and computes the dominant-ratio proxy `r_dom = sqrt(|c₂|/|c₄|)`.
   - Writes a consolidated CSV for downstream evidence building (see Section 10): `post_processing_real/dominant_ratio_summary.csv`.

**Outputs (saved to `post_processing_real/`)**

- **Logs**
  - `post_processing_real.log` — progress, warnings, file provenance.

- **Sensitivity (τ, δ)**
  - `sensitivity_detailed.csv`, `sensitivity_summary.txt`, `sensitivity_extra.txt`.

- **Calibration comparisons (from CV panel)**
  - `comparative_table.csv`
  - `calibration_ci_report.txt`
  - `calibration_stats_tests.csv` *(if SciPy available)*
  - `calibration_winrate_vs_none.csv`

- **Per‑anchor artifacts** (for each point *i*)
  - `benchmark_point<i>.txt` — comprehensive report (kink, fit, metrics, Puiseux, robustness, LIME/SHAP, resources).
  - `robustness_curves_point<i>.png`
  - `contour_point<i>_fix_dim=[1,3].png`, `contour_point<i>_fix_dim=[0,2].png`
  - `kink_sweep_point<i>.csv`
  - `resource_point<i>.txt`

- **Aggregate summaries**
  - `kink_summary.csv` — fractions of kink/active/inactive across points.
  - `resource_summary.csv` — Puiseux vs. saliency (time/memory).
  - `local_fit_summary.csv` — kept ratio, cond(A), degree, RMSE, sign‑agreement, residual stats.
  - `kink_global_summary.txt` — prevalence and effects on fits/residuals.
  - `dominant_ratio_summary.csv` — per-anchor table with `point`, `c2_max_abs`, `c4_max_abs`, and the dominant-ratio proxy `r_dom = sqrt(|c2|/|c4|)`; consumed by `NP-analysis_real.py` (Section 10).


- **Calibration CI (local 5‑fold)**
  - `calibration_folds_raw.csv`
  - `calibration_ci_table.csv`
  - `calibration_ci_report.txt` *(same name as above if present; latest run overwrites)*

- **Branch multiplicity sensitivity**
  - `branch_multiplicity_sensitivity.csv`
 

### 9. Post-Processing Radio Data (RadioML 2016.10A)

This step mirrors the MIT‑BIH post‑processing but uses artifacts from the **RadioML** pipeline (Section 4). It performs local Newton–Puiseux analysis in **C²** (two complex features → 4 real inputs), robustness probes, LIME/SHAP explanations, calibration comparisons with 95% CIs (plus Wilcoxon and win‑rate), sensitivity summaries for (τ, δ), and resource benchmarking.

**Prerequisites**

- You have already run the RadioML pipeline (Section 4) and produced at least:
  - `up_radio/best_model_full.pt`
  - `up_radio/scaler_full.pkl`
  - `up_radio/uncertain_full.csv`
  - *(optional)* `up_radio/T_calib.pt` (if temperature scaling was used)
  - *(optional)* `up_radio/sens_grid.csv`, `up_radio/cv_metrics_per_fold_multi.csv`
- The RadioML dataset is available locally as described in **Datasets ⚠️** (place `RML2016.10a_dict.pkl` in `radio-data/`).

**Run**

```bash
# From the repo root (module form)
python -m post_processing_radio.post_processing_radio 
```

**What this does**

1. **Load model & artifacts (C²)**  
   Loads `best_model_full.pt` (infers in/hidden dims from weights), optional `T_calib.pt`, `scaler_full.pkl`, and uncertain anchors from `uncertain_full.csv`. Verifies the model’s first layer accepts 4‑real input (C²).

2. **Background for LIME/SHAP**  
   Rebuilds a background set from raw IQ windows using `prepare_complex_input(method='stft_stats')` to obtain C² features; applies the saved scaler for consistency.

3. **(τ, δ) Sensitivity summary (if `sens_grid.csv` present)**  
   Writes:
   - `sensitivity_detailed.csv` (the full grid),
   - `sensitivity_summary.txt` and `sensitivity_extra.txt` (aggregates, correlations, budget‑constrained best).

4. **Calibration comparison with 95% CI**  
   If `cv_metrics_per_fold_multi.csv` exists, compiles:
   - `comparative_table.csv` (mean ± CI for ECE/NLL/Brier/Acc/AUC),
   - `calibration_ci_report.txt` (human‑readable CI table),
   - `calibration_stats_tests.csv` (pairwise Wilcoxon; skipped if SciPy missing),
   - `calibration_winrate_vs_none.csv` (fold‑wise win‑rate vs **NONE**).

5. **Per‑anchor local analysis (for each uncertain point)**  
   - **Kink diagnostics** (modReLU neighborhood): non‑holomorphicity probe and kink fraction (skipped if no modReLU‑like activations).  
   - **Robust local surrogate**: degree‑4 complex polynomial with outlier/weighting safeguards; reports kept ratio, cond(A), rank.  
   - **Quality metrics**: RMSE, MAE, Pearson correlation, sign‑agreement vs the CVNN.  
   - **Newton–Puiseux expansions + interpretation** for local branches.  
   - **Robustness** along phase‑selected directions (class change & flip radius) with plots.  
   - **LIME & SHAP** explanations on consistently scaled C² features.  
   - **2D local contours** of the decision boundary for fixed dim pairs (1,3) and (0,2).  
   - **Resource benchmark**: Puiseux pipeline vs gradient saliency (time/memory).

6. **Calibration CI table (local 5‑fold on RadioML)**  
   Builds per‑method mean ± 95% CI for `none`, `platt`, `isotonic`, `beta`, `vector`, `temperature` using raw model probabilities (T=None) as the baseline, with:
   - `calibration_folds_raw.csv` (per‑fold values),
   - `calibration_ci_table.csv` and `calibration_ci_report.txt` (summaries).

7. **Branch‑multiplicity sensitivity**  
   Saves `branch_multiplicity_sensitivity.csv` showing ECE sensitivity to multiplicity mis‑estimation (via `sweep_multiplicity_misestimation`).

8. **Dominant-ratio summary**
   - Extracts dominant Puiseux coefficients per anchor (max |c₂| and |c₄| across branches) and computes the dominant-ratio proxy `r_dom = sqrt(|c₂|/|c₄|)`.
   - Writes a consolidated CSV for downstream evidence building (see Section 11): `post_processing_radio/dominant_ratio_summary.csv`.

**Outputs (saved to `post_processing_radio/`)**

- **Logs**
  - `post_processing_radio.log`

- **Sensitivity (τ, δ)**
  - `sensitivity_detailed.csv`, `sensitivity_summary.txt`, `sensitivity_extra.txt`

- **Calibration comparisons (from CV panel)**
  - `comparative_table.csv`
  - `calibration_ci_report.txt`
  - `calibration_stats_tests.csv` *(if SciPy available)*
  - `calibration_winrate_vs_none.csv`

- **Per‑anchor artifacts** (for each point *i*)
  - `benchmark_point<i>.txt`
  - `robustness_curves_point<i>.png`
  - `contour_point<i>_fix_dim=[1,3].png`, `contour_point<i>_fix_dim=[0,2].png`
  - `kink_sweep_point<i>.csv`
  - `resource_point<i>.txt`

- **Aggregate summaries**
  - `kink_summary.csv`, `kink_global_summary.txt`
  - `resource_summary.csv`
  - `local_fit_summary.csv`
  - `dominant_ratio_summary.csv` — per-anchor table with `point`, `c2_max_abs`, `c4_max_abs`, and the dominant-ratio proxy `r_dom = sqrt(|c2|/|c4|)`; consumed by `NP-analysis_radio.py` (Section 11).


- **Calibration CI (local 5‑fold)**
  - `calibration_folds_raw.csv`
  - `calibration_ci_table.csv`
  - `calibration_ci_report.txt`

- **Branch multiplicity**
  - `branch_multiplicity_sensitivity.csv`

### 10. Newton–Puiseux Evidence & Triage (MIT-BIH)

Build a compact, publication-ready summary of Newton–Puiseux evidence by **joining uncertain anchors** with **per-anchor post-processing reports** and (optionally) **dominant-ratio** estimates. Produces correlation stats, triage PR curves (AUPRC), head-to-head flip-rate summaries vs. XAI baselines, figures, and a short Markdown report.

**Prerequisites**

- You have completed **Section 3** (MIT-BIH uncertain points) and **Section 8** (Post-Processing Real Data), which should produce at least:
  - `up_real/uncertain_full.csv`
  - `post_processing_real/benchmark_point<i>.txt` (per-anchor reports)
  - *(optional)* `post_processing_real/dominant_ratio_summary.csv`  
    If absent, the script will fall back to `r_dom_pred` parsed from TXT; if missing but `|c2|,|c4|` are available, it will compute `r_dom ≈ sqrt(|c2|/|c4|)` automatically.

**Run**

```bash
# From the repo root
python -m NP-analysis_real.NP-analysis_real.py
````

**What this does**

1. **Load & join**

   * Loads anchors from `up_real/uncertain_full.csv` and assigns a stable `point` index `0..N-1`.
   * Parses `post_processing_real/benchmark_point<i>.txt` for:

     * **Kink diagnostics**: `frac_kink`, `frac_active`, `frac_inactive`.
     * **Local fit**: `kept_ratio`, `cond`, `rank`, `n_monomials`, `degree_used`, `retry`.
     * **Approx. quality**: `RMSE`, `MAE`, `Pearson`, `Sign_Agreement`, residual moments.
     * **Robustness**: per-direction flip radii and `min_flip_radius` (mapped to `r_flip_obs`).
     * **Timings/footprint**: `puiseux_time_s`, `saliency_ms`, CPU/GPU memory, `saliency_grad_norm`.
     * *(optional)* `r_dom_pred` and axis-baseline sweep flips: `flip_grad`, `flip_lime`, `flip_shap`.
   * Merges `post_processing_real/dominant_ratio_summary.csv` if present (normalized to columns: `c2_max_abs`, `c4_max_abs`, `r_dom`, `r_flip`).
   * If `r_dom` is missing but `|c2|,|c4|` exist, computes `r_dom = sqrt(|c2|/|c4|)`.

2. **Correlation & error summary**

   * Computes `MAE(|r_dom - r_flip_obs|)`, **Pearson** and **Spearman** correlations.
   * Saves a scatter with reference line and a robust regression slope:

     * `NP-analysis_real/figures/scatter_rdom_vs_rflip.png`
   * If `frac_kink` exists, saves:

     * `NP-analysis_real/figures/scatter_kink_vs_rflip.png`

3. **Triage analysis (AUPRC)**

   * Defines an anchor as **fragile** if `r_flip_obs ≤ BUDGET`, with `BUDGET = 0.02` (edit inside the script to adjust).
   * Builds PR curve and **AUPRC** for ranking by `|c4|` (dominant quartic term magnitude); also reports **F1-max** and threshold at F1-max.
     Files:

     * `NP-analysis_real/pr_by_abs_c4.csv`
     * `NP-analysis_real/figures/pr_curve_by_abs_c4.png`
   * Compares additional scores when available:

     * `1/r_grad`, `1/r_lime`, `1/r_shap` (derived from flip radii),
     * `grad_norm` (from saliency logs),
       and writes a comparison table:
     * `NP-analysis_real/triage_compare_summary.csv`
       (Per-score PR curves are saved as `pr_by_<prefix>.csv/png`, e.g., `pr_by_per_grad.csv`, `figures/pr_by_per_grad.png`.)

4. **Head-to-head flip rates**

   * Reports share of anchors with `r ≤ BUDGET` for Puiseux vs. XAI baselines, plus median radii:

     * `NP-analysis_real/xai_vs_puiseux_summary.csv`

5. **Joined evidence table + one-pager**

   * Saves the full joined table:

     * `NP-analysis_real/evidence_anchors_joined.csv`
   * Saves a compact correlation/PR summary:

     * `NP-analysis_real/corr_summary.csv`
   * Writes a Markdown one-pager with key numbers and figure pointers:

     * `NP-analysis_real/np_evidence_report.md`

**Outputs (folder `NP-analysis_real/`)**

* `evidence_anchors_joined.csv` — anchors + benchmark + dominant-ratio (joined).
* `corr_summary.csv` — MAE/ρ stats, PR summary, F1-max & threshold.
* `triage_compare_summary.csv` — AUPRC by score (`|c4|`, `1/r_*`, `grad_norm`).
* `xai_vs_puiseux_summary.csv` — hit-rates ≤ `BUDGET` and median radii by method.
* `pr_by_abs_c4.csv`, `figures/pr_curve_by_abs_c4.png` — main PR curve.
* `pr_by_<prefix>.csv`, `figures/pr_by_<prefix>.png` — extra PR curves (if available).
* `figures/scatter_rdom_vs_rflip.png` — `r_dom` vs `r_flip_obs`.
* `figures/scatter_kink_vs_rflip.png` — `frac_kink` vs `r_flip_obs` (if available).
* `np_evidence_report.md` — textual summary (ready to paste into appendix/supplement).

**Notes**

* If `dominant_ratio_summary.csv` is missing, the script uses `r_dom_pred` from TXT or re-computes `r_dom` from available `|c2|,|c4|`.
* To change the fragility budget for triage, edit `BUDGET` near the top of the script.

### 11. Newton–Puiseux Evidence & Triage (RadioML 2016.10A)

Same evidence-building and triage pipeline as above, but for **RadioML 2016.10A** artifacts.

**Prerequisites**

- You have completed **Section 4** (RadioML uncertain points) and **Section 9** (Post-Processing Radio Data), which should produce at least:
  - `up_radio/uncertain_full.csv`
  - `post_processing_radio/benchmark_point<i>.txt`
  - *(optional)* `post_processing_radio/dominant_ratio_summary.csv`  
    If absent, the script will use `r_dom_pred` parsed from TXT; if missing but `|c2|,|c4|` are available, it will compute `r_dom ≈ sqrt(|c2|/|c4|)` automatically.

**Run**

```bash
# From the repo root
python -m NP-analysis_radio.NP-analysis_radio.py
````

**What this does**

* **Join & normalize** anchors + benchmark TXT + optional dominant-ratio.
* **Correlation & error**: MAE(|`r_dom`−`r_flip_obs`|), Pearson/Spearman, scatter(s).
* **Triage (AUPRC)**: PR for `|c4|` + optional scores (`1/r_grad`, `1/r_lime`, `1/r_shap`, `grad_norm`), with CSVs and PNGs.
* **Head-to-head flip-rates** vs. XAI baselines.
* **Markdown one-pager** with key metrics and figure references.

**Outputs (folder `NP-analysis_radio/`)**

* `evidence_anchors_joined.csv`
* `corr_summary.csv`
* `triage_compare_summary.csv`
* `xai_vs_puiseux_summary.csv`
* `pr_by_abs_c4.csv`, `figures/pr_curve_by_abs_c4.png`
* `pr_by_<prefix>.csv`, `figures/pr_by_<prefix>.png` *(if available)*
* `figures/scatter_rdom_vs_rflip.png`
* `figures/scatter_kink_vs_rflip.png` *(if available)*
* `np_evidence_report.md`

**Notes**

* The fragility budget is `BUDGET = 0.02` (edit in the script to change).
* The script auto-detects `up_radio/` and `post_processing_radio/` relative to its location and writes results into `NP-analysis_radio/`.

---

## License
This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

**MIT-BIH Arrhythmia Database**  
ECG recordings are redistributed under the PhysioNet open-access license.  
Please ensure compliance with the original terms: https://physionet.org/content/mitdb/1.0.0/

---

**RadioML 2016.10A**  
This dataset is released by DeepSig under the Creative Commons Attribution‑NonCommercial‑ShareAlike 4.0 International (CC BY‑NC‑SA 4.0) license.  
The dataset is **not redistributed** in this repository. Please ensure compliance with the original terms and cite the dataset appropriately (see the DeepSig datasets page for details).

---

## Contact
For questions or contributions, please open an issue or contact Piotr Migus at migus.piotr@gmail.com.


