---

## About this repo
This repository contains the complete, reproducible codebase that accompanies the paper *“Newton-Puiseux Analysis for Interpretability and Calibration of Complex-Valued Neural Networks”*.  

[![arXiv](https://img.shields.io/badge/arXiv-2504.19176-b31b1b.svg)](https://arxiv.org/abs/2504.19176)


https://arxiv.org/pdf/2504.19176

It implements our end-to-end pipeline – from data preprocessing through CVNN training to Newton-Puiseux-based local analysis – for both a controlled synthetic benchmark and the MIT-BIH Arrhythmia corpus.

---


# Newton‑Puiseux Analysis for Interpretability and Calibration of CVNNs

## Overview
This repository provides the full codebase and reproducible scripts for our Newton‑Puiseux framework, which enhances interpretability and calibration of complex‑valued neural networks (CVNNs). We demonstrate our approach on both a controlled synthetic dataset and the MIT‑BIH arrhythmia corpus.

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
│   ├── local_analysis.py         # Local surrogate + Puiseux wrapper
│   ├── puiseux.py                # Newton‑Puiseux solver
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
├── post_processing_real/         # Post‑processing for MIT‑BIH data
│   └── post_processing_real.py
└── README.md                     # This file
```

## Dataset ⚠️

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
## Quick Start (synthetic benchmark in 3 steps)
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

To train, evaluate, and extract uncertain predictions on the MIT‑BIH Arrhythmia dataset, run:

```bash
python -m up_real.up_real
```

This script executes the following pipeline:

1. **Data loading & preprocessing**: Reads raw ECG windows (L=128 samples, P=50 pre‑samples) from `--data_folder` and applies band‑pass filtering and Hilbert‑transform to obtain analytic signals.
2. **Complex feature extraction**: Converts real-valued time series into 2‑dimensional complex statistics (`prepare_complex_input` with `method='complex_stats'`).
3. **Cross‑patient K‑Fold CV**: Splits patient records into K folds (default 10), ensuring no subject appears in both train and test splits.
4. **Model training (per fold)**:
   - Builds a `SimpleComplexNet` (in_features=2 complex dims, hidden=64, out_features=2).
   - Trains for `--epochs` using Adam (learning rate `--lr`).
   - Records training history and selects best weights per fold.
   - Evaluates on test split and accumulates predictions (`y_true`, `y_pred`, `y_prob`).
5. **Global visualizations**:
   - **Training curves** (loss & accuracy over epochs across folds; saved as `training_history.png`).
   - **Calibration curve** (reliability diagram; `calibration_curve.png`).
   - **Uncertainty histogram** (distribution of predicted probabilities; `uncertainty_histogram.png`).
   - **Complex PCA scatter** of all data using `method='split_pca'` (`complex_pca_scatter.png`).
   - **Ablation bar plot** showing impact of components (`ablation_barplot.png`).
6. **Uncertain sample export**: Aggregates test‑fold samples with max softmax probability < `--threshold` and saves to `up_real/uncertain_full.csv`.
7. **Optional full‑data retraining**:
   - Splits the entire dataset 80/20, retrains the model, and saves best weights to `up_real/best_model_full.pt`.
   - Saves full‑model training plots (`*_full.png`).
8. **Temperature calibration**: Learns scalar temperature `T` on the full‑model test split (`tune_temperature`), then saves `up_real/T_calib.pt`.
9. **Final uncertainty detection**: Applies calibrated model on held‑out full‑data test split to flag uncertain beats (same `--threshold`) and writes them to `up_real/uncertain_full.csv`.

**Outputs:**

- `run.log` — detailed logs of training, fold splits, and calibration steps.
- `training_history.png`, `calibration_curve.png`, `uncertainty_histogram.png`, `complex_pca_scatter.png`, `ablation_barplot.png` — visualization artifacts.
- `uncertain_full.csv` — indices, features, true labels, and class probabilities of uncertain ECG windows.
- `best_model_full.pt` — best model weights after full‑data retraining.
- `T_calib.pt` — learned temperature parameter for scaled softmax.

### 4. Puiseux Test

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

### 5. Local Analysis test

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



### 6. Post-Processing Synthetic Data

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

### 7. Post-Processing Real Data

To analyze uncertain ECG windows, generate local explanations, and robustness analyses on the MIT‑BIH dataset, run:

```bash
python -m post_processing_real.post_processing_real
```

This script performs the following steps:

1. **Model & data loading**:
   - Loads the best retrained `SimpleComplexNet` weights from `up_real/best_model_full.pt`.
   - Loads the temperature scaling parameter `T_calib.pt` if available (for calibrated softmax).
   - Reads uncertain points from `up_real/uncertain_full.csv`.
2. **Background data for LIME**:
   - Samples 512 random ECG windows from the full dataset.
   - Compresses them to C2 features via `compress_to_C2` for background reference.
3. **Local polynomial fitting**: Computes a degree-4 surrogate `F̂` around each uncertain point (`delta=0.05`, `n_samples=300`) with `local_poly_approx_complex`.
4. **Approximation quality**: Evaluates RMSE, MAE, Pearson correlation, and sign-agreement with `evaluate_poly_approx_quality`.
5. **Puiseux expansions**: Computes series at each point using `puiseux_uncertain_point` and interprets them (`interpret_puiseux_expansions`).
6. **Adversarial robustness**:
   - Identifies top directions via `find_adversarial_directions` on the surrogate.
   - Measures flip radii with `test_adversarial_impact` and plots curves (`plot_robustness_curve`).
7. **Local explanations**:
   - Computes LIME explanations (`compute_lime_explanation`) using the compressed background set.
   - Computes SHAP values (`compute_shap_explanation`) for each test point.
8. **2D decision contours**: Generates contour plots fixing dimensions (1,3) and (0,2) via `plot_local_contour_2d`.
9. **Report generation**: For each point, writes `post_processing_real/benchmark_point<idx>.txt` containing:
   - Base point coordinates and approximation metrics.
   - Puiseux expressions & interpretations.
   - Robustness analysis table with directions, phases, class changes, and flip radii.
   - LIME & SHAP feature attributions.
   - File paths of generated robustness and contour plots.

**Outputs:**

- `post_processing_real/benchmark_point<idx>.txt` — detailed local analysis report for each uncertain ECG window.
- `post_processing_real/robustness_curves_point<idx>.png` — adversarial robustness plots.
- `post_processing_real/contour_point<idx>_fix_dim=[...].png` — 2D decision boundary visualizations.

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


