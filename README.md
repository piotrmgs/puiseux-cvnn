```markdown
# Newton‑Puiseux Analysis for Interpretability and Calibration of CVNNs

## Overview
This repository provides the full codebase and reproducible scripts for our Newton‑Puiseux framework, which enhances interpretability and calibration of complex‑valued neural networks (CVNNs). We demonstrate our approach on both a controlled synthetic dataset and the MIT‑BIH arrhythmia corpus.

## Project Structure

├── mit‑bih/                      # Raw MIT‑BIH dataset
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

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/piotrmgs/puiseux-cvnn.git
   cd puiseux-cvnn
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate   # Windows
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Usage
### 1. Synthetic Benchmark
```bash
python up_synth/up_synthetic.py --config configs/synthetic.yaml
python src/find_up_synthetic.py --threshold 0.1
python src/local_analysis.py --data uncertain_synthetic.csv --degree 4
python post_processing_synth/post_processing_synth.py --input results/synthetic/
```

### 2. MIT‑BIH Arrhythmia Corpus
1. Download and place the MIT‑BIH dataset in `mit‑bih/`.
2. Preprocess signals:
   ```bash
   python mit_bih_pre/pre_pro.py --input mit‑bih/ --output data/mit_bih_processed.npy
   ```
3. Train and calibrate CVNN:
   ```bash
   python up_real/up_real.py --train
   python post_processing_real/post_processing_real.py --input results/mit_bih/
   ```
4. Perform local interpretability analysis:
   ```bash
   python src/find_up_real.py --threshold 0.15
   python src/local_analysis.py --data uncertain_mitbih.csv --degree 4
   ```

## Module Descriptions
- **post_processing.py**: Common post‑processing utilities for both experiments
- **find_up_synthetic.py** / **find_up_real.py**: Identify uncertain samples for local analysis
- **local_analysis.py**: Fit local polynomial surrogates and invoke the Puiseux solver
- **puiseux.py**: Core Newton‑Puiseux decomposition algorithms
- **up_synthetic.py** / **up_real.py**: Training and evaluation pipelines for synthetic and real data
- **post_processing_synth.py** / **post_processing_real.py**: Generate summary reports and visualizations

## Results
All generated reports, plots, and metrics can be found in the `results/` directory, organized by experiment.

## License
This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Contact
For questions or contributions, please open an issue or contact Piotr Migus at migus.piotr@gmail.com.
```

