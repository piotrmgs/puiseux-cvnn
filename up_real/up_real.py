"""
main.py

Main script for training and evaluating a complex-valued neural network (CVNN) 
on the MIT-BIH Arrhythmia Dataset using K-fold cross-validation across patients.

This script performs the following:
- Loads and preprocesses ECG windows from MIT-BIH records.
- Transforms real-valued time series into complex-valued representations.
- Trains a complex-valued neural network using cross-patient K-fold CV.
- Calibrates the model using temperature scaling.
- Collects predictions for all folds and performs global analyses:
    - Training curves (loss and accuracy)
    - Calibration curve (reliability diagram)
    - Uncertainty histogram
    - Complex PCA scatter plot
    - Ablation support
- Extracts and saves uncertain predictions to CSV.
- Optionally retrains the model on the full dataset and saves the best weights.

Key Features:
-------------
- Complex feature extraction using Hilbert-based analytics or PCA.
- Model architecture: SimpleComplexNet (custom CVNN)
- Cross-validation ensures generalization across patients.
- Visual tools for model interpretability and calibration evaluation.
- CLI arguments for easy experiment control.

Usage:
------
$ python main.py --data_folder ./mit-bih --output_folder ./results --epochs 20 --folds 10 --threshold 0.6

CLI Arguments:
--------------
--data_folder       Path to preprocessed MIT-BIH data directory
--output_folder     Directory to save models, plots, and logs
--epochs            Number of training epochs
--lr                Learning rate
--batch_size        Batch size for DataLoaders
--folds             Number of cross-validation folds (default: 10)
--threshold         Probability threshold for uncertainty detection
--cpu               Force training on CPU (useful for debugging)

Outputs:
--------
- run.log                   : full training logs
- *.png                     : plots for performance, calibration, uncertainty
- uncertain_all.csv         : CSV with uncertain test examples
- best_model_full.pt        : best model weights from full retraining

Requires:
---------
- Python ≥ 3.9
- PyTorch
- NumPy, SciPy, scikit-learn, seaborn, matplotlib
- Custom modules: `mit_bih_pre` and `src.find_up_*`

Author:
-------
[Your Name or Team], [Year]
"""

import os
import argparse
import logging
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score
import time
from sklearn.metrics import accuracy_score, roc_auc_score

# ───────────────────────────────────────────────────────────
#  Project imports
# ───────────────────────────────────────────────────────────
from src.find_up_synthetic import (
    SimpleComplexNet,
    complex_modulus_to_logits,
    find_uncertain_points,
)
from mit_bih_pre.pre_pro import load_mitbih_data

from src.find_up_real import (
    prepare_complex_input,
    create_dataloaders,
    train_model as train_real,
    save_plots,
    tune_temperature,
    save_confusion_roc,
    expected_calibration_error,
    save_overall_history,
    save_calibration_curve,
    save_uncertainty_hist,
    save_complex_pca_scatter,
    save_ablation_barplot,
    WINDOW_SIZE,
    PRE_SAMPLES,
    FS,
)
# ───────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Complex‑Valued NN on MIT‑BIH with visualisations"
    )
    parser.add_argument("--data_folder", type=str, default="mit-bih")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--output_folder",
        type=str,
        default=script_dir,
        help="Where to save models and figures",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5001)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()

record_names = [
        "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
        "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
        "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
        "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
        "222", "223", "228", "230", "231", "232", "233", "234",
    ]

# ───────────────────────────────────────────────────────────
#  main
# ───────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_folder, "run.log")),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    logger.info("Using device: %s", device)

    

    # ── global collectors ─────────────────────────────────
    histories_all_folds = []
    y_true_all, y_pred_all, y_prob_all = [], [], []
    all_uncertain = []

    
    record_names = [
        "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
        "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
        "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
        "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
        "222", "223", "228", "230", "231", "232", "233", "234",
    ]


    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(record_names), 1):
        train_recs = [record_names[i] for i in train_idx]
        test_recs = [record_names[i] for i in test_idx]
        logger.info("Fold %d  train=%s  test=%s", fold, train_recs, test_recs)

        # ── load raw windows (real‑valued) ─────────────────
        X_train, y_train = load_mitbih_data(
            args.data_folder, train_recs, WINDOW_SIZE, PRE_SAMPLES, FS
        )
        X_test, y_test = load_mitbih_data(
            args.data_folder, test_recs, WINDOW_SIZE, PRE_SAMPLES, FS
        )

        
        # ───────────────────────────────────────────────────
        #  CVNN + split‑PCA  (główna ścieżka artykułu)
        # ───────────────────────────────────────────────────
        X_tr_s = prepare_complex_input(X_train, method='complex_stats')
        X_te_s = prepare_complex_input(X_test,  method='complex_stats')
        tr_s, te_s = create_dataloaders(X_tr_s, y_train, X_te_s, y_test,
                                        batch_size=args.batch_size)
        model_s = SimpleComplexNet(in_features=X_tr_s.shape[1]//2,
                                    hidden_features=64, out_features=2, bias=0.1)
        t0 = time.perf_counter()
        history, best_s = train_real(model_s, tr_s, te_s,
                                     epochs=args.epochs, lr=args.lr, device=device)
        train_time_s = (time.perf_counter() - t0) / args.epochs
        histories_all_folds.append(history)
        model_s.load_state_dict(best_s)
        #T = nn.Parameter(torch.ones(1, device=device) * 1.5)
        #T = tune_temperature(model_s, te_s, device=device)
        #torch.save(T, os.path.join(args.output_folder, "T_calib.pt"))
        #logger.info(f"[INFO] Learned temperature T={T.item():.3f}")
        
        model_s.eval()

        y_s_true, y_s_pred, y_s_prob = [], [], []
        with torch.no_grad():
            for xb, yb in te_s:
                out  = model_s(xb.to(device))
                logits = complex_modulus_to_logits(out)
                p    =  nn.Softmax(dim=1)(logits)
                preds= p.argmax(dim=1)
                y_s_true.extend(yb.tolist())
                y_s_pred.extend(preds.cpu().tolist())
                y_s_prob.extend(p[:,1].cpu().tolist())

        

        
       # extend for global visualisations
        y_true_all.extend(y_s_true)
        y_pred_all.extend(y_s_pred)
        y_prob_all.extend(y_s_prob)

        # uncertain points
        #uncertain = find_uncertain_points(
        #    model_s.to('cpu'),
        #    torch.tensor(X_te_s, dtype=torch.float32),
        #    torch.tensor(y_test,  dtype=torch.long),
        #    prob_thr=args.threshold,
        #    temperature=T 
        #)
        
        #all_uncertain.extend(uncertain)
        #logger.info("Fold %d uncertain: %d", fold, len(uncertain))

        

    # Global visualisations
    save_overall_history(histories_all_folds, args.output_folder)
    save_calibration_curve(y_true_all, y_prob_all, args.output_folder)
    save_uncertainty_hist(y_prob_all, args.threshold, args.output_folder)
    X_full, y_full = load_mitbih_data(args.data_folder, record_names, WINDOW_SIZE, PRE_SAMPLES, FS)
    save_complex_pca_scatter(
        prepare_complex_input(X_full, method='split_pca', n_real=2),
        y_full, args.output_folder
    )

    

    # Save uncertain CSV
    #with open(os.path.join(args.output_folder, "uncertain_all.csv"), "w", newline="") as f:
    #    writer = csv.DictWriter(f, fieldnames=["index","X","true_label","p1","p2"])
    #    writer.writeheader()
    #    for idx,u in enumerate(all_uncertain):
    #        p1,p2 = u['prob']
    #        writer.writerow({
    #            'index': idx, 'X': u['X'], 'true_label': u['true_label'], 'p1': p1, 'p2': p2
    #        })
    #logger.info("Saved uncertain CSV to %s", args.output_folder)

    # Retrain on full dataset (optional)
    logger.info("Retraining on full dataset…")
    X_tr_all, X_te_all, y_tr_all, y_te_all = train_test_split(
        prepare_complex_input(X_full, method='split_pca', n_real=2), y_full,
        test_size=0.2, stratify=y_full, random_state=42
    )
    tr_all, te_all = create_dataloaders(X_tr_all, y_tr_all, X_te_all, y_te_all, batch_size=args.batch_size)
    model_all = SimpleComplexNet(
        in_features=X_tr_all.shape[1]//2, hidden_features=64, out_features=2, bias=0.1
    )
    hist_all, best_all = train_real(model_all, tr_all, te_all, epochs=args.epochs, lr=args.lr, device=device)
    torch.save(best_all, os.path.join(args.output_folder, 'best_model_full.pt'))
    save_plots(hist_all, args.output_folder, 'full')
    logger.info('Saved full-model and plots')

    # Temperature calibration for the full model ──────────────────────
    model_all.load_state_dict(best_all)
    model_all.eval()
    #T_full = nn.Parameter(torch.ones(1, device=device) * 1.5)
    
    T_full = tune_temperature(model_all, te_all, device=device)
    torch.save(T_full, os.path.join(args.output_folder, 'T_calib.pt'))
    logger.info(f"[INFO] Full‑model temperature T={T_full.item():.3f} saved to {args.output_folder}")

    # Detect uncertain points using the calibrated model ─────────────
    uncertain_full = find_uncertain_points(
        model_all.to('cpu'),
        torch.tensor(X_te_all, dtype=torch.float32),
        torch.tensor(y_te_all, dtype=torch.long),
        prob_thr=args.threshold,
        temperature=T_full
    )
    logger.info(f"Full-model uncertain points: {len(uncertain_full)}")

    # ── 3) Save CSV only once ────────────────────────────────────────────
    csv_path = os.path.join(args.output_folder, 'uncertain_full.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index','X','true_label','p1','p2'])
        writer.writeheader()
        for u in uncertain_full:
            p1,p2 = u['prob']
            writer.writerow({
                'index':      u['index'],
                'X':          u['X'],
                'true_label': u['true_label'],
                'p1':         p1,
                'p2':         p2
            })
    logger.info(f"[INFO] Saved full-model uncertain points to {csv_path}")

    
if __name__ == "__main__":
    main()