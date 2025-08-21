# Copyright (c) 2025 Piotr Migus
# This code is licensed under the MIT License.
# See the LICENSE file in the repository root for full license information.

"""
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
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import logging
import pickle
import csv
import psutil
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
    create_train_val_test_loaders,
    train_model as train_real,
    save_plots,
    tune_temperature,
    fit_isotonic_on_val,          
    save_confusion_roc,
    expected_calibration_error,
    save_overall_history,
    save_calibration_curve,
    save_uncertainty_hist,
    save_complex_pca_scatter,
    save_feature_embedding_2d,    
    select_thresholds_budget_count,
    save_ablation_barplot,
    sensitivity_analysis,
    save_sensitivity_heatmaps,
    save_capture_abstain_curve,   
    negative_log_likelihood,
    brier_score,
    mean_ci,
    seed_everything,
    WINDOW_SIZE,
    PRE_SAMPLES,
    FS,
    get_calibrator,
)


# ───────────────────────────────────────────────────────────


def parse_args():
    """Parse CLI arguments controlling data paths, training hyperparameters, CV, and analysis options."""
    parser = argparse.ArgumentParser(
        description="Complex-Valued NN on MIT-BIH with visualizations"
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

    # ▼▼▼ Grid-based (tau/delta) sensitivity analysis + heatmaps — ENABLED by default ▼▼▼
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        default=True,
        help="Enable tau/delta grid analysis (ON by default)."
    )
    # Optional switch to disable the analysis if needed:
    parser.add_argument(
        "--no-sensitivity",
        dest="sensitivity",
        action="store_false",
        help="Disable tau/delta grid analysis."
    )

    parser.add_argument("--capture_target", type=float, default=0.80,
                        help="Target error-capture for mode='capture' (default 0.80).")
    parser.add_argument("--select_mode", type=str, default="budget",
                        choices=["capture","budget","risk","knee"],
                        help="Selection criterion for (tau, delta); default 'budget'.")
    parser.add_argument("--max_abstain", type=float, default=0.20,
                        help="Review budget (fraction) for mode='budget' (default 0.20).")
    parser.add_argument("--target_risk", type=float, default=None,
                        help="Target risk among accepted for mode='risk' (optional).")

    parser.add_argument("--review_budget", type=int, default=10,
                        help="Exact-count review budget on VAL when selecting (tau, delta).")
    parser.add_argument("--seed", type=int, default=12345,
                        help="Global seed for full reproducibility")
    parser.add_argument("--calibration", type=str, default="temperature",
                        choices=["temperature", "isotonic", "none"],
                        help="Calibration method fitted on VALIDATION (default: temperature).")
    parser.add_argument(
        "--calibs",
        type=str,
        default="temperature,isotonic,platt,beta,vector,none",
        help=("Comma-separated list of calibration methods to evaluate in one run. "
              "Supported: temperature,isotonic,platt,beta,vector,none")
    )


    parser.add_argument("--budget_fracs", type=str, default="",
                        help="Optional comma-separated list of fractional review budgets, e.g. '0.005,0.02'.")
    parser.add_argument("--embed_method", type=str, default="tsne",
                        choices=["tsne", "umap"],
                        help="2D projection method for penultimate features (default: tsne).")


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
    """Entry point: run cross-patient K-fold CV, collect metrics/plots, then retrain on full data."""
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    
    # logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_folder, "run.log"), mode="w"),
            logging.StreamHandler(),
        ],
        force=True,  # ensure reconfiguration even if logging was set elsewhere
    )


    logger = logging.getLogger()

    # Reproducibility: seed Python/NumPy/PyTorch (CPU/CUDA)
    seed_everything(args.seed)
    
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    logger.info("Using device: %s | seed=%d", device, args.seed)
    
    # Save run metadata (useful for reproducibility in rebuttals/appendices)
    try:
        import json, platform
        meta = {
            "seed": args.seed,
            "device": str(device),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        }
        with open(os.path.join(args.output_folder, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as _e:
        logger.warning("Could not write run_meta.json: %s", _e)

    # ── Global collectors across folds ─────────────────────────────────    
    histories_all_folds = []
    # Per-fold predictions (for plots)
    y_true_all, y_pred_all = [], []
    y_prob_all_pos = []   # P(y=1) — for calibration curve
    y_conf_all_max = []   # max softmax — for uncertainty histogram
    y_conf_all = []       # (kept for compatibility, not used)
    all_uncertain = []

    # RAW baseline collectors for calibration (no temperature scaling)
    y_true_all_raw = []
    y_prob_all_pos_raw = []

    y_margin_all = []        
    pred_rows_all = []        

    # K-fold CV over patients/records
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    # Per-fold metrics (RAW vs TS)
    ece_raw_folds, ece_ts_folds = [], []
    nll_raw_folds, nll_ts_folds = [], []
    br_raw_folds,  br_ts_folds  = [], []
    acc_ts_folds, auc_ts_folds  = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(record_names), 1):
        train_recs = [record_names[i] for i in train_idx]
        test_recs = [record_names[i] for i in test_idx]
        logger.info("Fold %d  train=%s  test=%s", fold, train_recs, test_recs)

        # ── Load raw (real-valued) windows ─────────────────
        X_train, y_train = load_mitbih_data(
            args.data_folder, train_recs, WINDOW_SIZE, PRE_SAMPLES, FS
        )
        X_test, y_test = load_mitbih_data(
            args.data_folder, test_recs, WINDOW_SIZE, PRE_SAMPLES, FS
        )

        # ───────────────────────────────────────────────────
        #  CVNN + complex_stats (main article path)
        # ───────────────────────────────────────────────────
        X_tr_s = prepare_complex_input(X_train, method='complex_stats')
        X_te_s = prepare_complex_input(X_test,  method='complex_stats')

        # Split TRAIN → TRAIN/VAL (stratified)
        X_tr_s_tr, X_tr_s_val, y_train_tr, y_train_val = train_test_split(
            X_tr_s, y_train, test_size=0.2, stratify=y_train, random_state=args.seed + 1000 + fold
        )

        # Build loaders: train/val/test (scaler fit on TRAIN only)
        tr_ld, val_ld, te_ld, scaler = create_train_val_test_loaders(
            X_tr_s_tr, y_train_tr,
            X_tr_s_val, y_train_val,
            X_te_s,     y_test,
            batch_size=args.batch_size,
            seed=args.seed
        )
        # Zapisz scaler dla tego folda (przyda się w postprocessingu)
        with open(os.path.join(args.output_folder, f"scaler_fold{fold}.pkl"), "wb") as f:
            pickle.dump(scaler, f)


        # Model
        model_s = SimpleComplexNet(
            in_features=X_tr_s.shape[1] // 2,
            hidden_features=64,
            out_features=2,
            bias=0.1,
        )
        t0 = time.perf_counter()
        history, best_s = train_real(
            model_s, tr_ld, val_ld, epochs=args.epochs, lr=args.lr, device=device
        )
        train_time_s = (time.perf_counter() - t0) / len(history["train_loss"])
        histories_all_folds.append(history)
        
        save_plots(history, args.output_folder, fold)

        # Resource snapshot (CPU RSS, optional GPU peak) per fold
        try:
            proc = psutil.Process(os.getpid())
            rss_mb = proc.memory_info().rss / (1024**2)
            if device.type == "cuda":
                peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
                torch.cuda.reset_peak_memory_stats(device)
                logging.info(f"[Fold {fold}] avg_epoch_time={train_time_s:.3f}s | CPU_RSS={rss_mb:.1f}MB | GPU_peak={peak_mb:.1f}MB")
            else:
                logging.info(f"[Fold {fold}] avg_epoch_time={train_time_s:.3f}s | CPU_RSS={rss_mb:.1f}MB")
        except Exception as _e:
            logging.warning(f"[Fold {fold}] memory logging failed: {_e}")

        # Load best weights (by val acc)
        model_s.load_state_dict(best_s)

        # --- Choose calibration on VALIDATION (temperature / isotonic / none) ---
        T_fold = None
        iso_cal = None
        is_binary = False
        if args.calibration == "temperature":
            T_fold = tune_temperature(model_s, val_ld, device=device)
            torch.save(T_fold, os.path.join(args.output_folder, f"T_calib_fold{fold}.pt"))
            logging.info(f"[Fold {fold}] Learned temperature T={T_fold.item():.3f}")
        elif args.calibration == "isotonic":
            iso_cal, is_binary = fit_isotonic_on_val(model_s, val_ld, device=device)
            if not is_binary:
                logging.warning("[Fold %d] Isotonic calibration skipped (multi-class).", fold)
        else:
            logging.info("[Fold %d] No calibration ('none').", fold)


        # --- Per-fold evaluation: RAW (no T) vs TS (with T) on TEST ---
        model_s.eval()
        y_true_raw, y_prob_pos_raw, y_max_raw = [], [], []
        y_true_ts,  y_prob_pos_ts,  y_max_ts  = [], [], []
        y_pred_ts = []
        probs_ts_all = []
        yb_all = []

        
        with torch.no_grad():
            for xb, yb in te_ld:
                xb = xb.to(device)
                logits = complex_modulus_to_logits(model_s(xb))

                # RAW probabilities (always computed)
                probs_raw = nn.Softmax(dim=1)(logits)

                # Calibrated probabilities according to --calibration
                if args.calibration == "temperature" and T_fold is not None:
                    probs_ts = nn.Softmax(dim=1)(logits / T_fold.to(device))
                elif args.calibration == "isotonic" and iso_cal is not None:
                    # iso_cal works on numpy probs -> return numpy -> convert back to tensor
                    probs_ts = torch.from_numpy(iso_cal(probs_raw.cpu().numpy()))
                else:
                    probs_ts = probs_raw

                probs_ts_all.append(probs_ts.cpu())
                yb_all.append(yb.cpu())

                # RAW collectors
                y_true_raw.extend(yb.tolist())
                y_prob_pos_raw.extend(probs_raw[:, 1].cpu().tolist())
                y_max_raw.extend(probs_raw.max(dim=1).values.cpu().tolist())

                # CALIBRATED collectors (keep *_ts variable names to avoid touching the rest)
                y_true_ts.extend(yb.tolist())
                y_prob_pos_ts.extend(probs_ts[:, 1].cpu().tolist())
                y_max_ts.extend(probs_ts.max(dim=1).values.cpu().tolist())
                y_pred_ts.extend(probs_ts.argmax(dim=1).cpu().tolist())

         
        # Acc/AUC (TS) for this fold
        acc_ts = float((np.array(y_true_ts) == np.array(y_pred_ts)).mean())
        try:
            auc_ts = float(roc_auc_score(np.array(y_true_ts), np.array(y_prob_pos_ts)))
        except Exception:
            auc_ts = float('nan')
        acc_ts_folds.append(acc_ts)
        auc_ts_folds.append(auc_ts)
        
        roc_auc = save_confusion_roc(y_true_ts, y_pred_ts, y_prob_pos_ts, args.output_folder, fold)
        logging.info(f"[Fold {fold}] ROC AUC (TS) = {roc_auc:.4f}")


        # --- Per-sample rows for CSV + margin (TS) ---
        probs_ts_all = torch.cat(probs_ts_all, dim=0)   # shape: [N_test, 2]
        yb_all       = torch.cat(yb_all, dim=0)         # shape: [N_test]

        top2 = torch.topk(probs_ts_all, k=2, dim=1).values
        margin_batch = (top2[:, 0] - top2[:, 1]).abs().tolist()
        y_margin_all.extend(margin_batch)

        pmax_batch  = probs_ts_all.max(dim=1).values.tolist()
        p1_batch    = probs_ts_all[:, 0].tolist()
        p2_batch    = probs_ts_all[:, 1].tolist()
        pred_batch  = probs_ts_all.argmax(dim=1).tolist()
        true_batch  = yb_all.tolist()

        base_row_id = len(pred_rows_all)
        for i in range(len(true_batch)):
            pred_rows_all.append({
                "fold": fold,
                "row_global": base_row_id + i,
                "true": int(true_batch[i]),
                "pred_TS": int(pred_batch[i]),
                "p1_TS": float(p1_batch[i]),
                "p2_TS": float(p2_batch[i]),
                "pmax_TS": float(pmax_batch[i]),
                "margin_TS": float(margin_batch[i]),
            })


        # Global collectors (TS for figures)
        y_true_all.extend(y_true_ts)
        y_pred_all.extend(y_pred_ts)
        y_prob_all_pos.extend(y_prob_pos_ts)
        y_conf_all_max.extend(y_max_ts)
        
        # Global collectors (RAW for baseline reliability curve)
        y_true_all_raw.extend(y_true_raw)
        y_prob_all_pos_raw.extend(y_prob_pos_raw)

        # Per-fold metrics (RAW vs TS)
        ece_raw = expected_calibration_error(np.array(y_true_raw), np.array(y_prob_pos_raw), n_bins=10)
        ece_ts  = expected_calibration_error(np.array(y_true_ts),  np.array(y_prob_pos_ts),  n_bins=10)
        nll_raw = negative_log_likelihood(np.array(y_true_raw), np.array(y_prob_pos_raw))
        nll_ts  = negative_log_likelihood(np.array(y_true_ts),  np.array(y_prob_pos_ts))
        br_raw  = brier_score(np.array(y_true_raw), np.array(y_prob_pos_raw))
        br_ts   = brier_score(np.array(y_true_ts),  np.array(y_prob_pos_ts))

        ece_raw_folds.append(ece_raw); ece_ts_folds.append(ece_ts)
        nll_raw_folds.append(nll_raw); nll_ts_folds.append(nll_ts)
        br_raw_folds.append(br_raw);   br_ts_folds.append(br_ts)

        cal_suffix = "TS" if args.calibration == "temperature" else ("ISO" if args.calibration == "isotonic" else "CAL")
        logger.info(
            "[Fold %d] epoch_avg=%.2fs | params=%d | ECE raw=%.4f→%.4f %s | NLL raw=%.4f→%.4f | Brier raw=%.4f→%.4f",
            fold, train_time_s, sum(p.numel() for p in model_s.parameters()),
            ece_raw, ece_ts, cal_suffix, nll_raw, nll_ts, br_raw, br_ts
        )

        # ========== NEW: multi-calibration block (runs in addition to RAW vs selected --calibration) ==========
        methods = [m.strip().lower() for m in args.calibs.split(",") if m.strip()]

        # 1) Cache TEST logits once (to avoid re-running the model per method)
        logits_list_mc, y_list_mc = [], []
        with torch.no_grad():
            for xb, yb in te_ld:
                xb = xb.to(device)
                lo = complex_modulus_to_logits(model_s(xb))
                logits_list_mc.append(lo.cpu().numpy())
                y_list_mc.extend(yb.cpu().numpy().tolist())
        logits_np_mc = np.vstack(logits_list_mc)          # [N_test, C]
        y_np_mc = np.asarray(y_list_mc, dtype=int)        # [N_test]

        # Softmax for RAW (shared by ISO/BETA)
        z_mc = logits_np_mc - logits_np_mc.max(axis=1, keepdims=True)
        expz_mc = np.exp(z_mc)
        probs_raw_np_mc = expz_mc / expz_mc.sum(axis=1, keepdims=True)

        fold_rows = []  # rows for this fold across all methods

        # 2) Fit calibrator on VALIDATION per method, then apply to TEST
        for method in methods:
            apply_fn, tag = get_calibrator(model_s, val_ld, method=method, device=device)
            if apply_fn is None:
                logging.info(f"[Fold {fold}] Skipping method={method} (not applicable).")
                continue

            # Apply on TEST
            if method in ("temperature", "vector", "platt"):
                probs_np = apply_fn(logits_np_mc)                # expects logits
            elif method in ("isotonic", "beta"):
                probs_np = apply_fn(probs_np=probs_raw_np_mc)    # expects probs
            elif method == "none":
                probs_np = probs_raw_np_mc
            else:
                probs_np = probs_raw_np_mc  # safe fallback

            # Metrics (binary-friendly; AUC/ECE/NLL/Brier only if C==2)
            preds = probs_np.argmax(axis=1)
            acc   = float((preds == y_np_mc).mean())
            if probs_np.shape[1] == 2:
                try:
                    auc   = float(roc_auc_score(y_np_mc, probs_np[:, 1]))
                except Exception:
                    auc = float("nan")
                ece   = expected_calibration_error(y_np_mc, probs_np[:, 1], n_bins=10)
                nll   = negative_log_likelihood(y_np_mc, probs_np[:, 1])
                brier = brier_score(y_np_mc, probs_np[:, 1])
            else:
                auc = ece = nll = brier = float("nan")

            fold_rows.append({
                "fold": fold, "method": method.upper(), "tag": tag,
                "ECE": ece, "NLL": nll, "Brier": brier, "Acc": acc, "AUC": auc
            })

        # 3) Append to a fold-wise CSV (one file for the whole run)
        multi_path = os.path.join(args.output_folder, "cv_metrics_per_fold_multi.csv")
        write_header = not os.path.exists(multi_path)
        with open(multi_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["fold","method","tag","ECE","NLL","Brier","Acc","AUC"])
            if write_header:
                w.writeheader()
            w.writerows(fold_rows)
        logging.info(f"[Fold {fold}] Wrote multi-calibration rows to {multi_path}")
        # ========== /NEW ==========


    if pred_rows_all:
        pred_csv_path = os.path.join(args.output_folder, "predictions_all_folds.csv")
        with open(pred_csv_path, "w", newline="") as fcsv:
            cw = csv.DictWriter(fcsv, fieldnames=list(pred_rows_all[0].keys()))
            cw.writeheader()
            cw.writerows(pred_rows_all)
        logging.info(f"Saved predictions with margins to {pred_csv_path}")

    # --- Save per-fold metrics for transparency (place right after the K-Fold loop) ---
    per_fold_path = os.path.join(args.output_folder, "cv_metrics_per_fold.csv")
    with open(per_fold_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold","ECE_raw","ECE_TS","NLL_raw","NLL_TS","Brier_raw","Brier_TS","Acc_TS","AUC_TS"])
        for i in range(len(ece_raw_folds)):
            w.writerow([i + 1,
                        f"{ece_raw_folds[i]:.6f}", f"{ece_ts_folds[i]:.6f}",
                        f"{nll_raw_folds[i]:.6f}", f"{nll_ts_folds[i]:.6f}",
                        f"{br_raw_folds[i]:.6f}",  f"{br_ts_folds[i]:.6f}",
                        f"{acc_ts_folds[i]:.6f}",  f"{auc_ts_folds[i]:.6f}"])

    logger.info("Saved per-fold CV metrics to %s", per_fold_path)

    save_overall_history(histories_all_folds, args.output_folder)

    # Save reliability diagrams for baseline (RAW) and calibrated (TS)
    save_calibration_curve(y_true_all_raw, y_prob_all_pos_raw, args.output_folder, suffix="RAW")
    cal_suffix = "TS" if args.calibration == "temperature" else ("ISO" if args.calibration == "isotonic" else "CAL")
    save_calibration_curve(y_true_all, y_prob_all_pos, args.output_folder, suffix=cal_suffix)


    def _row(name, raw_list, ts_list):
        """Helper: summary row with mean, 95% CI half-width, and relative drop from RAW to TS."""
        m_raw, ci_raw = mean_ci(raw_list)
        m_ts,  ci_ts  = mean_ci(ts_list)
        rel_drop = (m_raw - m_ts) / max(m_raw, 1e-12)
        return [name, m_raw, ci_raw, m_ts, ci_ts, rel_drop]

    table = [
        _row("ECE", ece_raw_folds, ece_ts_folds),
        _row("NLL", nll_raw_folds, nll_ts_folds),
        _row("Brier", br_raw_folds, br_ts_folds),
    ]

    with open(os.path.join(args.output_folder, "cv_metrics_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "RAW_mean", "RAW_CI95", "TS_mean", "TS_CI95", "Relative_drop"])
        w.writerows(table)

    logger.info("Saved CV metrics summary with 95%% CI to %s",
                os.path.join(args.output_folder, "cv_metrics_summary.csv"))

    # ========== NEW: global summary for multi-calibration ==========
    multi_path = os.path.join(args.output_folder, "cv_metrics_per_fold_multi.csv")
    if os.path.exists(multi_path):
        import pandas as pd
        dfm = pd.read_csv(multi_path)
        out_rows = []
        for m in sorted(dfm["method"].unique()):
            sub = dfm[dfm["method"] == m]
            def _ci95(a):
                a = np.asarray([x for x in a if not np.isnan(x)], dtype=float)
                if a.size == 0:
                    return (float("nan"), float("nan"))
                mean = float(a.mean())
                hw = 1.96 * float(a.std(ddof=1)) / max(len(a), 1)**0.5 if len(a) > 1 else 0.0
                return (mean, hw)
            for name in ["ECE","NLL","Brier","Acc","AUC"]:
                mean, hw = _ci95(sub[name])
                out_rows.append([m, name, mean, hw])

        path_sum_multi = os.path.join(args.output_folder, "cv_metrics_summary_multi.csv")
        with open(path_sum_multi, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Method","Metric","Mean","CI95"])
            w.writerows(out_rows)
        logging.info("Saved multi-calibration summary to %s", path_sum_multi)



    # Keep scatter consistent with features used during CV (complex_stats)
    X_full, y_full = load_mitbih_data(args.data_folder, record_names, WINDOW_SIZE, PRE_SAMPLES, FS)
    save_complex_pca_scatter(
        prepare_complex_input(X_full, method='complex_stats'),
        y_full, args.output_folder
    )
    
    # Retrain on full dataset with proper train/val/test split
    logger.info("Retraining on full dataset…")
    X_full, y_full = load_mitbih_data(args.data_folder, record_names, WINDOW_SIZE, PRE_SAMPLES, FS)
    X_full_c = prepare_complex_input(X_full, method='complex_stats')

    # First split: TrainVal/Test (80/20)
    X_trv, X_te, y_trv, y_te = train_test_split(
        X_full_c, y_full, test_size=0.2, stratify=y_full, random_state=args.seed + 1
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_trv, y_trv, test_size=0.125, stratify=y_trv, random_state=args.seed + 2
    )
    
    tr_all, va_all, te_all, scaler_full = create_train_val_test_loaders(
        X_tr, y_tr, X_va, y_va, X_te, y_te,
        batch_size=args.batch_size,
        seed=args.seed
    )
    with open(os.path.join(args.output_folder, "scaler_full.pkl"), "wb") as f:
        pickle.dump(scaler_full, f)


    model_all = SimpleComplexNet(
        in_features=X_tr.shape[1] // 2, hidden_features=64, out_features=2, bias=0.1
    )
    hist_all, best_all = train_real(model_all, tr_all, va_all, epochs=args.epochs, lr=args.lr, device=device)
    torch.save(best_all, os.path.join(args.output_folder, 'best_model_full.pt'))
    save_plots(hist_all, args.output_folder, 'full')
    logger.info('Saved full-model and plots')

    # Temperature calibration on VALIDATION

    model_all.load_state_dict(best_all)
    T_full = None
    iso_full = None
    is_binary = False
    if args.calibration == "temperature":
        T_full = tune_temperature(model_all, va_all, device=device)
        torch.save(T_full, os.path.join(args.output_folder, 'T_calib.pt'))
        logger.info(f"[INFO] Full-model temperature T={T_full.item():.3f} saved to {args.output_folder}")
    elif args.calibration == "isotonic":
        iso_full, is_binary = fit_isotonic_on_val(model_all, va_all, device=device)
        logger.info("[INFO] Full-model isotonic calibration %s", "enabled" if is_binary else "skipped (multi-class)")
    else:
        logger.info("[INFO] No calibration ('none') on full model.")

    # Collect calibrated probabilities on VALIDATION (needed to pick tau*, delta*)
    yva, pva, Xva = [], [], []
    model_all.eval()
    with torch.no_grad():
        for xb, yb in va_all:
            logits = complex_modulus_to_logits(model_all(xb.to(device)))
            probs_raw = nn.Softmax(dim=1)(logits)
            if args.calibration == "temperature" and T_full is not None:
                probs = nn.Softmax(dim=1)(logits / T_full.to(device))
            elif args.calibration == "isotonic" and iso_full is not None:
                probs = torch.from_numpy(iso_full(probs_raw.cpu().numpy()))
            else:
                probs = probs_raw
            yva.extend(yb.numpy().tolist())
            pva.extend(probs.cpu().numpy().tolist())
            Xva.extend(xb.cpu().numpy().tolist())


    yva = np.array(yva); pva = np.array(pva); Xva = np.array(Xva)

    kink_star = float('nan')  # set later if grid was computed
    if args.sensitivity:
        # Denser grid around small deltas (most action happens near 0); wider range overall
        taus   = np.linspace(0.50, 0.95, 10)
        deltas = np.linspace(0.00, 0.60, 13)

        sens = sensitivity_analysis(
            y_true=yva, probs=pva, X=Xva,
            taus=taus, deltas=deltas,
            target_capture=args.capture_target,
            max_abstain=args.max_abstain,
            target_risk=args.target_risk,
            mode=args.select_mode
        )
        grid = sens["grid"]

        # "Knee" score proxy: normalized (capture - abstain)
        abst = grid[:, 2]; capt = grid[:, 3]
        a = (abst - abst.min()) / (abst.max() - abst.min() + 1e-12)
        c = (capt - capt.min()) / (capt.max() - capt.min() + 1e-12)
        kink = c - a

        # Save the full grid with knee score
        sens_csv = os.path.join(args.output_folder, "sens_grid.csv")
        with open(sens_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tau","delta","abstain","capture","precision","dispersion","risk_accept","kink"])
            for row, k in zip(grid, kink):
                w.writerow([f"{row[0]:.6f}", f"{row[1]:.6f}"] + [f"{x:.6f}" for x in row[2:]] + [f"{float(k):.6f}"])
        logging.info("Saved sensitivity grid (with kink) to %s", sens_csv)

        # Heatmaps
        save_sensitivity_heatmaps(grid, args.output_folder, prefix="sens_full")

    # --- Choose (tau, delta) on VALIDATION of the full split using exact-count budget ---
    chosen = select_thresholds_budget_count(
        y_true=yva, probs=pva, X=Xva, budget_count=args.review_budget
    )
    tau_star   = float(chosen['tau'])
    delta_star = float(chosen['delta'])

    # Extract knee score for (tau*, delta*) if grid was computed
    kink_star = float('nan')
    if args.sensitivity:
        mask = (np.isclose(grid[:,0], tau_star)) & (np.isclose(grid[:,1], delta_star))
        if 'kink' in locals() and mask.any():
            kink_star = float(kink[mask][0])

    with open(os.path.join(args.output_folder, "sens_full.csv"), "w") as f:
        f.write("tau,delta,abstain,capture,precision,dispersion,risk_accept,kink_score\n")
        f.write(f"{tau_star},{delta_star},{chosen['abstain']},{chosen['capture']},"
                f"{chosen['precision']},{chosen['dispersion']},{chosen['risk_accept']},{kink_star}\n")

    logger.info(
        f"[FULL] chosen (tau*, delta*) = ({tau_star:.6f}, {delta_star:.6f}); "
        f"val_abstain≈{chosen['abstain']:.6f} "
        f"({int(round(chosen['abstain']*len(yva)))} samples), "
        f"capture≈{chosen['capture']:.3f}, risk_accept≈{chosen['risk_accept']:.3f}"
    )



    # --- Uncertainty histogram on TEST set (uses tau_star picked on VAL) ---
    y_conf_test = []
    y_margin_test = []   # NEW
    model_all.eval()
    with torch.no_grad():
        for xb, yb in te_all:
            logits = complex_modulus_to_logits(model_all(xb.to(device)))
            probs_raw = nn.Softmax(dim=1)(logits)
            if args.calibration == "temperature" and T_full is not None:
                probs = nn.Softmax(dim=1)(logits / T_full.to(device))
            elif args.calibration == "isotonic" and iso_full is not None:
                probs = torch.from_numpy(iso_full(probs_raw.cpu().numpy()))
            else:
                probs = probs_raw

            y_conf_test.extend(probs.max(dim=1).values.cpu().tolist())

            top2 = torch.topk(probs, k=2, dim=1).values
            y_margin_test.extend((top2[:, 0] - top2[:, 1]).abs().cpu().tolist())

    label = None
    if tau_star <= 1e-9:
        label = "tau* not used — only the condition margin < delta* is active"
    save_uncertainty_hist(y_conf_test, float(tau_star), args.output_folder, label_override=label)

    # NEW: margin histogram with a vertical line at delta*
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(y_margin_test, bins=30, alpha=0.7)
    ax.axvline(float(delta_star), linestyle="--", color="red", alpha=0.8, label=f"delta* = {delta_star:.4f}")
    cal_suffix = "TS" if args.calibration == "temperature" else ("ISO" if args.calibration == "isotonic" else "CAL")
    ax.set_xlabel(f"Margin |p1 - p2| ({cal_suffix})")
    ax.set_ylabel("Number of samples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_folder, "uncertainty_margin_hist.png"), dpi=300)
    plt.close(fig)
    logging.info("Saved uncertainty margin histogram")

    # --- Detect uncertain points on TEST (using the chosen tau_star & delta_star) ---
    X_te_scaled_t, y_te_t = te_all.dataset.tensors
    X_te_scaled_t = X_te_scaled_t.cpu()
    y_te_t        = y_te_t.cpu()
    temp_arg = T_full if (args.calibration == "temperature" and T_full is not None) else None
    uncertain_full = find_uncertain_points(
        model_all.to('cpu'),
        X_te_scaled_t,
        y_te_t,
        prob_thr=float(tau_star),
        margin_delta=float(delta_star),
        temperature=temp_arg
    )

    logger.info(f"[FULL] Uncertain flagged on TEST: {len(uncertain_full)} samples.")

    # --- Save uncertain points CSV ---
    csv_path = os.path.join(args.output_folder, 'uncertain_full.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index','X','true_label','p1','p2'])
        writer.writeheader()
        for u in uncertain_full:
            p1, p2 = u['prob']
            writer.writerow({
                'index':      u['index'],
                'X':          u['X'],
                'true_label': u['true_label'],
                'p1':         p1,
                'p2':         p2
            })


    logger.info(f"[INFO] Saved full-model uncertain points to {csv_path}")
    # Make sure all logs are flushed to disk
    logging.shutdown()
    
if __name__ == "__main__":
    main()
