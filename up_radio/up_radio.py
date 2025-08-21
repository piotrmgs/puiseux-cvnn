# Copyright (c) 2025 Piotr Migus
# This code is licensed under the MIT License.
# See the LICENSE file in the repository root for full license information.

"""
Train & evaluate a Complex-Valued Neural Network (CVNN) on RadioML 2016.10a.

What this script does
---------------------
- Loads a filtered subset of RadioML 2016.10a (by modulation classes and SNR).
- Transforms raw IQ windows into compact complex-aware features (STFT statistics).
- Runs Stratified K-Fold CV on samples and trains a SimpleComplexNet per fold.
- Calibrates probabilities on the validation split (temperature or isotonic; plus
  optional multi-calibration sweep via `get_calibrator`).
- Logs per-fold metrics (ECE/NLL/Brier/Acc/AUC) and saves confusion & ROC plots.
- Aggregates CV learning curves and reliability diagrams (RAW vs calibrated).
- (Optional) Sensitivity analysis for (tau, delta) thresholds with heatmaps.
- Full retrain on Train/Val/Test, threshold selection on Val, uncertainty plots.
- Exports predictions, uncertain samples, and metric summaries as CSVs.

Typical usage
-------------
$ python up_radio.py \
    --data_folder ./radio-data --output_folder ./results_radio \
    --mods BPSK QPSK --snr_low 5 --snr_high 15 \
    --epochs 20 --folds 10 --calibration temperature
"""

import os
import argparse
import logging
import pickle
import csv
import time
import psutil
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

from src.find_up_radio import load_rml2016_data
from src.find_up_synthetic import SimpleComplexNet, complex_modulus_to_logits, find_uncertain_points

from src.find_up_real import (
    prepare_complex_input,
    create_train_val_test_loaders,
    train_model as train_real,
    tune_temperature,
    fit_isotonic_on_val,
    save_plots,
    save_confusion_roc,
    expected_calibration_error,
    save_overall_history,
    save_calibration_curve,
    save_uncertainty_hist,
    save_complex_pca_scatter,
    select_thresholds_budget_count,
    sensitivity_analysis,
    save_sensitivity_heatmaps,
    negative_log_likelihood,
    brier_score,
    mean_ci,
    seed_everything,
    get_calibrator,
    WINDOW_SIZE,
)

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments controlling dataset filters, training, calibration, and analysis."""
    parser = argparse.ArgumentParser(description="CVNN on RadioML 2016.10a")
    parser.add_argument("--data_folder", type=str, default="radio-data")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--output_folder", type=str, default=script_dir)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--folds", type=int, default=10)

    parser.add_argument("--mods", nargs="+", default=["BPSK", "QPSK"])
    parser.add_argument("--snr_low", type=int, default=5)
    parser.add_argument("--snr_high", type=int, default=15)

    parser.add_argument("--threshold", type=float, default=0.5001)
    parser.add_argument("--sensitivity", action="store_true", default=True,
                        help="Enable tau/delta grid analysis and heatmaps.")
    parser.add_argument("--no-sensitivity", dest="sensitivity", action="store_false")
    parser.add_argument("--capture_target", type=float, default=0.80)
    parser.add_argument("--select_mode", type=str, default="budget", choices=["capture","budget","risk","knee"])
    parser.add_argument("--max_abstain", type=float, default=0.20)
    parser.add_argument("--target_risk", type=float, default=None)
    parser.add_argument("--review_budget", type=int, default=10)

    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--calibration", type=str, default="platt",
                        choices=["temperature", "isotonic", "platt", "beta", "vector", "none"],
                        help="Calibration fitted on VALIDATION (default: platt; binary tasks).")
    parser.add_argument("--calibs", type=str,
                        default="temperature,isotonic,platt,beta,vector,none",
                        help="Comma-separated list for multi-calibration sweep on TEST.")

    # Currently not used in this script (reserved for embedding/abstention sweeps).
    parser.add_argument("--budget_fracs", type=str, default="")
    parser.add_argument("--embed_method", type=str, default="tsne", choices=["tsne","umap"])

    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()

def main() -> None:
    """Entry point: run stratified CV on RadioML, aggregate metrics/plots, retrain, and export artifacts."""
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # Logging (force=True to reset any pre-existing handlers when run repeatedly)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_folder, "run.log"), mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger = logging.getLogger()

    # Reproducibility & device selection
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    logger.info("Using device: %s | seed=%d", device, args.seed)

    # Run metadata for reproducibility
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

    # ---------- Load full RadioML subset ----------
    # Returns raw IQ windows flattened to [I | Q] and integer labels.
    X_all, y_all = load_rml2016_data(
        args.data_folder,
        mods=args.mods,
        snr_range=(args.snr_low, args.snr_high),
        window_size=WINDOW_SIZE,
    )

    # ---------- CV collectors ----------
    histories_all_folds = []
    y_true_all, y_pred_all = [], []
    y_prob_all_pos, y_conf_all_max = [], []
    y_true_all_raw, y_prob_all_pos_raw = [], []
    y_margin_all, pred_rows_all = [], []

    ece_raw_folds, ece_ts_folds = [], []
    nll_raw_folds, nll_ts_folds = [], []
    br_raw_folds,  br_ts_folds  = [], []
    acc_ts_folds, auc_ts_folds  = [], []

    # ---------- Stratified K-Fold on samples ----------
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_all, y_all), 1):
        # Split raw IQ into train/test for this fold
        X_tr_raw, y_tr_raw = X_all[tr_idx], y_all[tr_idx]
        X_te_raw, y_te_raw = X_all[te_idx], y_all[te_idx]

        # Feature engineering: STFT-based summary stats → real+imag blocks
        X_tr = prepare_complex_input(X_tr_raw, method='stft_stats')
        X_te = prepare_complex_input(X_te_raw, method='stft_stats')

        # Train/Val split (stratified) within the training portion
        X_tr_tr, X_tr_val, y_tr_tr, y_tr_val = train_test_split(
            X_tr, y_tr_raw, test_size=0.20, stratify=y_tr_raw, random_state=args.seed + 1000 + fold
        )

        # Standardize with scaler fit on TRAIN only; deterministic sampling/workers
        # NOTE: this variant returns loaders AND a fitted scaler (persisted below).
        tr_ld, val_ld, te_ld, scaler = create_train_val_test_loaders(
            X_tr_tr, y_tr_tr,
            X_tr_val, y_tr_val,
            X_te,     y_te_raw,
            batch_size=args.batch_size,
            seed=args.seed
        )
        with open(os.path.join(args.output_folder, f"scaler_fold{fold}.pkl"), "wb") as f:
            pickle.dump(scaler, f)

        # Model definition and training
        model = SimpleComplexNet(in_features=X_tr.shape[1]//2, hidden_features=64, out_features=2, bias=0.1)
        t0 = time.perf_counter()
        history, best_state = train_real(model, tr_ld, val_ld, epochs=args.epochs, lr=args.lr, device=device)
        avg_epoch_time = (time.perf_counter() - t0)/len(history["train_loss"])
        histories_all_folds.append(history)
        save_plots(history, args.output_folder, fold)

        # Resource snapshot (CPU RSS and optional GPU peak)
        try:
            proc = psutil.Process(os.getpid())
            rss_mb = proc.memory_info().rss / (1024**2)
            if device.type == "cuda":
                peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
                torch.cuda.reset_peak_memory_stats(device)
                logging.info(f"[Fold {fold}] avg_epoch_time={avg_epoch_time:.3f}s | CPU_RSS={rss_mb:.1f}MB | GPU_peak={peak_mb:.1f}MB")
            else:
                logging.info(f"[Fold {fold}] avg_epoch_time={avg_epoch_time:.3f}s | CPU_RSS={rss_mb:.1f}MB")
        except Exception as _e:
            logging.warning(f"[Fold {fold}] memory logging failed: {_e}")

        # Load best-performing weights (by validation accuracy)
        model.load_state_dict(best_state)

        # --- Calibration on VALIDATION (unified across all methods)
        # Fit chosen calibrator on the VALIDATION split once per fold.
        apply_cal_fn, cal_tag = get_calibrator(model, val_ld, method=args.calibration, device=device)
        logging.info(f"[Fold {fold}] Calibration method: {args.calibration} | tag={cal_tag}")

        # Optional: keep saving T for temperature to match previous outputs.
        if args.calibration == "temperature":
            T_fold = tune_temperature(model, val_ld, device=device)
            torch.save(T_fold, os.path.join(args.output_folder, f"T_calib_fold{fold}.pt"))
            logging.info(f"[Fold {fold}] Learned T={T_fold.item():.3f}")


        # --- Evaluate on TEST: RAW vs selected calibration (using the unified calibrator)
        model.eval()
        y_true_raw, y_prob_pos_raw, y_max_raw = [], [], []
        y_true_ts,  y_prob_pos_ts,  y_max_ts  = [], [], []
        y_pred_ts = []
        probs_ts_all = []
        yb_all = []

        with torch.no_grad():
            for xb, yb in te_ld:
                xb = xb.to(device)
                logits = complex_modulus_to_logits(model(xb))
                probs_raw = torch.softmax(logits, dim=1)

                # Apply the chosen calibrator consistently on TEST
                if apply_cal_fn is None:
                    # No applicable calibrator → use RAW
                    probs_ts = probs_raw
                elif args.calibration in ("temperature", "vector", "platt"):
                    # These expect logits in logit-space
                    probs_ts_np = apply_cal_fn(logits_np=logits.detach().cpu().numpy())
                    probs_ts = torch.from_numpy(probs_ts_np)
                else:
                    # "isotonic" and "beta" expect probabilities in prob-space
                    probs_ts_np = apply_cal_fn(probs_np=probs_raw.detach().cpu().numpy())
                    probs_ts = torch.from_numpy(probs_ts_np)

                probs_ts_all.append(probs_ts.cpu())
                yb_all.append(yb.cpu())

                # RAW collectors (for baseline reliability diagram)
                y_true_raw.extend(yb.tolist())
                y_prob_pos_raw.extend(probs_raw[:, 1].detach().cpu().tolist())
                y_max_raw.extend(probs_raw.max(dim=1).values.detach().cpu().tolist())

                # CALIBRATED collectors (for calibrated reliability diagram)
                y_true_ts.extend(yb.tolist())
                y_prob_pos_ts.extend(probs_ts[:, 1].cpu().tolist())
                y_max_ts.extend(probs_ts.max(dim=1).values.cpu().tolist())
                y_pred_ts.extend(probs_ts.argmax(dim=1).cpu().tolist())


        # Fold-level Acc/AUC for calibrated predictions
        acc_ts = float((np.array(y_true_ts) == np.array(y_pred_ts)).mean())
        try:
            auc_ts = float(roc_auc_score(np.array(y_true_ts), np.array(y_prob_pos_ts)))
        except Exception:
            auc_ts = float('nan')
        acc_ts_folds.append(acc_ts)
        auc_ts_folds.append(auc_ts)

        # Confusion matrix and ROC curve per fold
        save_confusion_roc(y_true_ts, y_pred_ts, y_prob_pos_ts, args.output_folder, fold)

        # Prepare per-sample CSV rows (include margins, pmax)
        probs_ts_all = torch.cat(probs_ts_all, dim=0)
        yb_all       = torch.cat(yb_all, dim=0)
        
        # Decide a stable CAL suffix, e.g., "PLATT", to tag CSV columns
        cal_suffix = (cal_tag.upper() if cal_tag else (args.calibration.upper() if args.calibration else "NONE"))

        top2 = torch.topk(probs_ts_all, k=2, dim=1).values
        margin_batch = (top2[:,0] - top2[:,1]).abs().tolist()
        y_margin_all.extend(margin_batch)

        pmax_batch = probs_ts_all.max(dim=1).values.tolist()
        p1_batch   = probs_ts_all[:,0].tolist()
        p2_batch   = probs_ts_all[:,1].tolist()
        pred_batch = probs_ts_all.argmax(dim=1).tolist()
        true_batch = yb_all.tolist()

        base_row_id = len(pred_rows_all)
        k_pred = f"pred_CAL_{cal_suffix}"
        k_p1   = f"p1_CAL_{cal_suffix}"
        k_p2   = f"p2_CAL_{cal_suffix}"
        k_pmax = f"pmax_CAL_{cal_suffix}"
        k_mrg  = f"margin_CAL_{cal_suffix}"

        for i in range(len(true_batch)):
            pred_rows_all.append({
                "fold": fold,
                "row_global": base_row_id + i,
                "true": int(true_batch[i]),
                k_pred: int(pred_batch[i]),
                k_p1:   float(p1_batch[i]),
                k_p2:   float(p2_batch[i]),
                k_pmax: float(pmax_batch[i]),
                k_mrg:  float(margin_batch[i]),
            })


        # Global collectors for figures (TS and RAW)
        y_true_all.extend(y_true_ts)
        y_pred_all.extend(y_pred_ts)
        y_prob_all_pos.extend(y_prob_pos_ts)
        y_conf_all_max.extend(y_max_ts)
        y_true_all_raw.extend(y_true_raw)
        y_prob_all_pos_raw.extend(y_prob_pos_raw)

        # Per-fold ECE/NLL/Brier (RAW vs calibrated)
        ece_raw = expected_calibration_error(np.array(y_true_raw), np.array(y_prob_pos_raw), n_bins=10)
        ece_ts  = expected_calibration_error(np.array(y_true_ts),  np.array(y_prob_pos_ts),  n_bins=10)
        nll_raw = negative_log_likelihood(np.array(y_true_raw), np.array(y_prob_pos_raw))
        nll_ts  = negative_log_likelihood(np.array(y_true_ts),  np.array(y_prob_pos_ts))
        br_raw  = brier_score(np.array(y_true_raw), np.array(y_prob_pos_raw))
        br_ts   = brier_score(np.array(y_true_ts),  np.array(y_prob_pos_ts))

        ece_raw_folds.append(ece_raw); ece_ts_folds.append(ece_ts)
        nll_raw_folds.append(nll_raw); nll_ts_folds.append(nll_ts)
        br_raw_folds.append(br_raw);   br_ts_folds.append(br_ts)

        # ---- Multi-calibration sweep (temperature, isotonic, platt, beta, vector, none) on TEST
        methods = [m.strip().lower() for m in args.calibs.split(",") if m.strip()]

        # Cache TEST logits once
        logits_list_mc, y_list_mc = [], []
        with torch.no_grad():
            for xb, yb in te_ld:
                xb = xb.to(device)
                lo = complex_modulus_to_logits(model(xb))
                logits_list_mc.append(lo.cpu().numpy())
                y_list_mc.extend(yb.cpu().numpy().tolist())
        logits_np_mc = np.vstack(logits_list_mc)
        y_np_mc      = np.asarray(y_list_mc, dtype=int)

        # RAW probs (for iso/beta-family and for NONE baseline)
        z_mc = logits_np_mc - logits_np_mc.max(axis=1, keepdims=True)
        expz = np.exp(z_mc)
        probs_raw_np_mc = expz / expz.sum(axis=1, keepdims=True)

        fold_rows = []
        for method in methods:
            # Always include a TRUE 'NONE' baseline (identity); do not call a calibrator for it.
            if method == "none":
                tag = "NONE"
                probs_np = probs_raw_np_mc
            else:
                apply_fn, tag = get_calibrator(model, val_ld, method=method, device=device)
                if apply_fn is None:
                    logging.info(f"[Fold {fold}] Skipping method={method} (not applicable).")
                    continue

                # Use keyword args so the interface is unambiguous and robust.
                if method in ("temperature", "vector", "platt"):
                    probs_np = apply_fn(logits_np=logits_np_mc)
                elif method in ("isotonic", "beta"):
                    probs_np = apply_fn(probs_np=probs_raw_np_mc)
                else:
                    probs_np = probs_raw_np_mc  # fallback

            # Safety: ensure proper probability simplex and dtype
            probs_np = np.clip(probs_np, 1e-7, 1.0)
            probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)
            probs_np = probs_np.astype(np.float32, copy=False)

            preds = probs_np.argmax(axis=1)
            acc   = float((preds == y_np_mc).mean())
            if probs_np.shape[1] == 2:
                try:
                    auc = float(roc_auc_score(y_np_mc, probs_np[:, 1]))
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


        multi_path = os.path.join(args.output_folder, "cv_metrics_per_fold_multi.csv")
        write_header = not os.path.exists(multi_path)
        with open(multi_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["fold","method","tag","ECE","NLL","Brier","Acc","AUC"])
            if write_header:
                w.writeheader()
            w.writerows(fold_rows)
        logging.info(f"[Fold {fold}] Wrote multi-calibration rows to {multi_path}")

    # ----- After CV: persist predictions and per-fold metrics -----
    if pred_rows_all:
        pred_csv_path = os.path.join(args.output_folder, "predictions_all_folds.csv")
        with open(pred_csv_path, "w", newline="") as fcsv:
            cw = csv.DictWriter(fcsv, fieldnames=list(pred_rows_all[0].keys()))
            cw.writeheader()
            cw.writerows(pred_rows_all)
        logging.info(f"Saved predictions with margins to {pred_csv_path}")

    per_fold_path = os.path.join(args.output_folder, "cv_metrics_per_fold.csv")
    cal_hdr = (args.calibration.upper() if args.calibration else "NONE")
    with open(per_fold_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "fold",
            "ECE_RAW",  f"ECE_CAL_{cal_hdr}",
            "NLL_RAW",  f"NLL_CAL_{cal_hdr}",
            "Brier_RAW",f"Brier_CAL_{cal_hdr}",
                         f"Acc_CAL_{cal_hdr}",
                         f"AUC_CAL_{cal_hdr}",
        ])
        for i in range(len(ece_raw_folds)):
            w.writerow([i+1,
                        f"{ece_raw_folds[i]:.6f}", f"{ece_ts_folds[i]:.6f}",
                        f"{nll_raw_folds[i]:.6f}", f"{nll_ts_folds[i]:.6f}",
                        f"{br_raw_folds[i]:.6f}",  f"{br_ts_folds[i]:.6f}",
                        f"{acc_ts_folds[i]:.6f}",  f"{auc_ts_folds[i]:.6f}"])


    # Aggregate learning curves and reliability diagrams
    save_overall_history(histories_all_folds, args.output_folder)
    save_calibration_curve(y_true_all_raw, y_prob_all_pos_raw, args.output_folder, suffix="RAW")
    cal_suffix = (args.calibration.upper() if args.calibration else "NONE")
    save_calibration_curve(y_true_all, y_prob_all_pos, args.output_folder, suffix=cal_suffix)

    
    # CV summary with 95% CI half-widths
    def _row(name, raw_list, ts_list):
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
        cal_suffix = (args.calibration.upper() if args.calibration else "NONE")
        w.writerow(["Metric","RAW_mean","RAW_CI95",f"CAL_{cal_suffix}_mean",f"CAL_{cal_suffix}_CI95","Relative_drop"])
        w.writerows(table)

    logging.info("Saved CV metrics summary with 95%% CI.")

    # Optional: summarize multi-calibration results across folds
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

    # ---------- Global feature view (for parity with up_real) ----------
    save_complex_pca_scatter(prepare_complex_input(X_all, method='stft_stats'), y_all, args.output_folder)

    # ---------- Full retrain: Train/Val/Test ----------
    logging.info("Retraining on full dataset…")
    X_full_c = prepare_complex_input(X_all, method='stft_stats')
    X_trv, X_te, y_trv, y_te = train_test_split(
        X_full_c, y_all, test_size=0.2, stratify=y_all, random_state=args.seed + 1
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_trv, y_trv, test_size=0.125, stratify=y_trv, random_state=args.seed + 2
    )

    tr_all, va_all, te_all, scaler_full = create_train_val_test_loaders(
        X_tr, y_tr, X_va, y_va, X_te, y_te,
        batch_size=args.batch_size, seed=args.seed
    )
    with open(os.path.join(args.output_folder, "scaler_full.pkl"), "wb") as f:
        pickle.dump(scaler_full, f)

    model_all = SimpleComplexNet(in_features=X_tr.shape[1]//2, hidden_features=64, out_features=2, bias=0.1)
    hist_all, best_all = train_real(model_all, tr_all, va_all, epochs=args.epochs, lr=args.lr, device=device)
    torch.save(best_all, os.path.join(args.output_folder, 'best_model_full.pt'))
    save_plots(hist_all, args.output_folder, 'full')
    model_all.load_state_dict(best_all)

    # Full-model calibration on VALIDATION
    apply_full, cal_tag_full = get_calibrator(model_all, va_all, method=args.calibration, device=device)
    logging.info(f"[FULL] Calibration method: {args.calibration} | tag={cal_tag_full}")

    # Optional: keep saving T for temperature to match previous outputs.
    if args.calibration == "temperature":
        T_full = tune_temperature(model_all, va_all, device=device)
        torch.save(T_full, os.path.join(args.output_folder, 'T_calib.pt'))
        logging.info(f"[FULL] Temperature T={T_full.item():.3f} saved to {args.output_folder}")


    # Collect calibrated probs on VALIDATION for (tau, delta) selection
    yva, pva, Xva = [], [], []
    model_all.eval()
    with torch.no_grad():
        for xb, yb in va_all:
            xb_dev = xb.to(device)
            logits = complex_modulus_to_logits(model_all(xb_dev))
            probs_raw = torch.softmax(logits, dim=1)

            # Apply the chosen calibrator on VALIDATION (for selecting tau*, delta*)
            # IMPORTANT: use 'apply_full' (fitted on VAL of the full model).
            if apply_full is None:
                probs = probs_raw  # identity
            else:
                if args.calibration in ("temperature", "vector", "platt"):
                    probs_np = apply_full(logits_np=logits.detach().cpu().numpy())
                else:
                    probs_np = apply_full(probs_np=probs_raw.detach().cpu().numpy())

                # Safety: clip + renormalize + match dtype
                probs_np = np.clip(probs_np, 1e-7, 1.0)
                probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)
                probs = torch.from_numpy(probs_np.astype(np.float32, copy=False))

            # Keep everything on CPU for downstream numpy ops
            yva.extend(yb.cpu().numpy().tolist())
            pva.extend(probs.cpu().numpy().tolist())
            Xva.extend(xb.cpu().numpy().tolist())


    yva = np.array(yva); pva = np.array(pva); Xva = np.array(Xva)

    # Optionally evaluate a grid of (tau, delta) and save heatmaps/grid CSV
    if args.sensitivity:
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
        abst, capt = grid[:,2], grid[:,3]
        a = (abst - abst.min())/(abst.max() - abst.min() + 1e-12)
        c = (capt - capt.min())/(capt.max() - capt.min() + 1e-12)
        kink = c - a
        sens_csv = os.path.join(args.output_folder, "sens_grid.csv")
        with open(sens_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tau","delta","abstain","capture","precision","dispersion","risk_accept","kink"])
            for row, k in zip(grid, kink):
                w.writerow([f"{row[0]:.6f}", f"{row[1]:.6f}"] + [f"{x:.6f}" for x in row[2:]] + [f"{float(k):.6f}"])
        save_sensitivity_heatmaps(grid, args.output_folder, prefix="sens_full")

    # Select thresholds (tau*, delta*) on VALIDATION using an exact-count review budget
    chosen = select_thresholds_budget_count(y_true=yva, probs=pva, X=Xva, budget_count=args.review_budget)
    tau_star, delta_star = float(chosen['tau']), float(chosen['delta'])
    kink_star = float('nan')
    if args.sensitivity:
        mask = (np.isclose(grid[:,0], tau_star)) & (np.isclose(grid[:,1], delta_star))
        if 'kink' in locals() and mask.any():
            kink_star = float(kink[mask][0])

    with open(os.path.join(args.output_folder, "sens_full.csv"), "w") as f:
        f.write("tau,delta,abstain,capture,precision,dispersion,risk_accept,kink_score\n")
        f.write(f"{tau_star},{delta_star},{chosen['abstain']},{chosen['capture']},"
                f"{chosen['precision']},{chosen['dispersion']},{chosen['risk_accept']},{kink_star}\n")

    # Histograms on TEST: p_max (with tau*) and margin (with delta*)
    y_conf_test, y_margin_test = [], []
    with torch.no_grad():
        for xb, yb in te_all:
            xb_dev = xb.to(device)
            logits = complex_modulus_to_logits(model_all(xb_dev))
            probs_raw = torch.softmax(logits, dim=1)

            # Apply the same calibrator on TEST
            if apply_full is None:
                probs = probs_raw
            else:
                if args.calibration in ("temperature", "vector", "platt"):
                    probs_np = apply_full(logits_np=logits.detach().cpu().numpy())
                else:
                    probs_np = apply_full(probs_np=probs_raw.detach().cpu().numpy())
                probs_np = np.clip(probs_np, 1e-7, 1.0)
                probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)
                probs = torch.from_numpy(probs_np.astype(np.float32, copy=False))


            y_conf_test.extend(probs.max(dim=1).values.cpu().tolist())
            top2 = torch.topk(probs, k=2, dim=1).values
            y_margin_test.extend((top2[:, 0] - top2[:, 1]).abs().cpu().tolist())

    label = None
    if tau_star <= 1e-9:
        label = "tau* not used — only the condition margin < delta* is active"
    save_uncertainty_hist(y_conf_test, float(tau_star), args.output_folder, label_override=label)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5,3))
    ax.hist(y_margin_test, bins=30, alpha=0.7)
    ax.axvline(float(delta_star), linestyle="--", color="red", alpha=0.8, label=f"delta* = {delta_star:.4f}")
    cal_suffix = (args.calibration.upper() if args.calibration else "NONE")
    ax.set_xlabel(f"Margin |p1 - p2| ({cal_suffix})")
    ax.set_ylabel("Number of samples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_folder, "uncertainty_margin_hist.png"), dpi=300)
    plt.close(fig)

    # Detect uncertain points on TEST and save CSV
    X_te_scaled_t, y_te_t = te_all.dataset.tensors   # already scaled by the train-only scaler
    X_te_scaled_t = X_te_scaled_t.cpu()
    y_te_t        = y_te_t.cpu()

    # We compute calibrated probs per-sample and flag by (pmax < tau*) OR (margin < delta*).
    uncertain_rows = []
    model_all.eval()
    with torch.no_grad():
        for idx in range(X_te_scaled_t.shape[0]):
            xb = X_te_scaled_t[idx:idx+1].to(device)
            logit = complex_modulus_to_logits(model_all(xb))
            prob_raw = torch.softmax(logit, dim=1)

            # Calibrate
            if apply_full is None:
                prob = prob_raw
            else:
                if args.calibration in ("temperature", "vector", "platt"):
                    prob_np = apply_full(logits_np=logit.detach().cpu().numpy())
                else:
                    prob_np = apply_full(probs_np=prob_raw.detach().cpu().numpy())
                prob_np = np.clip(prob_np, 1e-7, 1.0)
                prob_np = prob_np / prob_np.sum(axis=1, keepdims=True)
                prob = torch.from_numpy(prob_np.astype(np.float32, copy=False))


            # Uncertainty conditions
            p = prob.squeeze(0).cpu().numpy()
            pmax = float(p.max())
            top2 = np.partition(p, -2)[-2:]
            margin = float(abs(top2[1] - top2[0]))

            if (pmax < float(tau_star)) or (margin < float(delta_star)):
                # Store the original scaled feature row for completeness
                uncertain_rows.append({
                    'index':      int(idx),
                    'X':          X_te_scaled_t[idx].tolist(),
                    'true_label': int(y_te_t[idx].item()),
                    'p1':         float(p[0]),
                    'p2':         float(p[1]),
                })

    csv_path = os.path.join(args.output_folder, 'uncertain_full.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index','X','true_label','p1','p2'])
        writer.writeheader()
        writer.writerows(uncertain_rows)
    logging.info(f"[INFO] Saved full-model uncertain points to {csv_path}")

    logging.shutdown()

if __name__ == "__main__":
    main()
