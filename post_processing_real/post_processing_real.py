# Copyright (c) 2025 Piotr Migus
# This code is licensed under the MIT License.
# See the LICENSE file in the repository root for full license information.
"""
Post-processing for MIT-BIH experiment:
- Loads trained CVNN and uncertain anchors.
- Fits local complex polynomial surrogates with robustness safeguards.
- Computes Puiseux expansions + interpretation.
- Robustness along adversarial directions.
- LIME & SHAP explanations with optional temperature scaling.
- Sensitivity analysis summary (tau, delta).
- Comparative calibration table with 95% CI (+ Wilcoxon & win-rate).
- Resource benchmark: Puiseux vs gradient saliency.
"""

import os
import csv
import random
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import shap
import sympy
import logging
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from lime.lime_tabular import LimeTabularExplainer

# =========================
# Model & data
# =========================
from src.find_up_synthetic import SimpleComplexNet, complex_modulus_to_logits  # noqa: F401
from mit_bih_pre.pre_pro import load_mitbih_data  # constants are imported below from find_up_real

# CLI & record list (robust import)
# Prefer importing via 'src'; keep fallbacks to avoid breaking environments.
try:
    from src.up_real import record_names, parse_args
except Exception:
    try:
        from up_real.up_real import record_names, parse_args
    except Exception:
        # Final fallback: define a minimal parser and a record list.
        import argparse
        record_names = [
            "100","101","102","103","104","105","106","107","108","109","111","112","113","114","115","116","117","118","119","121",
            "122","123","124","200","201","202","203","205","207","208","209","210","212","213","214","215","217","219","220","221",
            "222","223","228","230","231","232","233","234",
        ]
        def parse_args():
            p = argparse.ArgumentParser()
            p.add_argument("--data_folder", type=str, default="mit-bih")
            p.add_argument("--output_folder", type=str, default=os.path.dirname(os.path.abspath(__file__)))
            p.add_argument("--seed", type=int, default=12345)
            return p.parse_args()

# Local surrogate & utilities
from src.local_analysis import (
    local_poly_approx_complex,
    puiseux_uncertain_point,
    load_uncertain_points,
    evaluate_poly_approx_quality,
    estimate_nonholomorphic_fraction,      # NEW
    benchmark_local_poly_approx_and_puiseux  # NEW
)
from src.puiseux import puiseux_expansions  # for benchmark timing
from src.find_up_real import compress_to_C2, WINDOW_SIZE, PRE_SAMPLES, FS  # constants + C^2 compression

# Post-processing helpers (accept temperature T where relevant)
from src.post_processing import (
    interpret_puiseux_expansions,
    find_adversarial_directions,
    test_adversarial_impact,
    model_to_explain,
    compute_lime_explanation,
    compute_shap_explanation,
    plot_local_contour_2d,
    plot_robustness_curve,
    scalarize,
    time_gradient_saliency,
    ece_binary, 
    brier_binary, 
    nll_binary, 
    compute_binary_metrics,
    fit_binary_calibrator, 
    mean_ci_t, 
    format_mean_ci,
    kink_score
)


def mean_ci95(values):
    """Return (mean, 95% CI) for a list of floats; handles n=0/1."""
    vals = [float(v) for v in values if v is not None and str(v) != "nan"]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(vals) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    sd = var ** 0.5
    ci = 1.96 * sd / (n ** 0.5)
    return m, ci

def _normalize_shap_output(shap_values):
    """
    Always return a list [arr_class0, arr_class1], each of shape [M, P].

    Supports variants returned by different SHAP versions:
    - a list of arrays (use first two),
    - a single 2D array (sum-to-zero convention) -> reconstruct the second class via negation,
    - a 3D array (K, M, P) -> take [0] and [1].

    If shape or rank is unexpected, returns [arr_2d, None] where arr_2d is coerced to shape (1, P).
    """
    import numpy as np
    if isinstance(shap_values, list):
        if len(shap_values) >= 2:
            return [np.asarray(shap_values[0]), np.asarray(shap_values[1])]
        # Single element on the list — treat as sum-to-zero.
        a0 = np.asarray(shap_values[0])
        return [a0, -a0]
    arr = np.asarray(shap_values)
    if arr.ndim == 3 and arr.shape[0] >= 2:
        return [arr[0], arr[1]]
    elif arr.ndim == 2:
        return [arr, -arr]
    else:
        # Fallback: coerce to (1, P); second class unknown.
        arr = arr.reshape(1, -1)
        return [arr, None]



########################################################################
# MAIN SCRIPT
########################################################################
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # 0) Parse args, determinism & device
    # ----------------------------------------------------------------------
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # Determinism and numerical stability.
    random.seed(getattr(args, "seed", 12345))
    np.random.seed(getattr(args, "seed", 12345))
    torch.manual_seed(getattr(args, "seed", 12345))
    torch.cuda.manual_seed_all(getattr(args, "seed", 12345))
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

    # Script directory (artifacts will be SAVED here).
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUT_DIR = SCRIPT_DIR
    os.makedirs(OUT_DIR, exist_ok=True)

    # Input directory (model, anchors CSV, sensitivity grid, CV metrics, scaler).
    IN_DIR = os.path.abspath(getattr(args, "output_folder", os.path.join(SCRIPT_DIR, "..", "up_real")))

    # == Logging: to file + console ==
    log_path = os.path.join(OUT_DIR, "post_processing_real.log")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"),
                  logging.StreamHandler()]
    )
    logger = logging.getLogger("pp_real")
    logger.info("Log file: %s", log_path)

    # ----------------------------------------------------------------------
    # 1) Load the trained model, temperature and scaler
    # ----------------------------------------------------------------------
    model_path = os.path.join(IN_DIR, 'best_model_full.pt')
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = SimpleComplexNet(
        in_features=2,
        hidden_features=64,
        out_features=2,
        bias=0.1
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    T_path = os.path.join(IN_DIR, 'T_calib.pt')
    if os.path.isfile(T_path):
        T = torch.load(T_path, map_location=device)
        print(f"[INFO] Loaded temperature scaling T={T.item():.3f}")
    else:
        T = None
        print("[WARN] No temperature file – model will be uncalibrated")

    # Scaler (keeps feature space consistent for LIME/SHAP and CV).
    scaler_path = os.path.join(IN_DIR, "scaler_full.pkl")
    if os.path.isfile(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler_full = pickle.load(f)
        print("[INFO] Loaded scaler_full.pkl")
    else:
        scaler_full = None
        print("[WARN] scaler_full.pkl not found – explanations/CV will run on unscaled features.")

    model.eval()
    print(f"[INFO] Loaded MIT-BIH parameters from {model_path}")

    # ----------------------------------------------------------------------
    # 2) Load uncertain points & background data for LIME/SHAP
    # ----------------------------------------------------------------------
    unc_csv = os.path.join(IN_DIR, 'uncertain_full.csv')
    if not os.path.isfile(unc_csv):
        raise FileNotFoundError(f"CSV with uncertain points not found: {unc_csv}")
    up_list = load_uncertain_points(unc_csv)
    print(f"[INFO] Loaded {len(up_list)} uncertain points from {unc_csv}")

    # Training data for background (LIME/SHAP).
    X_full, y_full = load_mitbih_data(
        args.data_folder,
        record_names,
        WINDOW_SIZE,
        PRE_SAMPLES,
        FS
    )
    bg_size = min(512, len(X_full))
    idx_bg = np.random.choice(len(X_full), size=bg_size, replace=False)
    X_train_full = X_full[idx_bg]
    X_train_c2 = compress_to_C2(X_train_full)  # shape (B, 4)
    X_train = scaler_full.transform(X_train_c2) if scaler_full is not None else X_train_c2

    # Symbols for expansions.
    x_sym, y_sym = sympy.symbols('x y')

    # ----------------------------------------------------------------------
    # 3) Sensitivity (tau, delta) summary if available
    # ----------------------------------------------------------------------
    sens_grid = os.path.join(IN_DIR, 'sens_grid.csv')
    if os.path.isfile(sens_grid):
        rows = []
        with open(sens_grid, 'r', newline='') as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                try:
                    tau = float(r.get('tau', 'nan'))
                    delta = float(r.get('delta', 'nan'))
                    abstain = float(r.get('abstain', 'nan'))
                    capture = float(r.get('capture', 'nan'))
                    precision = float(r.get('precision', 'nan'))
                    dispersion = float(r.get('dispersion', 'nan'))
                    risk = float(r.get('risk_accept', 'nan'))
                    kink = float(r.get('kink', 'nan'))
                    rows.append((tau, delta, abstain, capture, precision, dispersion, risk, kink))
                except Exception:
                    continue
        R = np.array(rows, dtype=float) if rows else np.empty((0, 8))
        
        # Additional description: number of anchors, dispersion, and their "value" vs (tau, delta).
        # (Assumes 'dispersion' and 'capture' are present; else NaN.)
        try:
            n_anchors = int(np.nansum(R[:, 3] > 0))  # anchor counted if capture>0
        except Exception:
            n_anchors = int(np.sum(~np.isnan(R[:, 0])))

        sens_table_csv = os.path.join(OUT_DIR, "sensitivity_detailed.csv")
        with open(sens_table_csv, "w", newline="") as ftab:
            w = csv.writer(ftab)
            w.writerow(["tau","delta","abstain","capture","precision","dispersion","risk_accept","kink"])
            for r in rows:
                w.writerow(list(r))

        with open(os.path.join(OUT_DIR, "sensitivity_summary.txt"), "a") as f:
            f.write(f"Anchors (count proxy) : {n_anchors}\n")
            if R.size:
                f.write(f"dispersion median     : {np.nanmedian(R[:,5]):.3f}\n")
                f.write(f"capture median        : {np.nanmedian(R[:,3]):.3f}\n")

        
        if R.size:
            cap = R[:, 3]
            disp = R[:, 5]
            if np.std(cap) > 0 and np.std(disp) > 0:
                corr_cap_disp = float(np.corrcoef(cap, disp)[0, 1])
            else:
                corr_cap_disp = float('nan')
            # Best under abstain <= 0.20 (example budget).
            mask_budget = R[:, 2] <= min(0.20, np.nanmax(R[:, 2])) if R.size else np.zeros((), dtype=bool)
            best_idx = int(np.argmax(R[mask_budget, 3])) if np.any(mask_budget) else -1
            best_tuple = R[mask_budget][best_idx].tolist() if best_idx >= 0 else []

            sens_summary = os.path.join(OUT_DIR, "sensitivity_summary.txt")
            with open(sens_summary, "w") as f:
                f.write("=== Sensitivity (tau, delta) summary ===\n")
                f.write(f"Grid points: {len(R)}\n")
                f.write(f"Corr(capture, dispersion): {corr_cap_disp:.3f}\n")
                if best_tuple:
                    f.write("Best under abstain<=0.20: ")
                    f.write(f"(tau={best_tuple[0]:.3f}, delta={best_tuple[1]:.3f}) | ")
                    f.write(f"abstain={best_tuple[2]:.3f}, capture={best_tuple[3]:.3f}, ")
                    f.write(f"precision={best_tuple[4]:.3f}, dispersion={best_tuple[5]:.3f}, risk={best_tuple[6]:.3f}\n")

            # Additional summary of ranges/medians.
            with open(os.path.join(OUT_DIR, "sensitivity_extra.txt"), "w") as f:
                f.write(f"abstain range    : [{np.nanmin(R[:,2]):.3f}, {np.nanmax(R[:,2]):.3f}]\n")
                f.write(f"capture  range   : [{np.nanmin(R[:,3]):.3f}, {np.nanmax(R[:,3]):.3f}]\n")
                f.write(f"precision median : {np.nanmedian(R[:,4]):.3f}\n")
                f.write(f"risk_accept median: {np.nanmedian(R[:,6]):.3f}\n")

            print(f"[INFO] Sensitivity summary saved to {sens_summary}")
        else:
            print("[WARN] sens_grid.csv has no usable rows; skipping sensitivity summary.")
    else:
        print("[WARN] Sensitivity grid not found.")

    # ----------------------------------------------------------------------
    # 4) Comparative calibration table with 95% CI (+ Wilcoxon, win-rate)
    # ----------------------------------------------------------------------
    cv_multi = os.path.join(IN_DIR, 'cv_metrics_per_fold_multi.csv')
    if os.path.isfile(cv_multi):
        by_method = {}
        with open(cv_multi, 'r', newline='') as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                m = r.get("method", "UNKNOWN")
                by_method.setdefault(m, {"ECE": [], "NLL": [], "Brier": [], "Acc": [], "AUC": []})
                for k in by_method[m].keys():
                    try:
                        by_method[m][k].append(float(r.get(k, "nan")))
                    except Exception:
                        pass
        comp_path = os.path.join(OUT_DIR, "comparative_table.csv")
        with open(comp_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Method", "Metric", "Mean", "CI95"])
            for m in sorted(by_method.keys()):
                for metric in ["ECE", "NLL", "Brier", "Acc", "AUC"]:
                    mean, ci = mean_ci95(by_method[m][metric])
                    w.writerow([m, metric, f"{mean:.6f}", f"{ci:.6f}"])
        print(f"[INFO] Comparative table saved to {comp_path}")

        # Relative ECE reduction vs NONE (if present).
        base_ece_list = by_method.get("NONE", {}).get("ECE", [])
        temp_ece_list = by_method.get("TEMPERATURE", {}).get("ECE", [])
        if base_ece_list and temp_ece_list:
            base_ece_mean, _ = mean_ci95(base_ece_list)
            temp_ece_mean, _ = mean_ci95(temp_ece_list)
            rel_drop = (base_ece_mean - temp_ece_mean) / max(base_ece_mean, 1e-12)
            with open(os.path.join(OUT_DIR, "calibration_ci_report.txt"), "w") as f:
                f.write(f"ECE(base=NONE)       = {base_ece_mean:.6f}\n")
                f.write(f"ECE(TEMPERATURE)     = {temp_ece_mean:.6f}\n")
                f.write(f"Relative drop (ECE)  = {100*rel_drop:.2f}%\n")

        # --- Statistical comparisons & win-rate ---
        try:
            import itertools
            from scipy.stats import wilcoxon
            dfm = pd.read_csv(cv_multi)
            methods_present = sorted(dfm["method"].unique())
            with open(os.path.join(OUT_DIR, "calibration_stats_tests.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["Metric","MethodA","MethodB","p_value"])
                for metric in ["ECE","NLL","Brier","Acc","AUC"]:
                    for a,b in itertools.combinations(methods_present, 2):
                        sa = dfm[dfm["method"]==a][metric].astype(float).values
                        sb = dfm[dfm["method"]==b][metric].astype(float).values
                        n = min(len(sa), len(sb))
                        if n >= 5:
                            # For metrics with "lower is better" run one-sided 'less', else 'greater'.
                            alt = "less" if metric in ["ECE","NLL","Brier"] else "greater"
                            diff = sa[:n] - sb[:n]
                            if np.allclose(diff, 0.0):
                                p = 1.0  # identical distributions -> no difference
                            else:
                                with np.errstate(invalid='ignore', divide='ignore'):
                                    p = wilcoxon(sa[:n], sb[:n], alternative=alt).pvalue
                            w.writerow([metric, a, b, f"{p:.3e}"])
            base = dfm[dfm["method"]=="NONE"][["fold","ECE"]].rename(columns={"ECE":"ECE_NONE"})
            rows_wr = []
            for m in methods_present:
                if m == "NONE": 
                    continue
                sub = dfm[dfm["method"]==m][["fold","ECE"]].rename(columns={"ECE":f"ECE_{m}"})
                j = base.merge(sub, on="fold", how="inner")
                wins = int((j[f"ECE_{m}"] < j["ECE_NONE"]).sum())
                rows_wr.append([m, wins, int(j.shape[0])])
            with open(os.path.join(OUT_DIR, "calibration_winrate_vs_none.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["Method","Wins_vs_NONE","N_folds"])
                w.writerows(rows_wr)
        except Exception as e:
            logger.warning("Statistical comparison failed: %s", e)

    else:
        print("[WARN] cv_metrics_per_fold_multi.csv not found; skipping comparative CI table.")

    # ----------------------------------------------------------------------
    # 5) Process each uncertain point
    # ----------------------------------------------------------------------
    # Collectors for aggregate reports.
    kink_rows = []
    res_rows = []
    fit_rows = []
    dom_rows = []

    for i, up in enumerate(up_list):
        print(f"\n=== POINT # {i} ===")
        print("[DATA]", up)

        # Base anchor in R^4 = (Re(z1), Re(z2), Im(z1), Im(z2))
        xstar = np.array(up['X'], dtype=np.float32)

        # Non-holomorphicity heuristic.
        try:
            ks = kink_score(model, xstar, radius=0.01, samples=24, device=device)
            logger.info("[KINK] angular-std=%.3f rad (%s)", ks,
                        "suspected non-holomorphic sector" if ks > 0.5 else "smooth")
        except Exception as e:
            logger.warning("kink_score failed: %s", e)

        # --- Fraction of 'kink' (modReLU) in the neighborhood ---
        kdiag = estimate_nonholomorphic_fraction(
            model, xstar, delta=0.05, n_samples=2000, kink_eps=1e-6, device=device
        )
        
        # --- sweep over kink_eps (from very small to 1e-2) ---
        kink_sweep_eps = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
        rows_k = []
        for eps in kink_sweep_eps:
            ksweep = estimate_nonholomorphic_fraction(
                model, xstar, delta=0.05, n_samples=2000, kink_eps=eps, device=device
            )
            rows_k.append([i, float(eps), ksweep['frac_kink'], ksweep['frac_active'], ksweep['frac_inactive']])

        pd.DataFrame(rows_k, columns=["point","kink_eps","frac_kink","frac_active","frac_inactive"]).to_csv(
            os.path.join(OUT_DIR, f'kink_sweep_point{i}.csv'), index=False
        )

        print(f"[KINK] frac_kink={kdiag['frac_kink']:.3f} | active={kdiag['frac_active']:.3f} "
              f"| inactive={kdiag['frac_inactive']:.3f} | n={kdiag['n_samples']}")

        # ------------------------------------------------------------------
        # (A) Local polynomial approximation (robust, with kink filter/weights)
        # ------------------------------------------------------------------
        F_expr_zero, fit_diag = local_poly_approx_complex(
            model, xstar,
            delta=0.05, degree=4, n_samples=600,
            device=device, remove_linear=True,
            exclude_kink_eps=1e-6, weight_by_distance=True,
            return_diag=True
        )
        print("[FIT] kept={}/{} ({:.1f}%), cond={:.2e}, rank={}/{}".format(
            fit_diag['n_kept'], fit_diag['n_total'], 100 * fit_diag['kept_ratio'],
            fit_diag['cond'], fit_diag['rank'], fit_diag['n_monomials']
        ))

        # Evaluate local surrogate fidelity.
        metrics = evaluate_poly_approx_quality(
            model=model,
            poly_expr=F_expr_zero,
            xstar=xstar,
            delta=0.05,
            n_samples=200,
            device=device
        )
        print("[Approx Quality] RMSE={:.3f}, MAE={:.3f}, corr={:.3f}, sign_agree={:.3f}".format(
            metrics['RMSE'], metrics['MAE'], metrics['corr_pearson'], metrics['sign_agreement']
        ))
        if metrics['corr_pearson'] < 0.2:
            print("[WARNING] Low Pearson correlation detected (%.3f). "
                  "Local approximation may be imprecise; consider increasing samples/degree."
                  % metrics['corr_pearson'])

        # ------------------------------------------------------------------
        # (B) Puiseux expansions + interpretation
        # ------------------------------------------------------------------
        expansions_np = puiseux_uncertain_point(F_expr_zero, prec=4, base_point=xstar)
        interpret_results = interpret_puiseux_expansions(expansions_np, x_sym, y_sym)

        print("\n[PUISEUX EXPANSIONS & INTERPRETATION]")
        for idx_e, ir in enumerate(interpret_results):
            print(f"  Expansion {idx_e}:")
            print("    ", ir["puiseux_expr"])
            print("    =>", ir["comment"])

        # ------------------------------------------------------------------
        # (C) Robustness: adversarial directions
        # ------------------------------------------------------------------
        polynom = F_expr_zero
        best_dirs_info = find_adversarial_directions(polynom, x_sym, y_sym, num_random=20, radius=0.01)
        print("\n[ROBUSTNESS] Checking top directions from the polynomial's phase analysis:")
        all_adv_curves = []
        results_table = []
        for d_id, (dir_radians, phase_val) in enumerate(best_dirs_info):
            adv_list, changed_class, changed_radius = test_adversarial_impact(
                model, xstar, dir_radians, radius=0.02, steps=20, device=device
            )
            all_adv_curves.append(adv_list)
            results_table.append({
                "direction_id": d_id,
                "direction_radians": dir_radians,
                "phase": phase_val,
                "changed_class": changed_class,
                "changed_radius": changed_radius
            })

        print("DirID | (thx, thy)        | phase     | changed_class | changed_radius")
        for row in results_table:
            print("{:5d} | ({:.3f}, {:.3f}) | {:.3f}  | {}           | {}".format(
                row["direction_id"],
                row["direction_radians"][0], row["direction_radians"][1],
                row["phase"],
                row["changed_class"],
                f"{row['changed_radius']:.4f}" if row["changed_radius"] is not None else None
            ))

        # Plot robustness curves.
        robustness_plot_path = os.path.join(OUT_DIR, f'robustness_curves_point{i}.png')
        plot_robustness_curve(all_adv_curves, save_path=robustness_plot_path)

        # ------------------------------------------------------------------
        # (D) LIME & SHAP with temperature (apply scaler consistently)
        # ------------------------------------------------------------------
        xstar_reshaped = xstar.reshape(1, -1).astype(np.float32)
        xstar_scaled = (scaler_full.transform(xstar_reshaped) 
                        if scaler_full is not None else xstar_reshaped)

        # LIME.
        lime_exp = compute_lime_explanation(model, X_train, xstar_scaled[0], device, T=T)
        lime_list = lime_exp.as_list()
        print("\n[LIME Explanation]")
        for feat, val in lime_list:
            print(f" {feat}: {val:.3f}")

        # SHAP.
        shap_vals = compute_shap_explanation(
            model, xstar_scaled, device,
            background=X_train[:10], num_samples=100, T=T
        )
        shap_c = _normalize_shap_output(shap_vals)
        shap_class0 = shap_c[0][0]
        shap_class1 = (shap_c[1][0] if shap_c[1] is not None else None)

        print("\n[SHAP Explanation] per feature:")
        for fid, feat_name in enumerate(["Re(z1)", "Re(z2)", "Im(z1)", "Im(z2)"]):
            if shap_class1 is None:
                print("  {} => shap: {:.3f}".format(feat_name, scalarize(shap_class0[fid])))
            else:
                print("  {} => shap0: {:.3f}, shap1: {:.3f}".format(
                    feat_name, scalarize(shap_class0[fid]), scalarize(shap_class1[fid])
                ))

        # ==== AXIS-BASELINE RAY SWEEPS (add below SHAP block) ====
        def _flip_radius_along_vector(model, x0, v, device, T=None, r_max=0.02, steps=20):
            v = np.asarray(v, dtype=np.float32)
            n = np.linalg.norm(v)
            if not np.isfinite(n) or n < 1e-12:
                return None
            v = v / n
            x_init = torch.from_numpy(x0.astype(np.float32)).to(device).unsqueeze(0)
            with torch.no_grad():
                logits0 = complex_modulus_to_logits(model(x_init))
                if T is not None:
                    logits0 = logits0 / T.to(device)
                y0 = int(torch.softmax(logits0, dim=1).argmax(dim=1).item())
            for t in np.linspace(0.0, r_max, steps+1)[1:]:
                x_t = torch.from_numpy((x0 + t * v).astype(np.float32)).to(device).unsqueeze(0)
                logits = complex_modulus_to_logits(model(x_t))
                if T is not None:
                    logits = logits / T.to(device)
                y = int(torch.softmax(logits, dim=1).argmax(dim=1).item())
                if y != y0:
                    return float(t)
            return None  # no flip within budget

        def _idx_from_featname(name: str):
            # Map feature label to index in R^4 = [Re(z1), Re(z2), Im(z1), Im(z2)]
            name = name.replace(" ", "").lower()
            if "re(z1)" in name or "rex1" in name: return 0
            if "re(z2)" in name or "rex2" in name: return 1
            if "im(z1)" in name or "imx1" in name: return 2
            if "im(z2)" in name or "imx2" in name: return 3
            return None

        def _best_axis_from_lime(lime_list):
            # lime_list: [(feat_name, weight), ...] descending by |weight|
            for feat, w in sorted(lime_list, key=lambda kv: abs(kv[1]), reverse=True):
                j = _idx_from_featname(feat)
                if j is not None:
                    return j
            return None

        def _best_axis_from_shap(shap_vec):
            # shap_vec: ndarray shape (4,)
            import numpy as np
            j = int(np.nanargmax(np.abs(shap_vec)))
            return j

        # pick top axis for LIME/SHAP
        axis_lime = _best_axis_from_lime(lime_list)
        axis_shap = _best_axis_from_shap(shap_class1 if shap_class1 is not None else shap_class0)

        # gradient (saliency) direction in R^4
        model.zero_grad()
        x_t = torch.from_numpy(xstar.astype(np.float32)).to(device).unsqueeze(0).requires_grad_(True)
        logits = complex_modulus_to_logits(model(x_t))
        if T is not None:
            logits = logits / T.to(device)
        y_hat = int(torch.softmax(logits, dim=1).argmax(dim=1).item())
        logits[0, y_hat].backward()
        g = x_t.grad.detach().cpu().numpy().reshape(-1)
        grad_dir = g if np.linalg.norm(g) > 1e-12 else None
        E = np.eye(4, dtype=np.float32)
        def _min_flip_two_sides(vec):
            r1 = _flip_radius_along_vector(model, xstar, vec, device, T=T, r_max=0.02, steps=20)
            r2 = _flip_radius_along_vector(model, xstar, -vec, device, T=T, r_max=0.02, steps=20)
            vals = [r for r in [r1, r2] if r is not None]
            return (min(vals) if vals else None)

        flip_grad = _min_flip_two_sides(grad_dir) if grad_dir is not None else None
        flip_lime = _min_flip_two_sides(E[axis_lime]) if axis_lime is not None else None
        def _clamp_index(idx, n):
            if idx is None or n <= 0:
                return None
            idx = int(idx)
            if idx < 0:
                return 0
            if idx >= n:
                return n - 1
            return idx

        idx_shap = _clamp_index(axis_shap, len(E))
        flip_shap = _min_flip_two_sides(E[idx_shap]) if idx_shap is not None else None



        # 7b. Axis-baseline ray sweeps 
        axis_report_str = (
            "AXIS-BASELINE RAY SWEEPS (r_max=0.02, steps=20)\n"
            f"   flip_grad = {('N/A' if flip_grad is None else f'{float(flip_grad):.6f}')}\n"
            f"   flip_lime = {('N/A' if flip_lime is None else f'{float(flip_lime):.6f}')}\n"
            f"   flip_shap = {('N/A' if flip_shap is None else f'{float(flip_shap):.6f}')}\n\n"
        )

                
        # ------------------------------------------------------------------
        # (E) 2D local decision contour (fix pairs of dims)
        # ------------------------------------------------------------------
        save_dim_1_3 = os.path.join(OUT_DIR, f'contour_point{i}_fix_dim=[1,3].png')
        plot_local_contour_2d(model, xstar, fix_dims=(1, 3), delta=0.05, steps=50, device=device, save_path=save_dim_1_3)
        save_dim_0_2 = os.path.join(OUT_DIR, f'contour_point{i}_fix_dim=[0,2].png')
        plot_local_contour_2d(model, xstar, fix_dims=(0, 2), delta=0.05, steps=50, device=device, save_path=save_dim_0_2)

        # ------------------------------------------------------------------
        # (F) Resource benchmark (Puiseux vs gradient saliency)
        # ------------------------------------------------------------------
        times_pp, _, _ = benchmark_local_poly_approx_and_puiseux(
            model=model,
            xstar=xstar,
            local_poly_func=local_poly_approx_complex,
            puiseux_func=puiseux_expansions,
            delta=0.05,
            degree=4,
            n_samples=300,
            device=device,
            do_factor=True,
            do_simplify=True,
            puiseux_prec=4
        )
        sal = time_gradient_saliency(model, xstar, device, T=T, repeat=5)
        res_txt = os.path.join(OUT_DIR, f'resource_point{i}.txt')
        with open(res_txt, "w") as fz:
            fz.write("=== Resource benchmark (Puiseux vs. gradient saliency) ===\n")
            fz.write(f"Puiseux: sample={times_pp['time_sampling']:.2f}s, lsq={times_pp['time_lstsq']:.2f}s, "
                     f"factor={times_pp['time_factor']:.2f}s, simplify={times_pp['time_simplify']:.2f}s, "
                     f"expansion={times_pp['time_puiseux']:.2f}s, total={times_pp['time_total']:.2f}s\n")
            fz.write(f"Saliency: time={sal['time_ms']:.2f} ms, grad_norm={sal['grad_norm']:.3e}, "
                     f"cpu_dRSS={sal['cpu_rss_mb_delta']:.1f} MB, gpu_peak={sal['gpu_peak_mb']:.1f} MB\n")
        print(f"[INFO] Resource benchmark saved to {res_txt}")

        # Append to aggregate reports.
        kink_rows.append([i, kdiag['frac_kink'], kdiag['frac_active'], kdiag['frac_inactive'], kdiag['n_samples']])
        res_rows.append([i, times_pp['time_total'], times_pp.get('cpu_rss_mb', float('nan')),
                         times_pp.get('gpu_peak_mb', float('nan')), sal['time_ms'], sal['gpu_peak_mb']])
        fit_rows.append([i, fit_diag['kept_ratio'], fit_diag['cond'], fit_diag.get('degree_used', 4),
                         metrics['RMSE'], metrics['sign_agreement']])

        # Save extra residual diagnostics from local_poly_approx_complex -> fit_polynomial_complex.
        fit_rows[-1] += [
            fit_diag.get('resid_mean', float('nan')),
            fit_diag.get('resid_std', float('nan')),
            fit_diag.get('resid_skew', float('nan')),
            fit_diag.get('resid_kurt', float('nan')),
        ]

        
        # ------------------------------------------------------------------
        # (G) Save a comprehensive per-point report
        # ------------------------------------------------------------------
        out_txt_path = os.path.join(OUT_DIR, f'benchmark_point{i}.txt')
        with open(out_txt_path, "w") as f_out:
            f_out.write("=" * 80 + "\n")
            f_out.write(f"Local Analysis Report for Uncertain Point #{i}\n")
            f_out.write("=" * 80 + "\n\n")

            f_out.write("0. Kink diagnostics (modReLU neighborhood):\n")
            f_out.write(f"   frac_kink       : {kdiag['frac_kink']:.3f}\n")
            f_out.write(f"   frac_active     : {kdiag['frac_active']:.3f}\n")
            f_out.write(f"   frac_inactive   : {kdiag['frac_inactive']:.3f}\n")
            f_out.write(f"   samples_checked : {kdiag['n_samples']}\n\n")

            f_out.write("1. Base Point (xstar):\n")
            f_out.write(f"   {xstar.tolist()}\n\n")

            f_out.write("2. Local polynomial fit (robust):\n")
            f_out.write(f"   kept / total    : {fit_diag['n_kept']} / {fit_diag['n_total']} "
                        f"({100*fit_diag['kept_ratio']:.1f}%)\n")
            f_out.write(f"   cond(A)         : {fit_diag['cond']:.3e}\n")
            f_out.write(f"   rank / monomials: {fit_diag['rank']} / {fit_diag['n_monomials']}\n\n")

            f_out.write("3. Approximation Quality Metrics:\n")
            f_out.write(f"   RMSE            : {metrics['RMSE']:.3f}\n")
            f_out.write(f"   MAE             : {metrics['MAE']:.3f}\n")
            f_out.write(f"   Pearson Corr    : {metrics['corr_pearson']:.3f}\n")
            f_out.write(f"   Sign Agreement  : {metrics['sign_agreement']:.3f}\n\n")

            f_out.write("4. Puiseux Expansions and Interpretation:\n")
            for idx_e, ir in enumerate(interpret_results):
                f_out.write(f"   >> Expansion {idx_e}:\n")
                f_out.write(f"      Puiseux Expression: {ir['puiseux_expr']}\n")
                f_out.write(f"      Interpretation    : {ir['comment']}\n")
            f_out.write("\n")

            # --- Predicted onset radius from dominant quadratic/quartic Puiseux coefficients ---
            # We compute r_dom ≈ sqrt(|c2|/|c4|) using the largest-magnitude quadratic and quartic
            # coefficients across all local branches. This is a heuristic for the radius at which
            # the quartic term starts to dominate curvature (onset of rapid class flips).
            import re, math
            import numpy as np

            def _extract_c2_c4(ir_dict):
                """Extract |c2| and |c4| candidates from a single 'interpret_results' entry.
                Supports several possible field names; falls back to regex over the textual series."""
                c2_cands, c4_cands = [], []

                # 1) Structured fields that some implementations may provide
                for k in ("dominant_c2", "c2_dom", "c2", "quad_coeff", "c2_coeff"):
                    val = ir_dict.get(k, None)
                    try:
                        if val is not None:
                            c2_cands.append(abs(complex(val)))
                    except Exception:
                        pass
                for k in ("dominant_c4", "c4_dom", "c4", "quartic_coeff", "c4_coeff"):
                    val = ir_dict.get(k, None)
                    try:
                        if val is not None:
                            c4_cands.append(abs(complex(val)))
                    except Exception:
                        pass

                # 2) A generic 'coeffs' / 'terms' list with entries like {'degree': 2, 'coeff': ...}
                coeff_list = ir_dict.get("coeffs") or ir_dict.get("terms") or []
                for t in coeff_list:
                    try:
                        deg = t.get("degree", None)
                        val = t.get("coeff", None)
                    except AttributeError:
                        deg, val = None, None
                    if deg in (2, 2.0) and val is not None:
                        try: c2_cands.append(abs(complex(val)))
                        except Exception: pass
                    if deg in (4, 4.0) and val is not None:
                        try: c4_cands.append(abs(complex(val)))
                        except Exception: pass

                # 3) Regex fallback on the textual representation 'puiseux_expr'
                s = str(ir_dict.get("puiseux_expr", ""))
                # accept patterns like: 3.08 x^2, -291.86 x2, 1.09e7 x**4
                def _grab_deg(deg, bucket):
                    pat = rf'([+-]?\s*(?:\d+\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*(?:\*?\s*)?x(?:\s*\^\s*|\*\*|)\s*{deg}\b'
                    for g in re.findall(pat, s):
                        try:
                            bucket.append(abs(float(g)))
                        except Exception:
                            pass
                _grab_deg(2, c2_cands)
                _grab_deg(4, c4_cands)

                c2 = max(c2_cands) if len(c2_cands) else float("nan")
                c4 = max(c4_cands) if len(c4_cands) else float("nan")
                return c2, c4

            # Aggregate across all branches for this anchor
            c2_vals, c4_vals = [], []
            for ir in interpret_results:
                c2_i, c4_i = _extract_c2_c4(ir)
                if np.isfinite(c2_i):
                    c2_vals.append(c2_i)
                if np.isfinite(c4_i):
                    c4_vals.append(c4_i)

            c2_dom = max(c2_vals) if len(c2_vals) else float("nan")
            c4_dom = max(c4_vals) if len(c4_vals) else float("nan")

            if np.isfinite(c2_dom) and np.isfinite(c4_dom) and c4_dom > 0:
                r_dom = float(np.sqrt(c2_dom / max(c4_dom, 1e-12)))
                print(f"[PUISEUX] dominant |c2|={c2_dom:.3g}, |c4|={c4_dom:.3g} -> predicted onset radius r_dom≈{r_dom:.6f}")
                f_out.write(f"   Dominant |c2| = {c2_dom:.6g}\n")
                f_out.write(f"   Dominant |c4| = {c4_dom:.6g}\n")
                f_out.write(f"   Predicted onset radius r_dom ≈ sqrt(|c2|/|c4|) = {r_dom:.6f}\n")
            else:
                print("[PUISEUX] Could not extract dominant |c2|/|c4|; skipping r_dom.")
                f_out.write("   Predicted onset radius r_dom: N/A (could not extract |c2|/|c4|)\n")

            f_out.write("\n")

            
            
                        # --- Dominant-ratio heuristic (r_dom) and observed flip radius ---
            # We estimate when quartic terms overtake quadratic curvature: 
            # r_dom ≈ sqrt(max|c2| / max|c4|), where c2 and c4 are coefficients of x^2 and x^4
            # across all Puiseux branches. We also extract the minimal observed flip radius r_flip
            # from the robustness table above.
            try:
                # 1) Parse Puiseux expressions into SymPy and collect coefficients
                exprs = []
                for ir in interpret_results:
                    expr_repr = ir.get("puiseux_expr")
                    if isinstance(expr_repr, sympy.Expr):
                        e = sympy.expand(expr_repr)
                    else:
                        # robust to string formatting
                        e = sympy.expand(sympy.sympify(str(expr_repr)))
                    exprs.append(e)

                max_abs_c2 = 0.0
                max_abs_c4 = 0.0
                for e in exprs:
                    c2 = e.coeff(x_sym, 2)
                    c4 = e.coeff(x_sym, 4)

                    # Convert possibly-complex SymPy numbers to Python complex and take magnitude
                    if c2 != 0:
                        max_abs_c2 = max(max_abs_c2, float(abs(complex(c2.evalf()))))
                    if c4 != 0:
                        max_abs_c4 = max(max_abs_c4, float(abs(complex(c4.evalf()))))

                if (max_abs_c2 > 0.0) and (max_abs_c4 > 0.0):
                    r_dom = float(np.sqrt(max_abs_c2 / max(max_abs_c4, 1e-12)))
                else:
                    r_dom = float("nan")
            except Exception as e:
                logger.warning("Failed to compute r_dom for point %d: %s", i, e)
                max_abs_c2 = float("nan")
                max_abs_c4 = float("nan")
                r_dom = float("nan")

            # 2) Observed minimal flip radius from robustness results_table
            try:
                r_flip_candidates = [row["changed_radius"] for row in results_table
                                     if row.get("changed_radius") is not None]
                r_flip = float(min(r_flip_candidates)) if len(r_flip_candidates) else float("nan")
            except Exception as e:
                logger.warning("Failed to compute r_flip for point %d: %s", i, e)
                r_flip = float("nan")

            # 3) Persist both numbers to the TXT so external parsers can consume them
            f_out.write("   Dominant-ratio heuristic and flip comparison:\n")
            f_out.write(f"      max|c2| = {max_abs_c2:.6g}, max|c4| = {max_abs_c4:.6g}\n")
            f_out.write(f"      Predicted onset radius r_dom ≈ sqrt(|c2|/|c4|) = {r_dom:.6f}\n")
            f_out.write("      Observed min flip radius r_flip = "
                        f"{('N/A' if np.isnan(r_flip) else f'{r_flip:.6f}')}\n\n")

            # 4) Collect for a CSV summary across all anchors
            dom_rows.append([i, max_abs_c2, max_abs_c4, r_dom, r_flip])

            
            
            
            f_out.write("5. Robustness Analysis Results:\n")
            f_out.write("-" * 80 + "\n")
            f_out.write("{:<10s} {:<20s} {:<10s} {:<18s} {:<15s}\n".format(
                "Dir. ID", "(thx, thy)", "Phase", "Class Change", "Change Radius"))
            f_out.write("-" * 80 + "\n")
            for row in results_table:
                change_radius_str = f"{row['changed_radius']:.4f}" if row["changed_radius"] is not None else "N/A"
                f_out.write("{:<10d} ({:<6.3f}, {:<6.3f})    {:<10.3f} {:<18s} {:<15s}\n".format(
                    row["direction_id"],
                    row["direction_radians"][0], row["direction_radians"][1],
                    row["phase"],
                    "YES" if row["changed_class"] else "NO",
                    change_radius_str
                ))
            f_out.write("\n")

            f_out.write("6. LIME Explanation (Local Feature Importance):\n")
            for feat, val in lime_list:
                f_out.write(f"   {feat}: {val:.3f}\n")
            f_out.write("\n")

            f_out.write("7. SHAP Explanation (Feature Contributions per Class):\n")
            for fid, feat_name in enumerate(["Re(z1)", "Re(z2)", "Im(z1)", "Im(z2)"]):
                if shap_class1 is None:
                    f_out.write(f"  {feat_name} => Class: {scalarize(shap_class0[fid]):.3f}\n")
                else:
                    f_out.write(f"  {feat_name} => Class 0: {scalarize(shap_class0[fid]):.3f}, "
                                f"Class 1: {scalarize(shap_class1[fid]):.3f}\n")
            f_out.write("\n")

            # 7b. Axis-baseline ray sweeps
            f_out.write(axis_report_str)
            
            f_out.write("8. Resource benchmark (Puiseux vs gradient saliency):\n")
            f_out.write(f"   Puiseux times   : sample={times_pp['time_sampling']:.2f}s, lsq={times_pp['time_lstsq']:.2f}s, "
                        f"factor={times_pp['time_factor']:.2f}s, simplify={times_pp['time_simplify']:.2f}s, "
                        f"expansion={times_pp['time_puiseux']:.2f}s, total={times_pp['time_total']:.2f}s\n")
            f_out.write(f"   Saliency (1x avg): time={sal['time_ms']:.2f} ms, grad_norm={sal['grad_norm']:.3e}, "
                        f"cpu_dRSS={sal['cpu_rss_mb_delta']:.1f} MB, gpu_peak={sal['gpu_peak_mb']:.1f} MB\n")

            f_out.write("\n" + "=" * 80 + "\n")
            f_out.write("End of Report\n")
            f_out.write("=" * 80 + "\n")

    # ==================================================================
    # — CALIBRATION CI TABLE: mean ± 95% CI for ECE/Brier/NLL
    # ==================================================================
    logger.info("=== Building calibration CI table (Platt/Isotonic/Beta/Vector/Temp) ===")

    # 1) 5-fold stratified CV.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=getattr(args, "seed", 42))
    methods = ["none", "platt", "isotonic", "beta", "vector", "temperature"]
    metrics_by_method = {m: {"ECE": [], "Brier": [], "NLL": []} for m in methods}
    # Extra bucket for the ablation row (only if a temperature file is actually used).
    if T is not None:
        metrics_by_method["none_T0"] = {"ECE": [], "Brier": [], "NLL": []}


    def predict_probs_binary(X_batch_full):
        """Return p(class 1) for a batch of signals (in C^2 after compression and optional scaling)."""
        Xc2 = compress_to_C2(X_batch_full)
        Xc2 = scaler_full.transform(Xc2) if scaler_full is not None else Xc2
        probs = model_to_explain(Xc2, model, device, T=T)  # shape [N,2]
        return probs[:, 1].astype(float)

    def predict_probs_binary_Tnone(X_batch_full):
        """Return p(class 1) with NO temperature scaling (T=None) – ablation row for CI table."""
        Xc2 = compress_to_C2(X_batch_full)
        Xc2 = scaler_full.transform(Xc2) if scaler_full is not None else Xc2
        probs = model_to_explain(Xc2, model, device, T=None)  # force no temperature
        return probs[:, 1].astype(float)

    
    # 2) CV: fit calibrators on train-fold, evaluate on test-fold.
    for fold_id, (idx_tr, idx_te) in enumerate(skf.split(X_full, y_full)):
        logger.info("Fold %d/%d", fold_id + 1, skf.get_n_splits())

        p_tr_uncal = predict_probs_binary(X_full[idx_tr])
        y_tr = y_full[idx_tr].astype(int)
        p_te_uncal = predict_probs_binary(X_full[idx_te])
        y_te = y_full[idx_te].astype(int)
        
        # --- Ablation: raw model probabilities with T=None (no temperature scaling at inference) ---
        if T is not None:
            p_te_raw = predict_probs_binary_Tnone(X_full[idx_te])
            res_raw = compute_binary_metrics(p_te_raw, y_te, n_bins=15)
            for k, v in res_raw.items():
                metrics_by_method["none_T0"][k].append(v)


        cals = {m: fit_binary_calibrator(m, p_tr_uncal, y_tr) for m in methods}

        for m, cal in cals.items():
            p_cal = cal(p_te_uncal)
            res = compute_binary_metrics(p_cal, y_te, n_bins=15)
            for k, v in res.items():
                metrics_by_method[m][k].append(v)

    # Save raw per-fold metrics (for reproducible comparisons).    
    raw_rows = []
    for m in list(methods) + (["none_T0"] if ("none_T0" in metrics_by_method) else []):
        k2v = metrics_by_method[m]
        n = max(len(k2v["ECE"]), len(k2v["Brier"]), len(k2v["NLL"]))
        for i in range(n):
            raw_rows.append([
                m.upper(),
                k2v["ECE"][i] if i < len(k2v["ECE"]) else np.nan,
                k2v["Brier"][i] if i < len(k2v["Brier"]) else np.nan,
                k2v["NLL"][i] if i < len(k2v["NLL"]) else np.nan
            ])
    pd.DataFrame(raw_rows, columns=["method","ECE","Brier","NLL"]).to_csv(
        os.path.join(OUT_DIR, "calibration_folds_raw.csv"), index=False)

    
                
    # 3) Summarize mean ± 95% CI and save the table.        
    rows = []
    for m in list(methods) + (["none_T0"] if ("none_T0" in metrics_by_method) else []):
        E_m, E_lo, E_hi = mean_ci_t(metrics_by_method[m]["ECE"])
        B_m, B_lo, B_hi = mean_ci_t(metrics_by_method[m]["Brier"])
        N_m, N_lo, N_hi = mean_ci_t(metrics_by_method[m]["NLL"])
        pretty = (
            "Uncalibrated (T=None)" if m == "none_T0"
            else ("Uncalibrated" if m == "none" else m.capitalize())
        )
        rows.append({
            "Method": pretty,
            "ECE":   format_mean_ci(E_m, E_lo, E_hi, decimals=4),
            "Brier": format_mean_ci(B_m, B_lo, B_hi, decimals=4),
            "NLL":   format_mean_ci(N_m, N_lo, N_hi, decimals=4),
            # also store raw numbers
            "ECE_mean": E_m, "ECE_lo": E_lo, "ECE_hi": E_hi,
            "Brier_mean": B_m, "Brier_lo": B_lo, "Brier_hi": B_hi,
            "NLL_mean": N_m, "NLL_lo": N_lo, "NLL_hi": N_hi,
        })

        

    df_ci = pd.DataFrame(rows)
    table_csv_path = os.path.join(OUT_DIR, "calibration_ci_table.csv")
    df_ci.to_csv(table_csv_path, index=False)

    # Coherent TXT report (supersedes earlier narrower report).
    txt_path = os.path.join(OUT_DIR, "calibration_ci_report.txt")
    with open(txt_path, "w") as f:
        f.write("=== Calibration: mean ± 95% CI over 5 folds ===\n")
        f.write("Baseline for relative drop: NONE (uncalibrated probabilities), 5-fold mean.\n")
        f.write(df_ci[["Method", "ECE", "Brier", "NLL"]].to_string(index=False))
        f.write("\n")
    logger.info("Saved CI table -> %s", table_csv_path)
    logger.info("Saved CI report -> %s", txt_path)

    # ==================================================================
    # Aggregate reports across anchors + T-sweep
    # ==================================================================        
    try:
        pd.DataFrame(kink_rows, columns=["point","frac_kink","frac_active","frac_inactive","n"]).to_csv(
            os.path.join(OUT_DIR,"kink_summary.csv"), index=False)
        pd.DataFrame(res_rows, columns=["point","puiseux_time_s","cpu_rss_mb","gpu_peak_mb","saliency_ms","sal_gpu_peak_mb"]).to_csv(
            os.path.join(OUT_DIR,"resource_summary.csv"), index=False)
        pd.DataFrame(fit_rows, columns=[
            "point","kept_ratio","cond","degree_used","RMSE","sign_agree",
            "resid_mean","resid_std","resid_skew","resid_kurt"
        ]).to_csv(os.path.join(OUT_DIR,"local_fit_summary.csv"), index=False)
        pd.DataFrame(dom_rows, columns=["point","max_abs_c2","max_abs_c4","r_dom","r_flip"]).to_csv(
            os.path.join(OUT_DIR,"dominant_ratio_summary.csv"), index=False)


        # --- GLOBAL kink report ---
        ks = pd.DataFrame(kink_rows, columns=["point","frac_kink","frac_active","frac_inactive","n"])
        lf = pd.DataFrame(fit_rows, columns=[
            "point","kept_ratio","cond","degree_used","RMSE","sign_agree",
            "resid_mean","resid_std","resid_skew","resid_kurt"
        ])
        rep = []
        for thr in (0.01, 0.05, 0.10):
            mask = ks["frac_kink"] >= thr
            rep.append((thr, float(mask.mean()), float(ks.loc[mask,"frac_kink"].median())))
        with open(os.path.join(OUT_DIR, "kink_global_summary.txt"), "w") as f:
            f.write("=== Kink prevalence across anchors ===\n")
            for thr, frac, med in rep:
                f.write(f"frac_kink >= {thr:.2%}: share={frac:.3f}, median(frac_kink)={med:.3f}\n")
            # Effect of kinks on quality and residuals:
            for thr in (0.01, 0.05, 0.10):
                mask = ks["frac_kink"] >= thr
                lo = lf.loc[~mask, ["RMSE","sign_agree","resid_mean","resid_std","resid_skew","resid_kurt"]].mean()
                hi = lf.loc[ mask, ["RMSE","sign_agree","resid_mean","resid_std","resid_skew","resid_kurt"]].mean()
                f.write(f"\n-- threshold {thr:.2%} --\n")
                f.write(f"LOW-KINK mean: {lo.to_dict()}\n")
                f.write(f"HIGH-KINK mean: {hi.to_dict()}\n")
    except Exception as e:
        logger.warning("Saving aggregate summaries failed: %s", e)

    # ECE sensitivity to branch multiplicity error m:
    # Assume T ∝ m^{-1/2}. If m_est = m_true*(1+eps),
    # then T_mult = (m_true/m_est)^{1/2} = (1+eps)^{-1/2}.
    try:
        X_full_c2 = compress_to_C2(X_full)
        X_full_c2 = scaler_full.transform(X_full_c2) if scaler_full is not None else X_full_c2

        rel_errs = (-0.5, -0.25, -0.10, -0.05, 0.0, 0.05, 0.10, 0.25, 0.50)
        from src.post_processing import sweep_multiplicity_misestimation
        res = sweep_multiplicity_misestimation(
            model, X_full, y_full,
            compress_fn=compress_to_C2, scaler=scaler_full,
            device=device, T_base=T,
            rel_errors=rel_errs, gamma=0.5
        )
        df_m = pd.DataFrame(res, columns=["rel_err_m","ECE"])
        df_m.to_csv(os.path.join(OUT_DIR, "branch_multiplicity_sensitivity.csv"), index=False)
        logger.info("Saved branch_multiplicity_sensitivity.csv")
    except Exception as e:
        logger.warning("Multiplicity-sensitivity sweep failed: %s", e)


    print("\n[INFO] Completed analysis for all uncertain points.")
