# Copyright (c) 2025 Piotr Migus
# This code is licensed under the MIT License.
# See the LICENSE file in the repository root for full license information.

"""
Build Newton–Puiseux evidence by joining anchors, benchmark TXT, and
dominant-ratio CSV produced by post_processing_*.
"""

import re
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- PR tools (AUPRC for ranking by a score) ----------
def precision_recall_from_scores(y_true: np.ndarray, scores: np.ndarray):
    """
    Compute a precision–recall curve and step-based AP for a ranking signal.

    Parameters
    ----------
    y_true : np.ndarray
        1D array of {0, 1} where 1 = 'fragile' (flip within budget).
    scores : np.ndarray
        1D array of floats; higher = 'more fragile' (e.g., |c4|).

    Returns
    -------
    rec : np.ndarray
        Recall points (starts at 0).
    prec : np.ndarray
        Precision points (starts at 1 by convention).
    ap_step : float
        Average precision via step integration.
    df_curve : pandas.DataFrame
        Per-prefix table with k, threshold(score), tp, fp, fn, tn, precision, recall.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)

    # Keep only finite scores.
    valid = np.isfinite(s)
    y = y[valid]
    s = s[valid]

    # Stable sort by descending score.
    order = np.argsort(-s, kind='mergesort')
    y = y[order]
    s = s[order]

    tp_cum = np.cumsum(y)
    fp_cum = np.cumsum(1 - y)
    P = int(tp_cum[-1])            # positives
    N = int(fp_cum[-1])            # negatives

    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    recall = tp_cum / max(P, 1)

    # Prepend (recall=0, precision=1) for a conventional start.
    rec = np.concatenate(([0.0], recall))
    prec = np.concatenate(([1.0], precision))

    # Step-wise AP (area under the PR curve).
    ap_step = np.sum((rec[1:] - rec[:-1]) * prec[1:])

    k = np.arange(1, len(y) + 1)
    df_curve = pd.DataFrame({
        "k": k,
        "threshold": s,                   # score threshold at prefix k
        "tp": tp_cum,
        "fp": fp_cum,
        "fn": P - tp_cum,
        "tn": N - fp_cum,
        "precision": precision,
        "recall": recall
    })

    return rec, prec, float(ap_step), df_curve


# ---------- Robust float regex ----------
FLOAT_RE = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'

# ---------- TXT parser ----------
# ---------- TXT parser ----------
def parse_benchmark_file(txt_path: str) -> dict:
    """
    Parse a single post_processing_*/benchmark_point*.txt file and extract:
    - kink diagnostics (frac_kink/active/inactive, samples_checked),
    - local-fit metrics (kept_ratio, cond, rank, n_monomials, degree_used, retry),
    - approximation metrics (RMSE/MAE/Pearson/Sign_Agreement, residual moments),
    - robustness table (flip radii and minimum flip radius),
    - Puiseux and saliency timings (including CPU/GPU memory and grad_norm),
    - optional r_dom prediction and axis-baseline sweep flips (grad/LIME/SHAP).

    Returns
    -------
    dict
        A dictionary with parsed scalars and lists ready to be joined downstream.
    """
    p = Path(txt_path)
    with p.open('r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    out = {
        "point": None,
        "frac_kink": np.nan,
        "frac_active": np.nan,
        "frac_inactive": np.nan,
        "samples_checked": np.nan,

        "kept_ratio": np.nan,
        "cond": np.nan,
        "rank": np.nan,
        "n_monomials": np.nan,
        "degree_used": np.nan,   # NEW
        "retry": np.nan,         # NEW

        "RMSE": np.nan,
        "MAE": np.nan,
        "Pearson": np.nan,
        "Sign_Agreement": np.nan,
        "resid_mean": np.nan,    # NEW
        "resid_std": np.nan,     # NEW
        "resid_skew": np.nan,    # NEW
        "resid_kurt": np.nan,    # NEW

        "flip_radii": [],
        "min_flip_radius": np.nan,

        "puiseux_time_s": np.nan,
        "saliency_ms": np.nan,
        "saliency_cpu_dRSS_MB": np.nan,
        "saliency_gpu_peak_MB": np.nan,
        "saliency_grad_norm": np.nan,  # NOTE: the trailing comma was needed in the log line

        "r_dom_pred": np.nan,
        "flip_grad": np.nan,     # NEW default
        "flip_lime": np.nan,     # NEW default
        "flip_shap": np.nan,     # NEW default
    }

    m_pt = re.search(r'point(\d+)', p.name)
    if m_pt:
        out["point"] = int(m_pt.group(1))

    # Scalars, timings, optional r_dom_pred.
    for raw in lines:
        s = raw.strip()
        s_norm = s.replace("—", "-")  # normalize em-dash

        # Kink diagnostics.
        if "frac_kink" in s_norm:
            m = re.search(FLOAT_RE, s_norm);  out["frac_kink"] = float(m.group()) if m else np.nan
        elif "frac_active" in s_norm:
            m = re.search(FLOAT_RE, s_norm);  out["frac_active"] = float(m.group()) if m else np.nan
        elif "frac_inactive" in s_norm:
            m = re.search(FLOAT_RE, s_norm);  out["frac_inactive"] = float(m.group()) if m else np.nan
        elif "samples_checked" in s_norm:
            m = re.search(r'\d+', s_norm);    out["samples_checked"] = int(m.group()) if m else np.nan

        # Local-fit section.
        elif "kept / total" in s_norm:
            m = re.search(r'(\d+)\s*/\s*(\d+).*?\(\s*(' + FLOAT_RE + r')\s*%\)', s_norm)
            if m: out["kept_ratio"] = float(m.group(3)) / 100.0
        elif "cond(A)" in s_norm:
            m = re.search(r'cond\(A\)\s*:\s*(' + FLOAT_RE + r')', s_norm)
            if m: out["cond"] = float(m.group(1))
        elif "rank / monomials" in s_norm:
            m = re.search(r'(\d+)\s*/\s*(\d+)', s_norm)
            if m: out["rank"], out["n_monomials"] = int(m.group(1)), int(m.group(2))
        elif "degree_used" in s_norm:
            m = re.search(r'degree_used\s*[:=]\s*(' + FLOAT_RE + r')', s_norm)
            if m: out["degree_used"] = float(m.group(1))
        elif "retry" in s_norm:
            m = re.search(r'retry\s*[:=]\s*(\d+)', s_norm)
            if m: out["retry"] = int(m.group(1))

        # Approximation quality.
        elif s_norm.startswith("RMSE"):
            m = re.search(FLOAT_RE, s_norm);  out["RMSE"] = float(m.group()) if m else np.nan
        elif s_norm.startswith("MAE"):
            m = re.search(FLOAT_RE, s_norm);  out["MAE"] = float(m.group()) if m else np.nan
        elif "Pearson" in s_norm:
            m = re.search(FLOAT_RE, s_norm);  out["Pearson"] = float(m.group()) if m else np.nan
        elif "Sign Agreement" in s_norm:
            m = re.search(FLOAT_RE, s_norm);  out["Sign_Agreement"] = float(m.group()) if m else np.nan
        elif "resid_mean" in s_norm:
            m = re.search(r'resid_mean\s*[:=]\s*(' + FLOAT_RE + r')', s_norm)
            if m: out["resid_mean"] = float(m.group(1))
        elif "resid_std" in s_norm:
            m = re.search(r'resid_std\s*[:=]\s*(' + FLOAT_RE + r')', s_norm)
            if m: out["resid_std"] = float(m.group(1))
        elif "resid_skew" in s_norm:
            m = re.search(r'resid_skew\s*[:=]\s*(' + FLOAT_RE + r')', s_norm)
            if m: out["resid_skew"] = float(m.group(1))
        elif "resid_kurt" in s_norm:
            m = re.search(r'resid_kurt\s*[:=]\s*(' + FLOAT_RE + r')', s_norm)
            if m: out["resid_kurt"] = float(m.group(1))

        # Puiseux timing.
        elif s_norm.startswith("Puiseux times"):
            mt = re.search(r'total=('+FLOAT_RE+')s', s_norm)
            if mt: out["puiseux_time_s"] = float(mt.group(1))

        # Saliency timing + resources.
        elif s_norm.startswith("Saliency"):
            mt = re.search(r'time=(' + FLOAT_RE + r')\s*ms', s_norm)
            if mt: out["saliency_ms"] = float(mt.group(1))
            mr = re.search(r'cpu_dRSS=(' + FLOAT_RE + r')\s*MB', s_norm)
            if mr: out["saliency_cpu_dRSS_MB"] = float(mr.group(1))
            mp = re.search(r'gpu_peak=(' + FLOAT_RE + r')\s*MB', s_norm)
            if mp: out["saliency_gpu_peak_MB"] = float(mp.group(1))
            mg = re.search(r'grad_norm=(' + FLOAT_RE + r')', s_norm)
            if mg: out["saliency_grad_norm"] = float(mg.group(1))

        # Optional r_dom/onset radius line.
        elif re.search(r'(r_dom|onset radius)', s_norm, flags=re.IGNORECASE):
            m = re.search(r'(?:r_dom|onset radius)[^=]*=\s*(' + FLOAT_RE + r')',
                          s_norm, flags=re.IGNORECASE)
            if m:
                out["r_dom_pred"] = float(m.group(1))

        # AXIS-BASELINE header -> initialize flip_* fields.
        elif s_norm.startswith("AXIS-BASELINE RAY SWEEPS"):
            out["flip_grad"] = np.nan
            out["flip_lime"] = np.nan
            out["flip_shap"] = np.nan
        elif s_norm.startswith("flip_grad"):
            m = re.search(FLOAT_RE, s_norm);  out["flip_grad"] = float(m.group()) if m else np.nan
        elif s_norm.startswith("flip_lime"):
            m = re.search(FLOAT_RE, s_norm);  out["flip_lime"] = float(m.group()) if m else np.nan
        elif s_norm.startswith("flip_shap"):
            m = re.search(FLOAT_RE, s_norm);  out["flip_shap"] = float(m.group()) if m else np.nan

    # Parse robustness results table -> collect flip radii.
    in_table = False
    for raw in lines:
        s = raw.strip()
        s_norm = s.replace("—", "-")

        if s_norm.startswith("Dir. ID"):
            in_table = True
            continue
        if in_table:
            if not s_norm or s_norm.startswith("6."):
                in_table = False
                continue
            # skip separators
            if set(s_norm) <= {"-", " "}:
                continue
            if "YES" in s_norm or "yes" in s_norm:
                m = re.search(r'(' + FLOAT_RE + r')\s*$', s_norm)
                if m:
                    try:
                        out["flip_radii"].append(float(m.group(1)))
                    except ValueError:
                        pass  # e.g., "N/A"

    if out["flip_radii"]:
        out["min_flip_radius"] = min(out["flip_radii"])

    return out



def collect_benchmarks(pp_dir: Path) -> pd.DataFrame:
    """
    Read all benchmark_point*.txt files in a post_processing_* directory
    and assemble a normalized DataFrame of the parsed contents.

    Parameters
    ----------
    pp_dir : pathlib.Path
        Path to the post-processing directory.

    Returns
    -------
    pandas.DataFrame
        Subset of standardized columns useful for joining and analysis.
    """
    paths = sorted(glob.glob(str(pp_dir / "benchmark_point*.txt")))
    if not paths:
        print(f"[WARN] No benchmark_point*.txt found under: {pp_dir}")
        return pd.DataFrame()
    rows = [parse_benchmark_file(p) for p in paths]
    df = pd.DataFrame(rows)
    keep = [
        "point", "min_flip_radius", "flip_radii",
        "frac_kink", "frac_active", "frac_inactive",
        "kept_ratio", "cond", "rank", "n_monomials",
        "degree_used", "retry",
        "RMSE", "MAE", "Pearson", "Sign_Agreement",
        "resid_mean", "resid_std", "resid_skew", "resid_kurt",
        "puiseux_time_s", "saliency_ms", "saliency_cpu_dRSS_MB", "saliency_gpu_peak_MB",
        "saliency_grad_norm","r_dom_pred", "flip_grad", "flip_lime", "flip_shap"
    ]
    return df[[c for c in keep if c in df.columns]]


# ---------- Loaders ----------
def load_anchors(up_dir: Path) -> pd.DataFrame:
    """
    Load uncertain anchors and derive convenience columns.

    - Adds a 'point' index [0..N-1] for consistent joining.
    - Renames 'index' to 'anchor_index' if present (for clarity).
    - Computes pmax = max(p1, p2) and margin = |p1 - p2| when available.
    """
    path = up_dir / "uncertain_full.csv"
    df = pd.read_csv(path)
    df = df.reset_index(drop=True)
    df["point"] = np.arange(len(df), dtype=int)
    if "index" in df.columns:
        df = df.rename(columns={"index": "anchor_index"})
    if {"p1", "p2"}.issubset(df.columns):
        df["pmax"] = df[["p1", "p2"]].max(axis=1)
        df["margin"] = (df["p1"] - df["p2"]).abs()
    return df


def load_dom_ratio(pp_dir: Path) -> pd.DataFrame:
    """
    Load dominant-ratio summary if available and normalize column names.

    Returns a DataFrame with (subset of) columns:
    ['point', 'c2_max_abs', 'c4_max_abs', 'r_dom', 'r_flip'] when present.
    """
    path = pp_dir / "dominant_ratio_summary.csv"
    if not path.exists():
        print(f"[WARN] Missing {path.name}; will rely on r_dom_pred from TXT if available.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Normalize column names commonly seen across runs.
    df = df.rename(columns={
        "max_abs_c2": "c2_max_abs", "max_abs_c4": "c4_max_abs",
        "point_id": "point"  # if saved as 'point_id'
    })
    for c in ["c2_max_abs", "c4_max_abs", "r_dom", "r_flip"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[[c for c in ["point","c2_max_abs","c4_max_abs","r_dom","r_flip"] if c in df.columns]]


# ---------- Main ----------
def main():
    """
    Orchestrate loading of inputs, joining evidence, running triage analyses,
    computing PR/AUPRC summaries, writing figures and markdown, and exporting
    compact CSVs for downstream use.
    """
    BASE = Path(__file__).resolve().parent  # build_np_evidence/

    # Auto-detect REAL vs RADIO (prefer *_real if it exists, else *_radio).
    UP_DIR = (BASE.parent / "up_real") if (BASE.parent / "up_real").exists() else (BASE.parent / "up_radio")
    PP_DIR = (BASE.parent / "post_processing_real") if (BASE.parent / "post_processing_real").exists() else (BASE.parent / "post_processing_radio")
    OUT_DIR = BASE
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Using UP_DIR = {UP_DIR.name}, PP_DIR = {PP_DIR.name}")

    # 1) Load sources.
    df_anchors = load_anchors(UP_DIR)
    df_bench = collect_benchmarks(PP_DIR)
    df_dom = load_dom_ratio(PP_DIR)

    # Sanity check: are TXT points indexed consistently with anchors?
    if not df_bench.empty and df_bench["point"].max() >= len(df_anchors):
        print("[WARN] TXT points index > #anchors; check enumeration 0..N-1!")

    # 2) Join on 'point'.
    df = df_anchors.merge(df_bench, on="point", how="left", suffixes=("", "_bench"))
    if not df_dom.empty:
        before = df["point"].nunique()
        df = df.merge(df_dom, on="point", how="left")
        matched = int(df["r_dom"].notna().sum())
        print(f"[INFO] Dominant-ratio matched rows: {matched}/{before}")
    else:
        if "r_dom_pred" in df.columns:
            df["r_dom"] = df["r_dom_pred"]

    # 3) Observed flip radius.
    if "r_flip" in df.columns:
        df["r_flip_obs"] = df["r_flip"]
    if "r_flip_obs" not in df.columns or df["r_flip_obs"].isna().all():
        if "min_flip_radius" in df.columns:
            df["r_flip_obs"] = df["min_flip_radius"]
            
    BUDGET = 0.02

    def _hit_rate_col(df, name, BUDGET=0.02):
        """Share of anchors with r <= BUDGET for a given radius-like column."""
        s = pd.to_numeric(df.get(name), errors="coerce")
        hits = (s <= BUDGET).fillna(False).sum()
        n = int(df["point"].nunique())
        return float(hits) / max(n, 1)

    def _col(df, name):
        """Convenience accessor that returns a float Series or empty Series."""
        return df[name] if name in df.columns else pd.Series(dtype=float)

    summary = {
        "n_anchors": df["point"].nunique(),
        "hit_puiseux": _hit_rate_col(df, "r_flip_obs"),
        "hit_grad":    _hit_rate_col(df, "flip_grad"),
        "hit_lime":    _hit_rate_col(df, "flip_lime"),
        "hit_shap":    _hit_rate_col(df, "flip_shap"),
        "med_r_puiseux": float(pd.to_numeric(_col(df, "r_flip_obs"), errors="coerce").dropna().median()) if "r_flip_obs" in df else float("nan"),
        "med_r_grad":    float(pd.to_numeric(_col(df, "flip_grad"), errors="coerce").dropna().median()) if "flip_grad" in df.columns else float("nan"),
        "med_r_lime":    float(pd.to_numeric(_col(df, "flip_lime"), errors="coerce").dropna().median()) if "flip_lime" in df.columns else float("nan"),
        "med_r_shap":    float(pd.to_numeric(_col(df, "flip_shap"), errors="coerce").dropna().median()) if "flip_shap" in df.columns else float("nan"),
    }

    pd.DataFrame([summary]).to_csv(OUT_DIR / "xai_vs_puiseux_summary.csv", index=False)
    print("[INFO] Saved head-to-head flip summary ->", OUT_DIR / "xai_vs_puiseux_summary.csv")

    # 4) Compute r_dom if missing but c2/c4 available.
    if {"r_dom", "c2_max_abs", "c4_max_abs"}.issubset(df.columns):
        mask = df["r_dom"].isna() & df["c2_max_abs"].notna() & df["c4_max_abs"].notna() & (df["c4_max_abs"] > 0)
        if mask.any():
            df.loc[mask, "r_dom"] = np.sqrt(df.loc[mask, "c2_max_abs"] / df.loc[mask, "c4_max_abs"])

    # 5) Save joined anchors.
    out_csv = OUT_DIR / "evidence_anchors_joined.csv"
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved joined anchors -> {out_csv}")

    # 6) Correlations and summary stats.
    sub = df[["r_dom", "r_flip_obs"]].dropna()
    if len(sub) >= 2:
        pear = float(sub.corr(method="pearson").iloc[0, 1])
        spear = float(sub.corr(method="spearman").iloc[0, 1])
        mae = float(np.mean(np.abs(sub["r_dom"] - sub["r_flip_obs"])))
    else:
        pear = spear = mae = float("nan")

    # Full triage curve by |c4| → PR curve + AUPRC.
    auprc_c4 = np.nan
    pr_at_topk = re_at_topk = np.nan
    f1_max_c4 = np.nan
    thr_at_f1max = np.nan

    if {"c4_max_abs", "r_flip_obs"}.issubset(df.columns):
        # Label: fragile=1 if a flip is observed within the budget (r <= BUDGET).
        # Missing flips (NaN) are treated as 0.
        y_true = ((pd.to_numeric(df["r_flip_obs"], errors="coerce").fillna(np.inf)) <= BUDGET).astype(int).to_numpy()
        scores = pd.to_numeric(df["c4_max_abs"], errors="coerce").to_numpy()

        if np.isfinite(scores).any() and y_true.sum() > 0:
            rec, prec, auprc_c4, df_curve = precision_recall_from_scores(y_true, scores)

            # F1 along the curve (skip the initial (1, 0) point).
            with np.errstate(divide="ignore", invalid="ignore"):
                f1 = 2 * prec[1:] * rec[1:] / np.clip(prec[1:] + rec[1:], 1e-12, None)
            if len(f1) > 0:
                imax = int(np.nanargmax(f1))
                f1_max_c4 = float(f1[imax])
                thr_at_f1max = float(df_curve.loc[imax, "threshold"])

            # Legacy point: top-K where K = #positives (kept for comparability).
            K = int(y_true.sum())
            if K > 0:
                tp_at_k = int(df_curve.loc[min(K, len(df_curve)) - 1, "tp"])
                pr_at_topk = tp_at_k / K
                re_at_topk = tp_at_k / K  # at top-K with K=#pos, precision == recall

            # Save CSV and the PR plot.
            curve_csv = OUT_DIR / "pr_by_abs_c4.csv"
            df_curve.to_csv(curve_csv, index=False)

            plt.figure(figsize=(6, 5))
            plt.step(rec, prec, where="post")
            pos_rate = y_true.mean()
            plt.hlines(pos_rate, 0.0, 1.0, linestyles="--", linewidth=1.0, label=f"baseline={pos_rate:.2f}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Triage by |c4|: Precision–Recall")
            plt.legend()
            pr_fig = OUT_DIR / "figures" / "pr_curve_by_abs_c4.png"
            plt.tight_layout()
            plt.savefig(pr_fig, dpi=160)
            plt.close()
            print(f"[INFO] Saved PR curve -> {pr_fig}")
            print(f"[INFO] Saved PR table -> {curve_csv}")

    def _auprc_for_score(name, scores, y_true, out_prefix):
        """
        Helper: compute AUPRC for an arbitrary score, save its PR curve (CSV + PNG),
        and return the AP value.
        """
        scores = np.asarray(scores, dtype=float)
        valid = np.isfinite(scores)
        if not valid.any() or y_true.sum() == 0:
            return np.nan
        rec2, prec2, ap2, df_curve2 = precision_recall_from_scores(y_true, scores)
        # Save curve table and figure
        (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
        df_curve2.to_csv(OUT_DIR / f"pr_by_{out_prefix}.csv", index=False)
        plt.figure(figsize=(6,5))
        plt.step(rec2, prec2, where="post", label=name)
        pos_rate2 = y_true.mean()
        plt.hlines(pos_rate2, 0.0, 1.0, linestyles="--", linewidth=1.0, label=f"baseline={pos_rate2:.2f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR: {name}")
        plt.legend(); plt.tight_layout()
        plt.savefig(OUT_DIR / "figures" / f"pr_by_{out_prefix}.png", dpi=160)
        plt.close()
        return float(ap2)

    results_pr = []
    # Include already-computed |c4| in the comparison table.
    if np.isfinite(auprc_c4):
        results_pr.append(("|c4|", float(auprc_c4)))

    EPS = 1e-9

    # 1/r scores for radii parsed from TXT.
    for col, label, pref in [
        ("flip_grad", "1/r_grad", "per_grad"),
        ("flip_lime", "1/r_lime", "per_lime"),
        ("flip_shap", "1/r_shap", "per_shap"),
    ]:
        if col in df.columns:
            r = pd.to_numeric(df[col], errors="coerce").to_numpy()
            score = 1.0 / np.where(np.isfinite(r), r + EPS, np.inf)  # NaN -> 0
            ap = _auprc_for_score(label, score, y_true, pref)
            results_pr.append((label, ap))

    # grad_norm from Saliency logs (if available).
    if "saliency_grad_norm" in df.columns:
        g = pd.to_numeric(df["saliency_grad_norm"], errors="coerce").fillna(0.0).to_numpy()
        ap = _auprc_for_score("grad_norm", g, y_true, "grad_norm")
        results_pr.append(("grad_norm", ap))

    # Save the AUPRC comparison summary.
    pd.DataFrame(results_pr, columns=["score","AUPRC"]).to_csv(
        OUT_DIR / "triage_compare_summary.csv", index=False
    )
            
    corr_df = pd.DataFrame({
        "anchors_total": [df["point"].nunique()],
        "anchors_with_flip": [int((pd.to_numeric(df.get("r_flip_obs"), errors="coerce") <= BUDGET).sum())],
        "mean_r_dom": [float(np.nanmean(df.get("r_dom")))],
        "mean_r_flip": [float(np.nanmean(pd.to_numeric(df.get("r_flip_obs"), errors="coerce")))],
        "mae_abs(r_dom,r_flip)": [mae],
        "pearson_r": [pear],
        "spearman_r": [spear],
        "precision_topk_by_c4": [pr_at_topk],
        "recall_topk_by_c4": [re_at_topk],
        "AUPRC_by_c4": [auprc_c4],
        "F1_max_by_c4": [f1_max_c4],
        "threshold_at_F1max_by_c4": [thr_at_f1max],
    })

    corr_df.to_csv(OUT_DIR / "corr_summary.csv", index=False)
    print(f"[INFO] Saved correlation summary -> {OUT_DIR / 'corr_summary.csv'}")

    # 7) Figures
    # 7a) r_dom vs r_flip with a regression line (and y=x reference).
    plt.figure(figsize=(6, 5))
    if len(sub) > 0:
        plt.scatter(sub["r_dom"], sub["r_flip_obs"], alpha=0.85)
        # Use standard least squares for a simple slope (not constrained through origin).
        try:
            X = sub["r_dom"].to_numpy().reshape(-1,1)
            y = sub["r_flip_obs"].to_numpy()
            coef = (X.T @ X + 1e-12*np.eye(1)) @ (X.T @ y)
            xg = np.linspace(0, max(sub.max()), 100)
            plt.plot(xg, coef[0]*xg, linestyle="--", label=f"slope≈{coef[0]:.2f}")
            plt.legend()
        except Exception:
            pass
        mmax = float(max(sub["r_dom"].max(), sub["r_flip_obs"].max()))
        plt.plot([0, mmax], [0, mmax], color="grey", linewidth=1)
    plt.xlabel("Predicted onset radius r_dom ≈ sqrt(|c2|/|c4|)")
    plt.ylabel("Observed min flip radius")
    plt.title("Puiseux prediction vs observed flip")
    plt.tight_layout()
    fig1 = OUT_DIR / "figures" / "scatter_rdom_vs_rflip.png"
    plt.savefig(fig1, dpi=160)
    plt.close()
    print(f"[INFO] Saved {fig1}")

    # 7b) Kink fraction vs flip radius.
    if "frac_kink" in df.columns:
        s2 = df[["frac_kink", "r_flip_obs"]].dropna()
        plt.figure(figsize=(6, 5))
        if len(s2) > 0:
            plt.scatter(s2["frac_kink"], s2["r_flip_obs"], alpha=0.85)
        plt.xlabel("Kink fraction in neighborhood")
        plt.ylabel("Observed min flip radius")
        plt.title("Non-holomorphicity vs fragility")
        plt.tight_layout()
        fig2 = OUT_DIR / "figures" / "scatter_kink_vs_rflip.png"
        plt.savefig(fig2, dpi=160)
        plt.close()
        print(f"[INFO] Saved {fig2}")

    # 8) Markdown report.
    md = OUT_DIR / "np_evidence_report.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Newton–Puiseux Evidence Report\n")
        f.write(f"- Anchors analyzed: **{df['point'].nunique()}**\n")
        n_flips = int((pd.to_numeric(df.get('r_flip_obs'), errors='coerce') < BUDGET).sum())
        f.write(f"- Anchors with observed class flip (r < {BUDGET:.3f}): **{n_flips}**\n")
        mean_rdom = np.nanmean(pd.to_numeric(df.get('r_dom'), errors='coerce'))
        mean_rflip = np.nanmean(pd.to_numeric(df.get('r_flip_obs'), errors='coerce'))
        f.write(f"- Mean predicted onset radius r_dom: **{mean_rdom:.6f}**\n")
        f.write(f"- Mean observed min flip radius: **{mean_rflip:.6f}**\n")
        f.write(f"- MAE |r_dom - r_flip|: **{mae if np.isfinite(mae) else float('nan'):.6f}**\n")
        f.write(f"- Pearson(r_dom, r_flip): **{pear if np.isfinite(pear) else 'nan'}**\n")
        f.write(f"- Spearman(r_dom, r_flip): **{spear if np.isfinite(spear) else 'nan'}**\n")
        if np.isfinite(pr_at_topk) and np.isfinite(re_at_topk):
            f.write(f"- Triage @top-K by |c4| → precision≈**{pr_at_topk:.2f}**, recall≈**{re_at_topk:.2f}**\n")
        f.write("\n## Key Figures\n")
        f.write("- Puiseux prediction vs observed flip: `figures/scatter_rdom_vs_rflip.png`\n")
        if (OUT_DIR / 'figures' / 'scatter_kink_vs_rflip.png').exists():
            f.write("- Kink fraction vs flip radius: `figures/scatter_kink_vs_rflip.png`\n")
        if np.isfinite(auprc_c4):
            f.write(f"- Triage by |c4| → AUPRC=**{auprc_c4:.3f}**, PR curve: `figures/pr_curve_by_abs_c4.png`\n")
    print(f"[INFO] Wrote report -> {md}")



if __name__ == "__main__":
    main()
