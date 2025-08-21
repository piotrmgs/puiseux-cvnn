# Copyright (c) 2025 Piotr Migus
# This code is licensed under the MIT License.
# See the LICENSE file in the repository root for full license information.


############################################################
# Experiment with a toy problem in C^2 (4-dimensional real space):
#
# 1) Load uncertain points from a CSV file (`uncertain_synthetic.csv`)
#    and a pretrained neural network model (`model_parameters.pt`).
#
# 2) For selected points, build a local polynomial approximation `F`
#    (difference of logit outputs) as a polynomial of degree ≥ 4
#    in a 4D space: (Re(z1), Re(z2), Im(z1), Im(z2)).
#
# 3) Apply `compute_puiseux` (from puis.py) to obtain local Puiseux
#    expansions around those points.
#
# Notes:
# - Default degree=4 and n_samples=200 (configurable).
# - `F` is evaluated locally around `xstar` via random perturbations
#   inside the cube [-delta, delta]^4 in R^4 and mapped to C^2.
############################################################

# --- Determinism: make runs reproducible ---
import os as _os, random as _random, numpy as _np, torch as _torch
_os.environ.setdefault("PYTHONHASHSEED", "12345")
_random.seed(12345); _np.random.seed(12345)
_torch.manual_seed(12345); _torch.cuda.manual_seed_all(12345)
try:
    _torch.backends.cudnn.deterministic = True
    _torch.backends.cudnn.benchmark = False
    _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if _torch.cuda.is_available():
        _torch.backends.cuda.matmul.allow_tf32 = False
        _torch.backends.cudnn.allow_tf32 = False
    _torch.use_deterministic_algorithms(True)
except Exception:
    pass
del _os, _random, _np, _torch

import csv
import time
import ast
import numpy as np
import sympy
import torch
from sympy import factor

# Custom modules:
#  - SimpleComplexNet (complex-valued neural model),
#  - complex_modulus_to_logits (compute logit moduli),
#  - puiseux_expansions (local Puiseux expansions).
from src.find_up_synthetic import SimpleComplexNet, complex_modulus_to_logits  # noqa: F401
from src.puiseux import puiseux_expansions


# ---------------------------------------------------------------------
# 0) Kink / non-holomorphicity diagnostics
# ---------------------------------------------------------------------
def estimate_nonholomorphic_fraction(model, xstar, delta=0.01, n_samples=2000,
                                     kink_eps=1e-6, device='cpu'):
    """
    Estimate the fraction of perturbations for which a 'kink' occurs in the first
    layer (fc1 + modReLU):

        shifted = sqrt(xr^2 + xi^2) + bias_modrelu  ≤  kink_eps

    Parameters
    ----------
    model : torch.nn.Module
        CVNN with a first linear layer named `fc1` followed by modReLU-like nonlinearity.
    xstar : array-like of shape (4,)
        Base point in R^4 interpreted as (Re z1, Re z2, Im z1, Im z2).
    delta : float
        Side half-length of the sampling cube [-delta, delta]^4 around `xstar`.
    n_samples : int
        Number of random perturbations to evaluate.
    kink_eps : float
        Threshold for detecting proximity to the modReLU 'kink'.
    device : {"cpu","cuda"} or torch.device
        Torch device for evaluation.

    Returns
    -------
    dict
        {'frac_kink', 'frac_active', 'frac_inactive', 'n_samples', 'kink_eps'}
    """
    model.eval()
    xstar = np.asarray(xstar, dtype=np.float32)

    shifts = (2 * delta) * np.random.rand(n_samples, 4).astype(np.float32) - delta
    X = xstar.reshape(1, 4) + shifts
    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    with torch.no_grad():
        H = model.fc1(X_t)  # [N, 2*hidden]
        half = H.size(1) // 2
        xr, xi = H[:, :half], H[:, half:]
        mag = torch.sqrt(torch.clamp(xr**2 + xi**2, min=1e-12))
        bias = getattr(model, "bias_modrelu", 0.0)
        if not isinstance(bias, torch.Tensor):
            bias = torch.tensor(bias, device=mag.device)
        shifted = mag + bias

        kink_mask = (shifted <= float(kink_eps))
        any_kink = kink_mask.any(dim=1)
        frac = float(any_kink.float().mean().item())

        active_mask = (shifted > 0)
        frac_active = float(active_mask.float().mean().item())
        frac_inactive = float((~active_mask).float().mean().item())

    return {
        "frac_kink": frac,
        "frac_active": frac_active,
        "frac_inactive": frac_inactive,
        "n_samples": int(n_samples),
        "kink_eps": float(kink_eps),
    }


# ---------------------------------------------------------------------
# 1) Benchmark: local polynomial + Puiseux (with timing)
# ---------------------------------------------------------------------
def benchmark_local_poly_approx_and_puiseux(
    model,
    xstar,
    local_poly_func,
    puiseux_func,
    delta=0.01,
    degree=4,
    n_samples=200,
    device='cpu',
    do_factor=True,
    do_simplify=True,
    puiseux_prec=4
):
    """
    Time the main stages:
      - local sampling and LSQ fit (via `local_poly_func`)
      - optional factor/simplify of the symbolic polynomial
      - Puiseux expansions (via `puiseux_func`)

    Parameters
    ----------
    model : torch.nn.Module
    xstar : array-like (4,)
        Base point in R^4.
    local_poly_func : callable
        Function building the local polynomial; must accept
        (model, xstar, delta, degree, n_samples, device, return_diag=True)
        and return (expr, diag).
    puiseux_func : callable
        Function computing Puiseux expansions: (expr, x, y, prec) -> expansions.
    delta, degree, n_samples, device : see local_poly_func
    do_factor, do_simplify : bool
        Apply `sympy.factor` / `sympy.simplify` before Puiseux.
    puiseux_prec : int
        Expansion precision passed to `puiseux_func`.

    Returns
    -------
    times : dict
        Stage timings and resource snapshots.
    expr : sympy.Expr
        (Optionally simplified/factored) polynomial.
    expansions : Any
        Return value of `puiseux_func`.
    """
    x, y = sympy.symbols('x y')

    # Reset GPU peak stats & sync before timing (if applicable)
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if torch.cuda.is_available() and dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize()

    t0 = time.time()

    # 1) Local polynomial + diagnostics (sampling and fit timings)
    expr, diag = local_poly_func(
        model=model,
        xstar=xstar,
        delta=delta,
        degree=degree,
        n_samples=n_samples,
        device=device,
        return_diag=True
    )

    if torch.cuda.is_available() and dev.type == "cuda":
        torch.cuda.synchronize()
    time_sampling = diag.get("time_sampling", float("nan"))
    time_lstsq = diag.get("time_fit", float("nan"))

    # 2) Optional factor/simplify
    time_factor = 0.0
    time_simplify = 0.0
    if do_factor:
        tf0 = time.time()
        expr = sympy.factor(expr)
        if torch.cuda.is_available() and dev.type == "cuda":
            torch.cuda.synchronize()
        time_factor = time.time() - tf0
    if do_simplify:
        ts0 = time.time()
        expr = sympy.simplify(expr)
        if torch.cuda.is_available() and dev.type == "cuda":
            torch.cuda.synchronize()
        time_simplify = time.time() - ts0

    # 3) Puiseux expansions
    tp0 = time.time()
    expansions = puiseux_func(expr, x, y, puiseux_prec)
    if torch.cuda.is_available() and dev.type == "cuda":
        torch.cuda.synchronize()
    time_puiseux = time.time() - tp0

    time_total = time.time() - t0

    # Resource snapshot
    try:
        import psutil, os
        cpu_rss_mb = float(psutil.Process(os.getpid()).memory_info().rss / (1024**2))
    except Exception:
        cpu_rss_mb = float('nan')
    gpu_peak_mb = (float(torch.cuda.max_memory_allocated(dev) / (1024**2))
                   if torch.cuda.is_available() and dev.type == "cuda" else float('nan'))

    times = {
        'time_sampling': float(time_sampling),
        'time_lstsq': float(time_lstsq),
        'time_factor': float(time_factor),
        'time_simplify': float(time_simplify),
        'time_puiseux': float(time_puiseux),
        'time_total': float(time_total),
        'cpu_rss_mb': cpu_rss_mb,
        'gpu_peak_mb': gpu_peak_mb,
    }
    return times, expr, expansions


# ---------------------------------------------------------------------
# 2) Approximation quality (fast version via lambdify)
# ---------------------------------------------------------------------
def evaluate_poly_approx_quality(model,
                                 poly_expr,
                                 xstar,
                                 delta=0.01,
                                 n_samples=500,
                                 device='cpu'):
    """
    Evaluate the quality of polynomial approximation (`poly_expr`) against
    the true function f(x) = alpha_0(x) - alpha_1(x) locally around `xstar`.

    Returns
    -------
    dict
        {'RMSE', 'MAE', 'corr_pearson', 'sign_agreement'}.
    """
    x_sym, y_sym = sympy.symbols('x y')
    f_num = sympy.lambdify((x_sym, y_sym), poly_expr, modules=["numpy"])

    z1_star = xstar[0] + 1j * xstar[2]
    z2_star = xstar[1] + 1j * xstar[3]

    shifts = (2 * delta) * np.random.rand(n_samples, 4) - delta
    fvals_list, pvals_list = [], []

    for shift in shifts:
        x_loc = xstar + shift
        # 1) f(x)
        x_ten = torch.tensor(x_loc, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            out = model(x_ten)
            half = out.shape[1] // 2
            xr, xi = out[:, :half], out[:, half:]
            alpha = torch.sqrt(xr**2 + xi**2 + 1e-9)
            f_val = (alpha[:, 0] - alpha[:, 1]).item()
        fvals_list.append(f_val)

        # 2) p(x) — local shift then lambdify evaluation
        z1 = (x_loc[0] + 1j * x_loc[2]) - z1_star
        z2 = (x_loc[1] + 1j * x_loc[3]) - z2_star
        p_val = f_num(z1, z2)
        pvals_list.append(np.real(p_val))

    fvals_arr = np.array(fvals_list)
    pvals_arr = np.array(pvals_list)

    rmse = np.sqrt(np.mean((pvals_arr - fvals_arr) ** 2))
    mae = np.mean(np.abs(pvals_arr - fvals_arr))
    corr = (np.corrcoef(fvals_arr, pvals_arr)[0, 1]
            if np.std(pvals_arr) * np.std(fvals_arr) > 1e-12 else 0.0)
    sign_agreement = np.mean((fvals_arr > 0) == (pvals_arr > 0))

    return {
        'RMSE': float(rmse),
        'MAE': float(mae),
        'corr_pearson': float(corr),
        'sign_agreement': float(sign_agreement),
    }


# ---------------------------------------------------------------------
# 3) Polynomial feature builder (optional helper)
# ---------------------------------------------------------------------
def polynomial_features_complex(z, degree=4, remove_linear=False):
    """
    Return a list of monomials z1^i * z2^j for i + j ≤ degree.

    Parameters
    ----------
    z : tuple (z1, z2)
        Complex variables.
    degree : int
        Maximum total degree.
    remove_linear : bool
        If True, remove constant and linear terms (keep only total degree ≥ 2).
    """
    x, y = z
    feats = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            if remove_linear and (i + j < 2):
                continue
            feats.append(x**i * y**j)
    return feats


# ---------------------------------------------------------------------
# 4) LSQ in C^2: WLS + Tikhonov (ridge) + conditioning/residual diagnostics
# ---------------------------------------------------------------------
# (duplicated banner kept intentionally for parity with the original notes)
def fit_polynomial_complex(z_vals, Fvals, degree=4, remove_linear=False,
                           weights=None, ridge=1e-8):
    """
    Fit a polynomial in C^2 using weighted least squares (WLS) with Tikhonov (ridge).

    Parameters
    ----------
    z_vals : list[tuple(complex, complex)]
        Sampled local offsets (z1, z2) relative to the base point.
    Fvals : array-like of complex
        Target values F(z1, z2) for each sample.
    degree : int
        Maximum total degree i + j ≤ degree.
    remove_linear : bool
        If True, remove constant and linear terms from the basis.
    weights : array-like or None
        Optional positive weights; rows are scaled by sqrt(w).
    ridge : float
        Ridge strength; if > 0, augments normal equations with sqrt(ridge) * I.

    Returns
    -------
    coeffs : np.ndarray (complex)
        Fitted coefficients aligned with the constructed monomial basis.
    expr : sympy.Expr
        Symbolic polynomial (factored), simplified and shifted so that P(0,0)=0
        is enforced downstream by the caller when needed.
    info_dict : dict
        {
          'cond'      : condition number of A,
          'rank'      : rank(A),
          'n_monos'   : number of monomials,
          'resid_mean': mean |residual|,
          'resid_std' : std  |residual|,
          'resid_skew': skewness of |residual|,
          'resid_kurt': excess kurtosis of |residual|
        }

    Notes
    -----
    - Issues a console warning if A is ill-conditioned or rank-deficient.
    """
    # Design matrix
    matA = []
    monomials = None
    for z in z_vals:
        feats = []
        current_monos = []
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                if remove_linear and (i + j < 2):
                    continue
                feats.append(z[0]**i * z[1]**j)
                current_monos.append((i, j))
        if monomials is None:
            monomials = current_monos
        matA.append(feats)

    A = np.asarray(matA, dtype=np.complex128)
    b = np.asarray(Fvals, dtype=np.complex128)

    # Weights (WLS)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        w = np.clip(w, 1e-8, np.inf)
        sw = np.sqrt(w)
        A = (sw[:, None] * A)
        b = (sw * b)

    # Ridge: [A; sqrt(lambda) I], [b; 0]
    n_cols = A.shape[1] if A.ndim == 2 else 0
    if ridge and ridge > 0 and n_cols > 0:
        Ar = np.vstack([A, np.sqrt(ridge) * np.eye(n_cols, dtype=np.complex128)])
        br = np.concatenate([b, np.zeros(n_cols, dtype=np.complex128)])
    else:
        Ar, br = A, b

    coeffs, _, rank_aug, s_aug = np.linalg.lstsq(Ar, br, rcond=None)

    # Conditioning and rank diagnostics on the original A
    condA = (np.linalg.cond(A) if A.size > 0 else np.inf)
    rankA = (np.linalg.matrix_rank(A) if A.size > 0 else 0)
    if condA > 1e8:
        print(f"[WARN] Condition number is large: {condA:.2e}. Fit may be unstable (degree={degree}).")
    if rankA < n_cols:
        print(f"[WARN] Rank-deficient system: rank(A)={rankA} < {n_cols} monomials. Potential over/underfitting.")

    # Residual diagnostics (on the weighted A if weights were applied)
    try:
        b_hat = A @ coeffs[:A.shape[1]]
        r = b - b_hat
        r_abs = np.abs(r)
        resid_mean = float(np.mean(r_abs)) if r_abs.size else float('nan')
        resid_std  = float(np.std(r_abs, ddof=1)) if r_abs.size > 1 else 0.0
        if r_abs.size >= 3:
            m = np.mean(r_abs); s = np.std(r_abs, ddof=1) + 1e-12
            resid_skew = float(np.mean(((r_abs - m)/s)**3))
            resid_kurt = float(np.mean(((r_abs - m)/s)**4) - 3.0)
        else:
            resid_skew, resid_kurt = float('nan'), float('nan')
    except Exception:
        resid_mean = float('nan'); resid_std = float('nan')
        resid_skew = float('nan'); resid_kurt = float('nan')

    # Assemble symbolic polynomial
    x, y = sympy.symbols('x y')
    expr = 0
    for idx, (i, j) in enumerate(monomials):
        expr += coeffs[idx] * x**i * y**j
    expr = sympy.simplify(expr)

    info_dict = dict(
        cond=float(condA),
        rank=int(rankA),
        n_monos=int(len(monomials)),
        resid_mean=resid_mean,
        resid_std=resid_std,
        resid_skew=resid_skew,
        resid_kurt=resid_kurt
    )
    return coeffs, factor(expr), info_dict


# ---------------------------------------------------------------------
# 5) Load uncertain points (safe list parser)
# ---------------------------------------------------------------------
def load_uncertain_points(csv_path):
    """
    Load uncertain samples from CSV.

    Expected columns
    ----------------
    'index' : int
    'X'     : stringified Python list with 4 floats
    'true_label' : int
    'p1', 'p2'   : float probabilities for the two classes

    Returns
    -------
    list[dict]
        Each dict contains {'index','X','true_label','p1','p2'} with X as a list of 4 floats.
    """
    up_list = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vals = ast.literal_eval(row['X'])
            X_vals = [float(v) for v in vals]
            if len(X_vals) != 4:
                raise ValueError(f"Expected 4 real dims in 'X', got {len(X_vals)}")

            up = {
                'index': int(row['index']),
                'X': X_vals,
                'true_label': int(row['true_label']),
                'p1': float(row['p1']),
                'p2': float(row['p2'])
            }
            up_list.append(up)
    return up_list


# ---------------------------------------------------------------------
# 6) Local polynomial (kink-robust): filtering, weights, retries, degree fallback
# ---------------------------------------------------------------------
# (duplicated banner kept intentionally for parity with the original notes)
def local_poly_approx_complex(
    model,
    xstar,
    delta=0.01,
    degree=4,
    n_samples=2000,
    device='cpu',
    remove_linear=True,
    exclude_kink_eps=1e-6,
    weight_by_distance=True,
    return_diag=False,
    min_keep_ratio=0.25,
    max_retries=1,
    ridge=1e-8,
):
    """
    Build a local polynomial F = (|c1| - |c2|) around `xstar` in C^2.

    Robustness tricks:
    - `exclude_kink_eps`: drop samples near modReLU kinks in fc1 (|z| + b <= eps).
    - `weight_by_distance`: row weights ~ min(shifted) so samples farther from kinks get more weight.
    - `retry`: if too few clean samples, re-sample in a smaller cube and augment.
    - degree fallback: iteratively decrease the degree until rank/conditioning are stable.

    Parameters
    ----------
    model : torch.nn.Module
    xstar : array-like (4,)
        Base point in R^4.
    delta : float
        Side half-length of the initial sampling cube.
    degree : int
        Initial max total degree; may be reduced if the system is unstable.
    n_samples : int
        Number of initial perturbations.
    device : {"cpu","cuda"} or torch.device
    remove_linear : bool
        Remove constant/linear monomials from the basis (focus on non-linear terms).
    exclude_kink_eps : float
        Threshold for detecting and excluding kink-near samples.
    weight_by_distance : bool
        If True, use min(shifted) as WLS weights (favoring points farther from kinks).
    return_diag : bool
        If True, return (expr, diagnostics_dict); otherwise return expr only.
    min_keep_ratio : float
        Minimum fraction of kept samples; otherwise perform a retry in a smaller cube.
    max_retries : int
        Number of retries with a smaller cube if keep ratio is too low.
    ridge : float
        Tikhonov regularization strength for LSQ.

    Returns
    -------
    sympy.Expr or (sympy.Expr, dict)
        Local polynomial (shifted so that P(0,0)=0) and diagnostics if requested.
    """
    t0 = time.time()
    model.eval()
    xstar = np.asarray(xstar, dtype=np.float32)
    x_sym, y_sym = sympy.symbols('x y')

    # 1) Sample perturbations around xstar
    shifts = (2 * delta) * np.random.rand(n_samples, 4).astype(np.float32) - delta
    X_loc = xstar.reshape(1, 4) + shifts
    X_t = torch.tensor(X_loc, dtype=torch.float32, device=device)

    # 2) Forward pass: compute F and kink mask in fc1
    with torch.no_grad():
        out = model(X_t)  # [N, 2*out_features]
        half = out.shape[1] // 2
        xr, xi = out[:, :half], out[:, half:]
        alpha = torch.sqrt(torch.clamp(xr**2 + xi**2, min=1e-9))
        Fvals = (alpha[:, 0] - alpha[:, 1]).cpu().numpy()

        H = model.fc1(X_t)  # [N, 2*hidden]
        hhalf = H.size(1) // 2
        hr, hi = H[:, :hhalf], H[:, hhalf:]
        mag = torch.sqrt(torch.clamp(hr**2 + hi**2, min=1e-12))
        bias = getattr(model, "bias_modrelu", 0.0)
        if not isinstance(bias, torch.Tensor):
            bias = torch.tensor(bias, device=mag.device)
        shifted = mag + bias  # kink if <= 0

    time_sampling = time.time() - t0

    # 3) Map to local complex offsets (z1, z2) relative to xstar
    z1_star = xstar[0] + 1j * xstar[2]
    z2_star = xstar[1] + 1j * xstar[3]
    z_samps = []
    for p in X_loc:
        z1 = (p[0] + 1j * p[2]) - z1_star
        z2 = (p[1] + 1j * p[3]) - z2_star
        z_samps.append((z1, z2))

    # 4) Filter by kink proximity; construct weights
    kink_eps = float(exclude_kink_eps)
    mask_kink_any = (shifted <= kink_eps).any(dim=1).cpu().numpy()
    keep = ~mask_kink_any
    z_kept = [z for z, m in zip(z_samps, keep) if m]
    F_kept = [f for f, m in zip(Fvals, keep) if m]

    if weight_by_distance:
        min_shift = shifted.min(dim=1).values.cpu().numpy()
        w_all = np.maximum(min_shift, 0.0) + 1e-9
        w_kept = [w for w, m in zip(w_all, keep) if m]
    else:
        w_kept = None

    # Retry in a smaller cube if too few clean samples remain
    attempt = 0
    while (len(z_kept) < 10 or (len(z_kept) / max(n_samples, 1)) < min_keep_ratio) and attempt < max_retries:
        attempt += 1
        shifts2 = (2 * (0.5 * delta)) * np.random.rand(n_samples, 4).astype(np.float32) - (0.5 * delta)
        X_loc2 = xstar.reshape(1, 4) + shifts2
        X_t2 = torch.tensor(X_loc2, dtype=torch.float32, device=device)
        with torch.no_grad():
            H2 = model.fc1(X_t2)
            hhalf2 = H2.size(1) // 2
            hr2, hi2 = H2[:, :hhalf2], H2[:, hhalf2:]
            mag2 = torch.sqrt(torch.clamp(hr2**2 + hi2**2, min=1e-12))
            shifted2 = mag2 + bias
            out2 = model(X_t2)
            half2 = out2.shape[1] // 2
            xr2, xi2 = out2[:, :half2], out2[:, half2:]
            alpha2 = torch.sqrt(torch.clamp(xr2**2 + xi2**2, min=1e-9))
            Fvals2 = (alpha2[:, 0] - alpha2[:, 1]).cpu().numpy()
        mask2 = (shifted2 <= kink_eps).any(dim=1).cpu().numpy()
        keep2 = ~mask2
        for p, f, m in zip(X_loc2, Fvals2, keep2):
            if m:
                z1 = (p[0] + 1j * p[2]) - z1_star
                z2 = (p[1] + 1j * p[3]) - z2_star
                z_kept.append((z1, z2))
                F_kept.append(f)
        if weight_by_distance:
            min_shift2 = shifted2.min(dim=1).values.cpu().numpy()
            w_kept += [w for w, m in zip(np.maximum(min_shift2, 0.0) + 1e-9, keep2) if m]

    kept_ratio = float(len(F_kept) / max(n_samples, 1))

    # 5) Fit (WLS + ridge) with degree fallback until stable rank/conditioning
    t1 = time.time()
    expr = None
    info = {}
    used_degree = int(degree)
    # Try from 'degree' down to 2
    for d in range(int(degree), 1, -1):
        _, expr_try, info_try = fit_polynomial_complex(
            z_kept, F_kept, degree=d, remove_linear=remove_linear, weights=w_kept, ridge=ridge
        )
        stable = (info_try.get("rank", 0) == info_try.get("n_monos", 0)) and (info_try.get("cond", np.inf) < 1e10)
        # Initialize or accept stable solution
        if expr is None:
            expr, info, used_degree = expr_try, info_try, d
        if stable:
            expr, info, used_degree = expr_try, info_try, d
            break
        # If still unstable — keep the better-conditioned candidate
        if info_try.get("cond", np.inf) < info.get("cond", np.inf):
            expr, info, used_degree = expr_try, info_try, d

    # Enforce P(0,0)=0 (shift constant term)
    expr = sympy.simplify(expr - expr.subs({x_sym: 0, y_sym: 0}))
    time_fit = time.time() - t1

    diag = {
        "time_sampling": float(time_sampling),
        "time_fit": float(time_fit),
        "n_total": int(n_samples),
        "n_kept": int(len(F_kept)),
        "kept_ratio": kept_ratio,
        "cond": info.get("cond", float("nan")),
        "rank": info.get("rank", -1),
        "n_monomials": info.get("n_monos", -1),
        "kink_eps": float(kink_eps),
        "degree_used": int(used_degree),
        "retry": int(attempt),
        # Residual diagnostics from LSQ
        "resid_mean": info.get("resid_mean", float("nan")),
        "resid_std": info.get("resid_std", float("nan")),
        "resid_skew": info.get("resid_skew", float("nan")),
        "resid_kurt": info.get("resid_kurt", float("nan")),
    }
    return (expr, diag) if return_diag else expr


# ---------------------------------------------------------------------
# 7) Puiseux expansions with optional shift to (z1*, z2*)
# ---------------------------------------------------------------------
def puiseux_uncertain_point(F_hat_expr, prec=4, base_point=None):
    """
    Compute Newton–Puiseux expansions for F_hat_expr(x,y).

    Parameters
    ----------
    F_hat_expr : sympy.Expr
        Local polynomial in the shifted coordinates (x = z1 - z1*, y = z2 - z2*).
    prec : int
        Expansion precision/length passed to `puiseux_expansions`.
    base_point : array-like of shape (4,), optional
        If provided, return expansions shifted back to global coordinates by
        substituting x -> x + z1*, y -> y + z2*, where:
            z1* = base_point[0] + i * base_point[2]
            z2* = base_point[1] + i * base_point[3]

    Returns
    -------
    list[str]
        Stringified sympy expressions of the Puiseux branches (shifted if requested).
    """
    x, y = sympy.symbols('x y')
    exps = puiseux_expansions(F_hat_expr, x, y, prec)

    out = []
    if base_point is not None:
        z1_star = base_point[0] + 1j * base_point[2]
        z2_star = base_point[1] + 1j * base_point[3]
        for e in exps:
            e_sym = e if isinstance(e, sympy.Expr) else sympy.sympify(e)
            e_shift = sympy.expand(e_sym.subs({x: x + z1_star, y: y + z2_star}))
            out.append(str(e_shift))
    else:
        for e in exps:
            out.append(str(e if isinstance(e, sympy.Expr) else sympy.sympify(e)))
    return out
