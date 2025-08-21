# Copyright (c) 2025 Piotr Migus
# This code is licensed under the MIT License.
# See the LICENSE file in the repository root for full license information.

"""
Utilities for local analysis, robustness probing, and post-hoc explainability
around complex-valued neural networks whose outputs are interpreted as two
class magnitudes (alpha_0, alpha_1).

Sections:
1) Interpretation helpers for Puiseux expansions.
2) Robustness search over random complex directions.
3) SHAP / LIME compatibility wrappers.
4) 2D local contour plotting for F = alpha0 - alpha1.
5) Robustness curve plotting utilities.
6) Binary calibration metrics and simple calibrators.
7) Heuristic "kink score" via gradient-direction variability.
"""

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import shap
import sympy
import os

from typing import Callable
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy import stats, optimize

from lime.lime_tabular import LimeTabularExplainer

# It is assumed that the following functions are defined in separate modules.
# For clarity we import them as shown below:
from src.find_up_synthetic import SimpleComplexNet, generate_synthetic_data, complex_modulus_to_logits
from src.local_analysis import (
    local_poly_approx_complex,
    puiseux_uncertain_point,
    load_uncertain_points,
    evaluate_poly_approx_quality
)

########################################################################
# 1. ENHANCED INTERPRETATION OF PUISEUX COEFFICIENTS
########################################################################
def interpret_puiseux_expansions(expansions, x_sym, y_sym):
    """
    Interpret Puiseux expansions by extracting dominant terms (by |coefficient|)
    and producing a short textual comment about their magnitude and phase.

    This is robust to fractional exponents for which `sympy.Poly` may fail
    to construct a polynomial object; a manual fallback is used in that case.

    Parameters
    ----------
    expansions : iterable
        A list/iterable of expressions convertible to `sympy.Expr`.
    x_sym, y_sym : sympy.Symbol
        Symbols corresponding to the two variables.

    Returns
    -------
    results : list[dict]
        Each dict contains:
          - "puiseux_expr": str, stringified input expression,
          - "dominant_terms": list[((px, py), complex)], top-3 by |coef|,
          - "comment": str, short human-readable summary.
    """
    results = []
    for exp_item in expansions:
        # (1) Make sure we have a sympy expression.
        expr_sym = sympy.sympify(exp_item)

        # (2) Symbolically expand.
        expanded = sympy.expand(expr_sym)

        # (3) Collect terms as ((px, py), coef). Prefer Poly; fall back if needed.
        terms_list = []
        try:
            poly_obj = sympy.Poly(expanded, x_sym, y_sym)
            for (px, py), coef in poly_obj.terms():
                c = complex(sympy.N(coef))
                terms_list.append(((px, py), c))
        except Exception:
            # Fallback: decompose into additive summands manually.
            for term in sympy.Add.make_args(expanded):
                # Extract powers of x,y; px/py can be Rational.
                pdict = term.as_powers_dict()
                px = pdict.get(x_sym, sympy.Integer(0))
                py = pdict.get(y_sym, sympy.Integer(0))
                # Coefficient is term / (x^px * y^py).
                try:
                    base = (x_sym**px) * (y_sym**py)
                    coef = term / base
                except Exception:
                    coef = term
                c = complex(sympy.N(coef))
                # Keep even when px/py are Rational — OK for interpretation.
                terms_list.append(((px, py), c))

        # (4) Sort by coefficient magnitude (descending) and keep Top-3.
        sorted_terms = sorted(terms_list, key=lambda u: abs(u[1]), reverse=True)
        top_terms = sorted_terms[:3]

        # (5) Build a concise commentary: magnitude and phase of coefficients.
        def _fmt_power(p):
            # Pretty-print integer/rational exponents.
            try:
                if sympy.Integer(p) == p:
                    return str(int(p))
            except Exception:
                pass
            return str(p)

        commentary = "Dominant coefficients: " + ", ".join(
            [f"(x^{_fmt_power(tx)}, y^{_fmt_power(ty)}): |c|={abs(c):.2e}, arg={np.angle(c):+.2f} rad"
             for (tx, ty), c in top_terms]
        )

        results.append({
            "puiseux_expr": str(expr_sym),
            "dominant_terms": top_terms,
            "comment": commentary
        })

    return results



########################################################################
# 2. EXPANDED ROBUSTNESS ANALYSIS OVER MULTIPLE DIRECTIONS
########################################################################
def find_adversarial_directions(expansion_expr, x_sym, y_sym, num_random=10, radius=0.01):
    """
    Sample random complex directions and evaluate a Puiseux-like expansion
    along them to identify potentially adversarial perturbation paths.

    Heuristic: directions whose evaluated phase is closest to ±π may signal
    sign changes in F and thus potential decision flips.

    Parameters
    ----------
    expansion_expr : sympy.Expr
        Expansion to evaluate (e.g., a Puiseux series).
    x_sym, y_sym : sympy.Symbol
        Symbols corresponding to the first and second dimensions.
    num_random : int, optional
        Number of random directional samples (default 10).
    radius : float, optional
        Complex perturbation magnitude (default 0.01).

    Returns
    -------
    top_directions : list[tuple]
        Top-3 entries of ((thx, thy), phase) sorted by closeness of phase to ±π.
    """
    best_dirs = []
    for _ in range(num_random):
        thx = 2 * np.pi * random.random()
        thy = 2 * np.pi * random.random()
        # Complex numbers with fixed radius and random phase.
        z1 = radius * np.exp(1j * thx)
        z2 = radius * np.exp(1j * thy)
        val = expansion_expr.subs({x_sym: z1, y_sym: z2})
        cval = complex(val.evalf())
        phase = np.angle(cval)
        best_dirs.append(((thx, thy), phase))
    
    # Sort by closeness to ±π (potential sign reversal).
    best_dirs = sorted(best_dirs, key=lambda x: abs(abs(x[1]) - np.pi))
    
    # Return the top-3 adversarial directions.
    top_k = 3
    return best_dirs[:top_k]


def test_adversarial_impact(model, base_point, direction_radians, radius=0.01, steps=20, device='cpu'):
    """
    Probe a model along a complex-direction perturbation and detect where
    (if anywhere) the predicted class flips.

    The model output is assumed to contain real/imag parts concatenated:
    first half = real logits, second half = imaginary logits. Class scores
    are magnitudes per class: alpha = sqrt(xr^2 + xi^2).

    Parameters
    ----------
    model : torch.nn.Module
        The neural network under test.
    base_point : np.ndarray
        1D input vector to perturb (length 4: Re/Im of two complex dims).
    direction_radians : tuple(float, float)
        Angles (thx, thy) defining complex directions for dims 1 and 2.
    radius : float, optional
        Maximum perturbation radius (default 0.01).
    steps : int, optional
        Number of interpolation steps up to `radius` (default 20).
    device : str, optional
        'cpu' or 'cuda'.

    Returns
    -------
    out_list : list[tuple]
        Sequence of (current_radius, F_value, predicted_class).
    changed_class : bool
        Whether the predicted class changed along the path.
    changed_radius : float or None
        First radius at which the class changed, else None.
    """
    out_list = []
    thx, thy = direction_radians
    changed_class = False
    base_class = None
    changed_radius = None

    for i in range(steps + 1):
        r = (i / steps) * radius
        # Perturb along the specified complex directions.
        z1 = r * np.exp(1j * thx)
        z2 = r * np.exp(1j * thy)
        # Map complex perturbation to 4D real vector [Re(z1), Re(z2), Im(z1), Im(z2)].
        x_loc = base_point.copy()
        x_loc[0] += z1.real
        x_loc[2] += z1.imag
        x_loc[1] += z2.real
        x_loc[3] += z2.imag

        # Forward pass.
        x_ten = torch.tensor(x_loc, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            out_ = model(x_ten)
            half_ = out_.shape[1] // 2
            xr_ = out_[:, :half_]
            xi_ = out_[:, half_:]
            alpha_ = torch.sqrt(xr_**2 + xi_**2 + 1e-9)
            F_ = (alpha_[:, 0] - alpha_[:, 1]).item()

            pred_class = torch.argmax(alpha_, dim=1).item()
            if i == 0:
                base_class = pred_class
            else:
                if (pred_class != base_class) and not changed_class:
                    changed_class = True
                    changed_radius = r
        
        out_list.append((r, F_, pred_class))
    
    return out_list, changed_class, changed_radius


########################################################################
# 3. COMPATIBLE FUNCTIONS FOR SHAP AND LIME EXPLANATIONS
########################################################################
def model_to_explain(x_numpy, model, device, T: torch.Tensor | None = None):
    """
    Return P(y | x) with optional temperature scaling (T >= 1).

    Assumes the model output concatenates real/imag parts; magnitudes per
    class are computed as alpha = sqrt(xr^2 + xi^2). If T is provided,
    magnitudes are divided by T before softmax.

    Parameters
    ----------
    x_numpy : np.ndarray, shape (N, 4)
        Batch of 4D inputs [Re(z1), Re(z2), Im(z1), Im(z2)].
    model : torch.nn.Module
        Model producing concatenated real/imag outputs.
    device : torch.device or str
        Device for evaluation.
    T : torch.Tensor or None
        Optional temperature (scalar tensor).

    Returns
    -------
    probs : np.ndarray, shape (N, 2)
        Softmax probabilities over two classes.
    """
    x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to(device)
    with torch.no_grad():
        out = model(x_tensor)
        half = out.shape[1] // 2
        logits = torch.sqrt(out[:, :half]**2 + out[:, half:]**2 + 1e-9)
        if T is not None:
            logits = logits / T.to(logits.device)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs



def compute_lime_explanation(model, training_data, point_np, device, class_names=["class0","class1"], num_samples=500, T=None):
    """
    Compute a LIME explanation for a single point using a probability
    predictor compatible with the model above.

    Parameters
    ----------
    model : torch.nn.Module
    training_data : np.ndarray
        Background data for LIME (tabular).
    point_np : np.ndarray, shape (4,)
        Point to explain.
    device : torch.device or str
    class_names : list[str]
        Names for classes 0/1.
    num_samples : int
        Number of LIME samples.
    T : torch.Tensor or None
        Optional temperature used inside `model_to_explain`.

    Returns
    -------
    lime_explanation : lime.explanation.Explanation
    """
    explainer = LimeTabularExplainer(
        training_data,
        feature_names=["Re(z1)", "Re(z2)", "Im(z1)", "Im(z2)"],
        class_names=class_names
    )
    def predict_fn(x):
        return model_to_explain(x, model, device, T=T)
    exp = explainer.explain_instance(point_np, predict_fn, num_samples=num_samples)
    return exp



def compute_shap_explanation(model, single_point_np, device, background=None, num_samples=100, T=None):
    """
    Compute Kernel SHAP values for a single point using a probability
    predictor compatible with the model above.

    Parameters
    ----------
    model : torch.nn.Module
    single_point_np : np.ndarray, shape (4,) or (1, 4)
        Point to explain.
    device : torch.device or str
    background : np.ndarray or None
        Background dataset for KernelExplainer (defaults to zeros).
    num_samples : int
        Number of SHAP samples (nsamples).
    T : torch.Tensor or None
        Optional temperature used inside `model_to_explain`.

    Returns
    -------
    shap_values : list[np.ndarray] or np.ndarray
        Kernel SHAP values as returned by `explainer.shap_values`.
    """
    if background is None:
        background = np.zeros((1, 4))
    explainer = shap.KernelExplainer(lambda x: model_to_explain(x, model, device, T=T), background)
    shap_values = explainer.shap_values(single_point_np, nsamples=num_samples)
    return shap_values



########################################################################
# 4. PLOTTING THE 2D LOCAL CONTOUR OF THE FUNCTION F = alpha0 - alpha1
########################################################################
def plot_local_contour_2d(model, xstar, fix_dims=(1, 3), delta=0.02, steps=50, device='cpu', save_path=None):
    """
    Plot a local 2D slice of F = alpha0 - alpha1 while holding two dimensions fixed.

    Colormap is symmetric around 0; the F=0 isoline is overlaid for clarity.

    Parameters
    ----------
    model : torch.nn.Module
    xstar : np.ndarray, shape (4,)
        Reference point around which to sample.
    fix_dims : tuple[int, int]
        Indices of the two fixed dimensions (the remaining two vary).
    delta : float
        +/- range around xstar for each varying dimension.
    steps : int
        Number of grid steps per axis.
    device : str
        'cpu' or 'cuda'.
    save_path : str or None
        If provided, save the figure; otherwise, show it.
    """
    var_dims = [d for d in [0, 1, 2, 3] if d not in fix_dims]
    if len(var_dims) != 2:
        raise ValueError("plot_local_contour_2d requires exactly 2 variable dims.")
    x_index, y_index = var_dims

    xstar_clone = xstar.copy()
    x_lin = np.linspace(xstar[x_index] - delta, xstar[x_index] + delta, steps)
    y_lin = np.linspace(xstar[y_index] - delta, xstar[y_index] + delta, steps)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = np.zeros_like(X, dtype=float)

    for i in range(steps):
        for j in range(steps):
            x_loc = xstar_clone.copy()
            x_loc[x_index] = X[i, j]
            x_loc[y_index] = Y[i, j]
            x_ten = torch.tensor(x_loc, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                out_ = model(x_ten)
                half_ = out_.shape[1] // 2
                xr_ = out_[:, :half_]
                xi_ = out_[:, half_:]
                alpha_ = torch.sqrt(xr_**2 + xi_**2 + 1e-9)
                F_ = (alpha_[:, 0] - alpha_[:, 1]).item()
            Z[i, j] = float(F_)

    vmax = np.nanmax(np.abs(Z)) if np.isfinite(Z).any() else 1.0
    plt.figure(figsize=(6, 5))
    cs = plt.contourf(X, Y, Z, levels=30, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(cs)
    plt.contour(X, Y, Z, levels=[0.0], colors='black', linestyles='--', linewidths=1.5)
    plt.scatter(xstar[x_index], xstar[y_index], s=100, c='yellow', edgecolors='k', marker='o', zorder=3)

    plt.title(f"Local contour around uncertain point (variable dims={var_dims})")
    plt.xlabel(f"Dimension {x_index}")
    plt.ylabel(f"Dimension {y_index}")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()



###############################################
# 5. Plot robustness curves
###############################################

def plot_robustness_curve(all_adv_curves, save_path=None):
    """
    Plot decision function value (F) versus perturbation radius for multiple
    adversarial directions.

    Parameters
    ----------
    all_adv_curves : list[list[tuple]]
        Each inner list contains (radius, F, predicted_class) tuples from
        `test_adversarial_impact`.
    save_path : str or None
        If provided, the plot is saved to that path; otherwise it's shown.
    """
    plt.figure(figsize=(6, 5))
    for idx, adv_list in enumerate(all_adv_curves):
        rs = [elem[0] for elem in adv_list]
        Fs = [elem[1] for elem in adv_list]
        plt.plot(rs, Fs, marker='o', label=f'Direction {idx}')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Perturbation Radius")
    plt.ylabel("Decision Function Value (F)")
    plt.title("Robustness Analysis: F vs. Perturbation Radius")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Robustness curve saved to {save_path}")
        plt.close()
    else:
        plt.show()


def time_gradient_saliency(model, xstar_np, device, T: torch.Tensor | None = None, repeat: int = 5):
    """
    Measure the cost of saliency (backprop) for F = |c1| - |c2|.

    Returns a dict with: time_ms, grad_norm, cpu_rss_mb_delta, gpu_peak_mb.
    CUDA is synchronized before/after timing; x.grad is cleared each repeat.
    grad_norm is averaged across repeats.

    Parameters
    ----------
    model : torch.nn.Module
    xstar_np : np.ndarray, shape (4,)
        Input point.
    device : torch.device or str
    T : torch.Tensor or None
        Optional temperature applied to magnitudes before F.
    repeat : int
        Number of repeated timing measurements.

    Notes
    -----
    - CPU memory (RSS) is sampled via `psutil` before/after.
    - GPU peak memory is read via `torch.cuda.max_memory_allocated` if CUDA.
    """
    import psutil, time
    model.eval()

    x = torch.tensor(xstar_np.reshape(1, -1), dtype=torch.float32, device=device, requires_grad=True)

    # Memory snapshot (CPU).
    try:
        proc = psutil.Process(os.getpid())
        rss0 = proc.memory_info().rss / (1024**2)
    except Exception:
        rss0 = float('nan')

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    times_ms = []
    grad_norms = []

    for _ in range(repeat):
        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        out = model(x)
        half = out.shape[1] // 2
        logits = torch.sqrt(out[:, :half]**2 + out[:, half:]**2 + 1e-9)
        if T is not None:
            logits = logits / T.to(logits.device)
        F = (logits[:, 0] - logits[:, 1]).sum()
        F.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1e3
        times_ms.append(dt_ms)

        grad_norms.append(float(x.grad.detach().norm().item()))

    try:
        rss1 = proc.memory_info().rss / (1024**2)
        d_rss = float(rss1 - rss0)
    except Exception:
        d_rss = float('nan')

    gpu_peak = float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else float('nan')

    return {
        "time_ms": float(np.mean(times_ms)),
        "grad_norm": float(np.mean(grad_norms)),
        "cpu_rss_mb_delta": d_rss,
        "gpu_peak_mb": gpu_peak
    }


        
def scalarize(x):
    """
    Convert a numpy array containing a single value into a Python scalar.

    Parameters
    ----------
    x : np.ndarray

    Returns
    -------
    scalar : float or int
        If not inherently scalar, the first element is returned.
    """
    sx = np.squeeze(x)
    return sx.item() if np.isscalar(sx) or sx.size == 1 else float(sx.flat[0])



########################################################################
# 6) CALIBRATION: binary metrics + CIs + simple calibrators
########################################################################


def _clip01(p, eps: float = 1e-12):
    p = np.asarray(p, dtype=float).ravel()
    return np.clip(p, eps, 1.0 - eps)

def _logit(p):
    p = _clip01(p)
    return np.log(p) - np.log(1.0 - p)

def ece_binary(p, y, n_bins: int = 15) -> float:
    p = _clip01(p)
    y = np.asarray(y, dtype=int).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(p, bins) - 1  # [0..n_bins-1]
    ece = 0.0
    n = len(y)
    for b in range(n_bins):
        mask = inds == b
        if not np.any(mask):
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)

def brier_binary(p, y) -> float:
    p = _clip01(p)
    y = np.asarray(y, dtype=float).ravel()
    return float(np.mean((p - y) ** 2))

def nll_binary(p, y) -> float:
    p = _clip01(p)
    y = np.asarray(y, dtype=float).ravel()
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

def compute_binary_metrics(p, y, n_bins: int = 15):
    return {
        "ECE": ece_binary(p, y, n_bins=n_bins),
        "Brier": brier_binary(p, y),
        "NLL": nll_binary(p, y),
    }

def fit_binary_calibrator(method: str,
                          p_train: np.ndarray,
                          y_train: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a calibrator function `calibrate(p) -> p_cal` for one of:
    'none' | 'platt' | 'isotonic' | 'beta' | 'vector' | 'temperature'

    Note: in binary classification, 'vector' ≡ 'platt'.

    Parameters
    ----------
    method : str
        Calibration method name (case-insensitive).
    p_train : np.ndarray
        Uncalibrated probabilities for the positive class.
    y_train : np.ndarray
        Binary labels.

    Returns
    -------
    calibrate : Callable[[np.ndarray], np.ndarray]
        Function that maps probabilities to calibrated probabilities.
    """
    method = method.lower()
    p_train = _clip01(p_train)
    y_train = np.asarray(y_train, dtype=int).ravel()

    if method in ("platt", "vector"):
        Z = _logit(p_train).reshape(-1, 1)
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(Z, y_train)
        def calibrate(p):
            z = _logit(p).reshape(-1, 1)
            s = lr.decision_function(z)  # w*z + b
            return 1.0 / (1.0 + np.exp(-s))
        return calibrate

    elif method == "beta":
        X = np.column_stack((np.log(p_train), np.log(1.0 - p_train)))
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(X, y_train)
        def calibrate(p):
            p = _clip01(p)
            Z = np.column_stack((np.log(p), np.log(1.0 - p)))
            s = lr.decision_function(Z)
            return 1.0 / (1.0 + np.exp(-s))
        return calibrate

    elif method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_train, y_train)
        def calibrate(p):
            return iso.predict(_clip01(p))
        return calibrate

    elif method in ("temperature", "temp"):
        z = _logit(p_train)
        def objective(t):
            t = float(t)
            q = 1.0 / (1.0 + np.exp(-z / t))
            return -np.mean(y_train * np.log(_clip01(q)) + (1 - y_train) * np.log(_clip01(1 - q)))
        res = optimize.minimize_scalar(objective, bounds=(0.05, 10.0),
                                       method="bounded", options={"xatol": 1e-4})
        T = float(res.x)
        def calibrate(p):
            z = _logit(p)
            return 1.0 / (1.0 + np.exp(-z / T))
        calibrate.T_ = T  # for reporting
        return calibrate

    elif method in ("none", "identity"):
        def calibrate(p):
            return _clip01(p)
        return calibrate

    raise ValueError(f"Unknown calibration method: {method}")

def mean_ci_t(values, alpha: float = 0.05):
    """Mean and (1 - alpha) CI via Student's t-distribution over folds/runs."""
    v = np.asarray(values, dtype=float).ravel()
    n = len(v)
    m = float(np.mean(v))
    if n < 2:
        return m, m, m
    s = float(np.std(v, ddof=1))
    hw = stats.t.ppf(1 - alpha/2, df=n-1) * s / np.sqrt(n)
    return m, m - hw, m + hw

def format_mean_ci(mean, lo, hi, decimals: int = 4):
    half = abs(hi - lo) / 2.0
    return f"{mean:.{decimals}f} ± {half:.{decimals}f}"


########################################################################
# 7) Heuristic "kink score": variance of gradient direction
########################################################################
def kink_score(model, xstar, radius: float = 1e-2, samples: int = 24, device='cpu') -> float:
    """
    A simple non-smoothness heuristic: sample gradients of F in random nearby
    directions and compute the standard deviation of the angle to the mean
    gradient direction. Larger values indicate more "kinky" (less smooth)
    local behavior.

    Parameters
    ----------
    model : torch.nn.Module
    xstar : array-like, shape (4,)
        Base point.
    radius : float
        Perturbation radius in complex plane for each of the two dims.
    samples : int
        Number of random directions.
    device : str
        'cpu' or 'cuda'.

    Returns
    -------
    float
        Standard deviation of the angular deviations (radians).
    """
    import torch, numpy as np
    xstar = np.asarray(xstar, dtype=np.float32)

    def grad_at(xvec):
        # Make a leaf tensor; clone+detach+requires_grad ensures leaf after unsqueeze.
        x_leaf = torch.tensor(xvec, dtype=torch.float32, device=device).unsqueeze(0)
        x_leaf = x_leaf.clone().detach().requires_grad_(True)

        out = model(x_leaf)
        half = out.shape[1] // 2
        xr, xi = out[:, :half], out[:, half:]
        alpha = torch.sqrt(xr**2 + xi**2 + 1e-9)
        F = (alpha[:, 0] - alpha[:, 1]).sum()

        # Compute gradient w.r.t. inputs directly.
        g = torch.autograd.grad(F, x_leaf, retain_graph=False, create_graph=False)[0]
        return g.detach().cpu().numpy().ravel()


    grads = []
    for _ in range(samples):
        thx, thy = 2*np.pi*np.random.rand(), 2*np.pi*np.random.rand()
        z1 = radius*np.exp(1j*thx); z2 = radius*np.exp(1j*thy)
        x = xstar.copy()
        x[0]+=z1.real; x[2]+=z1.imag; x[1]+=z2.real; x[3]+=z2.imag
        grads.append(grad_at(x))

    G = np.stack(grads, axis=0)  # [samples, 4]
    U = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)
    mean_dir = U.mean(axis=0, keepdims=True)
    cosang = np.sum(U * mean_dir, axis=1).clip(-1, 1)
    ang = np.arccos(cosang)
    return float(ang.std())  # the larger, the more "kinked"



def multiplicity_to_T_multiplier(m_true: float, m_est: float, gamma: float = 0.5) -> float:
    """
    Heuristic mapping from branch multiplicity mismatch to temperature factor.

    Assumes T ∝ m^{-gamma}. For gamma=1/2: T ~ 1 / sqrt(m).
    Returns T_mult = (m_true / m_est) ** gamma.

    Parameters
    ----------
    m_true : float
        True multiplicity (or a proxy; enforced >= 1e-6).
    m_est : float
        Estimated multiplicity (enforced >= 1e-6).
    gamma : float
        Exponent controlling sensitivity.

    Returns
    -------
    float
        Temperature multiplier.
    """
    m_true = max(float(m_true), 1e-6)
    m_est  = max(float(m_est),  1e-6)
    return (m_true / m_est) ** float(gamma)


def sweep_multiplicity_misestimation(
    model, X_full, y_full, compress_fn, scaler, device, T_base: torch.Tensor | None,
    rel_errors=( -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5), gamma=0.5
):
    """
    Evaluate ECE sensitivity to multiplicity misestimation.

    For a relative error eps (m_est = m_true * (1 + eps)), compute the
    ECE of class-1 probabilities after rescaling temperature by
    T_mult = (m_true / m_est) ** gamma. Returns [(eps, ECE)].

    Parameters
    ----------
    model : torch.nn.Module
    X_full : np.ndarray
        Full dataset (before compression/scaling).
    y_full : np.ndarray
        Binary labels.
    compress_fn : Callable
        Function mapping X_full -> compressed features for the model.
    scaler : sklearn-like or None
        Optional scaler with `.transform`.
    device : torch.device or str
    T_base : torch.Tensor or None
        Base temperature tensor; if None, no temperature scaling is used.
    rel_errors : iterable of float
        Relative errors eps to test.
    gamma : float
        Exponent in the multiplicity-to-temperature mapping.

    Returns
    -------
    list[tuple[float, float]]
        Pairs (eps, ECE) for each tested relative error.
    """
    # Prepare (flattened) evaluation set.
    Xc2 = compress_fn(X_full)
    Xc2 = scaler.transform(Xc2) if scaler is not None else Xc2

    def probs_at_Tmult(Tmult):
        T_eff = None if T_base is None else (T_base.to(device) * float(Tmult))
        p = model_to_explain(Xc2, model, device, T=T_eff)
        return p[:, 1]

    results = []
    for eps in rel_errors:
        m_est_mult = 1.0 + float(eps)
        Tmult = multiplicity_to_T_multiplier(m_true=1.0, m_est=m_est_mult, gamma=gamma)
        p1 = probs_at_Tmult(Tmult)
        e = ece_binary(p1, y_full)
        results.append((eps, float(e)))
    return results
