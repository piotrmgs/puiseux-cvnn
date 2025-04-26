# Copyright (c) 2025 Piotr Migus
# This code is licensed under the MIT License.
# See the LICENSE file in the repository root for full license information.
############################################################
# Experiment with a toy problem in C^2 (4-dimensional real space):
#
# 1) We load uncertain points from a CSV file (`uncertain_synthetic.csv`)
#    and a pretrained neural network model (`model_parameters.pt`).
#
# 2) For selected points, we build a local polynomial approximation `F`
#    (difference of logit outputs) as a polynomial of degree ≥ 4
#    in a 4D space: (Re(z1), Im(z1), Re(z2), Im(z2)).
#
# 3) We apply the `compute_puiseux` function (from puis.py) to obtain
#    local Puiseux expansions around these selected points.
#
# Notes:
# - By default, we use degree=4 and n_samples=200, adjustable if needed.
# - `F` is evaluated locally around `xstar` through random perturbations
#   within the cube [-delta, delta]^4 in the real domain, and mapped to complex.
############################################################


import os
import csv
import torch
import time
import numpy as np
import sympy
import matplotlib.pyplot as plt
import scipy.linalg as la
import random
from sympy import factor

# For reproducibility: we fix random seeds in Python and NumPy
random.seed(42)
np.random.seed(42)

# Importing our custom modules:
#  - SimpleComplexNet (the complex-valued neural model),
#  - complex_modulus_to_logits (function computing logit moduli),
#  - puiseux_expansions (function performing local Puiseux expansions).
from src.find_up_synthetic import SimpleComplexNet, complex_modulus_to_logits
from src.puiseux import puiseux_expansions

def benchmark_local_poly_approx_and_puiseux(model, 
                                            xstar, 
                                            local_poly_func, 
                                            puiseux_func, 
                                            delta=0.01, 
                                            degree=4, 
                                            n_samples=200,
                                            device='cpu',
                                            do_factor=True,
                                            do_simplify=True,
                                            puiseux_prec=4):
    """
    Measures execution times of the following steps:
      1) Sample generation and computation of f(x) = alpha_0(x) - alpha_1(x).
      2) Polynomial fitting (least squares).
      3) Optional sympy.factor and sympy.simplify.
      4) Computation of Puiseux expansions.

    Parameters:
    ----------
    model : torch.nn.Module
        A trained model (e.g., SimpleComplexNet).
    xstar : ndarray (4,)
        A base point in R^4 = C^2.
    local_poly_func : callable
        A function of the form (model, xstar, delta, degree, n_samples, device) -> expr
        (or similar),
        which performs polynomial fitting in a neighborhood of xstar.
    puiseux_func : callable
        A function that takes an expression and returns a list of Puiseux expansions.
    delta, degree, n_samples, device : as above.
    do_factor : bool
        Whether to apply sympy.factor to the polynomial.
    do_simplify : bool
        Whether to apply sympy.simplify.
    remove_linear : bool
        Whether to remove linear terms from the polynomial (if implemented).
    puiseux_prec : int
        Precision used for Puiseux expansions in puiseux_func.

    Returns:
    --------
    times : dict
        Dictionary with keys: 
        'time_sampling', 'time_lstsq', 'time_factor', 'time_simplify', 'time_puiseux', 'time_total'
    expr : sympy.Expr
        The final polynomial (after optional factor and simplify).
    expansions : list 
        List of Puiseux expansions (from puiseux_func).
    """

    x, y = sympy.symbols('x y')
    I = sympy.I

    # Measure total execution time
    t0 = time.time()

    # 1) Sampling and function evaluation
    tA0 = time.time()
    # Sampling can be done externally or within `local_poly_func`. 
    # Here, we assume sampling is handled by `local_poly_func`.
    tA1 = time.time()
    time_sampling = tA1 - tA0

    # 2) Polynomial fitting (least squares)
    tB0 = time.time()
    expr = local_poly_func(model=model,
                           xstar=xstar,
                           delta=delta,
                           degree=degree,
                           n_samples=n_samples,
                           device=device)
    tB1 = time.time()
    time_lstsq = tB1 - tB0

    # 3) Factor and simplify polynomial (optional)
    time_factor = 0.0
    time_simplify = 0.0
    if do_factor:
        tf0 = time.time()
        expr = sympy.factor(expr)
        tf1 = time.time()
        time_factor = tf1 - tf0
    
    if do_simplify:
        ts0 = time.time()
        expr = sympy.simplify(expr)
        ts1 = time.time()
        time_simplify = ts1 - ts0

    # 4) Puiseux expansions computation
    tC0 = time.time()
    expansions = puiseux_expansions(expr,x, y, puiseux_prec)
    tC1 = time.time()
    time_puiseux = tC1 - tC0

    time_total = time.time() - t0

    times = {
        'time_sampling': time_sampling,
        'time_lstsq': time_lstsq,
        'time_factor': time_factor,
        'time_simplify': time_simplify,
        'time_puiseux': time_puiseux,
        'time_total': time_total
    }
    return times, expr, expansions

def evaluate_poly_approx_quality(model,
                                 poly_expr,
                                 xstar,
                                 delta=0.01,
                                 n_samples=500,
                                 device='cpu'):
    """
    Evaluate the quality of polynomial approximation (`poly_expr`) against
    the true function f(x) = alpha_0(x) - alpha_1(x) locally around `xstar`.

    Parameters:
    ----------
    model : nn.Module
        Trained neural model (e.g., SimpleComplexNet).
    poly_expr : sympy.Expr
        Local polynomial in variables (x, y), where:
        - x corresponds to z1 - z1*
        - y corresponds to z2 - z2*
        and (z1 = x1 + i*x3, z2 = x2 + i*x4).
        Assumption: poly_expr(0,0)=0 (already aligned).
    xstar : np.ndarray, shape (4,)
        Base point coordinates in (Re(z1), Re(z2), Im(z1), Im(z2)).
    delta : float
        Perturbation range for each of the 4 coordinates.
    n_samples : int
        Number of samples used to estimate approximation quality.
    device : str or torch.device
        'cpu' or 'cuda'.

    Returns:
    -------
    metrics : dict
        Dictionary containing keys:
        'RMSE', 'MAE', 'corr_pearson', 'sign_agreement'
    """

    x_sym, y_sym = sympy.symbols('x y')

    # Zapisujemy z1*, z2* w formie zespolonej
    z1_star = xstar[0] + 1j*xstar[2]
    z2_star = xstar[1] + 1j*xstar[3]

    # Losowo generujemy punkty w otoczeniu
    all_shifts = (2 * delta) * np.random.rand(n_samples, 4) - delta
    # Będziemy zapisywać wyniki w listach
    fvals_list = []
    pvals_list = []

    for shift in all_shifts:
        x_loc = xstar + shift
        # 1) Oblicz f(x_loc):
        x_ten = torch.tensor(x_loc, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            out = model(x_ten)  # shape (1,4)
            half = out.shape[1] // 2
            xr = out[:, :half]
            xi = out[:, half:]
            alpha = torch.sqrt(xr**2 + xi**2 + 1e-9)
            f_val = (alpha[:, 0] - alpha[:, 1]).item()
        fvals_list.append(f_val)

        # 2) Oblicz p(x_loc):
        #    Zamieniamy x_loc na z1, z2 -> z1 = x1 + i*x3, z2 = x2 + i*x4
        z1 = (x_loc[0] + 1j*x_loc[2]) - z1_star  # "local shift"
        z2 = (x_loc[1] + 1j*x_loc[3]) - z2_star
        # Substitujemy do expr
        p_val = poly_expr.subs({x_sym: z1, y_sym: z2})
        # Możliwe, że p_val może być zespolone, bierzemy np. część rzeczywistą:
        # ale jeśli ten wielomian jest realny, to i p_val powinno być (prawie) real.
        # Dla bezpieczeństwa:
        p_val = complex(p_val)
        #p_val_float=float(p_val)
        pvals_list.append(p_val.real)  # realna część

    # Konwertujemy na tablice numpy
    fvals_arr = np.array(fvals_list)
    pvals_arr = np.array(pvals_list)

    # Policzmy podstawowe metryki:
    # RMSE
    rmse = np.sqrt(np.mean((pvals_arr - fvals_arr) ** 2))
    # MAE
    mae = np.mean(np.abs(pvals_arr - fvals_arr))
    # Korelacja Pearson
    #   Tylko gdy wariancja niezerowa
    corr = np.corrcoef(fvals_arr, pvals_arr)[0, 1] if np.std(pvals_arr)*np.std(fvals_arr) > 1e-12 else 0.0
    # Zgodność znaku
    sign_f = (fvals_arr > 0).astype(int)
    sign_p = (pvals_arr > 0).astype(int)
    sign_agreement = np.mean(sign_f == sign_p)

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'corr_pearson': corr,
        'sign_agreement': sign_agreement
    }
    return metrics


def polynomial_features_complex(z, degree=4, remove_linear=False):
    """
    Construct polynomial features for a complex-valued bivariate input.

    This function computes a list of monomials of the form z1^i * z2^j for an input complex pair z = (z1, z2),
    with non-negative integers i and j that satisfy i + j ≤ degree. Optionally, if the remove_linear flag is set,
    features corresponding to monomials with a total degree less than 2 (i.e., both constant and first-order terms)
    are omitted from the output.

    Parameters
    ----------
    z : tuple of complex numbers
        A tuple (z1, z2) containing two complex numbers for which the polynomial features are evaluated.
    degree : int, optional
        The maximum allowable total degree for the generated monomials (i + j ≤ degree). Default is 4.
    remove_linear : bool, optional
        When True, omits all monomials whose total degree is less than 2 (i.e., skips both the constant term and
        linear terms). Default is False.

    Returns
    -------
    list
        A list of complex values, each corresponding to a monomial z1^i * z2^j for all valid exponent pairs (i, j)
        that satisfy i + j ≤ degree, with the optional exclusion of features having i + j < 2 if remove_linear is True.
    """
    x, y = z
    feats = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            if remove_linear and (i + j < 2):
                continue
            feats.append(x**i * y**j)
    return feats


def fit_polynomial_complex(z_vals, Fvals, degree=4, remove_linear=False):
    """
    Fit a bivariate polynomial in complex variables via least squares optimization.

    This function constructs and fits a polynomial in two complex variables, z1 and z2,
    by solving a linear least-squares problem. It is assumed that the input coordinate
    pairs (z1, z2) have been translated such that the center of interest is at the origin (0,0).
    The goal is to approximate a function F(z1, z2) – which may be either real or complex –
    by a polynomial model.

    The polynomial is built from all monomials of the form z1^i * z2^j that satisfy i + j ≤ degree.
    Optionally, if the flag `remove_linear` is True, monomials with total degree less than 2 
    (i.e., constant and first-order terms) are omitted, thereby emphasizing higher-order contributions.

    The polynomial coefficients are determined by solving an overdetermined complex linear system
    via least-squares. The function assesses the conditioning of the design matrix and prints warnings
    if the matrix is ill-conditioned (condition number > 1e8) or if it is rank-deficient,
    which may signal potential overfitting or numerical instability.

    Parameters:
    -----------
    z_vals : list of tuple
        A list of tuples, where each tuple (z1, z2) consists of complex numbers representing the 
        shifted coordinates with respect to the origin.
    Fvals : list or array-like
        An array-like structure containing the function values corresponding to each coordinate pair.
        The values may be real or complex (e.g., representing differences between logit outputs).
    degree : int, optional
        The maximum total degree (i.e., i+j) for the polynomial terms to be included in the model 
        (default is 4).
    remove_linear : bool, optional
        If True, excludes all monomials with total degree less than 2 (i.e., omits the constant 
        and linear terms), thereby focusing the fit on higher-order interactions (default is False).

    Returns:
    --------
    coeffs : numpy.ndarray
        A one-dimensional numpy array of complex coefficients corresponding to the fitted polynomial.
        The ordering of the coefficients matches the order of the monomials generated during the feature
        matrix construction.
    expr_sym : sympy.Expr
        A sympy expression representing the factorized form of the fitted polynomial in the symbolic
        variables x and y, which correspond to z1 and z2, respectively.
    """
    # Build the feature matrix (complex):
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
    matA = np.array(matA, dtype=np.complex128)
    Fvals = np.array(Fvals, dtype=np.complex128)
    
    # Solve by complex least squares
    coeffs, residuals, rank, s = np.linalg.lstsq(matA, Fvals, rcond=None)
    
    # Compute the condition number for stability check
    condA = np.linalg.cond(matA)
    if condA > 1e8:
        print(f"[WARN] The condition number is large: {condA:.2e}. Fit may be unstable (degree={degree}).")

    # Check rank deficiency
    if rank < matA.shape[1]:
        print(f"[WARN] Rank-deficient system: rank={rank} < {matA.shape[1]} monomials. "
              f"Potential overfitting or nearly singular matrix.")

    

    # Build the sympy expression in x, y
    x, y = sympy.symbols('x y')
    expr = 0
    #idx = 0
    for idx, (i, j) in enumerate(monomials):
        expr += coeffs[idx] * x**i * y**j
    expr = sympy.simplify(expr)
    return coeffs, factor(expr)


def load_uncertain_points(csv_path):
    """
    Load a CSV file containing uncertain points. Each row is expected to have:
      - 'index'
      - 'X' : a list of 4 floats (Re(z1), Im(z1), Re(z2), Im(z2)) inside brackets
      - 'true_label'
      - 'p1', 'p2' : probabilities or any other confidence measure.

    Example of a row's 'X': "[ 1.2, 0.2, -0.3, 0.8 ]"

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file with uncertain points.

    Returns:
    --------
    up_list : list of dict
        Each dict has keys:
          'index' : int,
          'X' : [re(z1), im(z1), re(z2), im(z2)],
          'true_label' : int,
          'p1' : float,
          'p2' : float
    """
    up_list = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            X_str = row['X'].strip('[]')
            X_vals = [float(x) for x in X_str.split(',')]
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


def local_poly_approx_complex(model, xstar, delta=0.01, degree=4, n_samples=2000, device='cpu', remove_linear=True):
    """
    Construct a local complex polynomial approximation of F = (logit1 - logit2) around a given 4D point.
    
    The base point xstar is expressed as [Re(z1), Re(z2), Im(z1), Im(z2)], so that the complex variables are:
         z1 = xstar[0] + i*xstar[2]    and    z2 = xstar[1] + i*xstar[3].
         
    The approximation is obtained by:
      1) Generating n_samples random perturbations in [-delta, delta]^4.
      2) Converting each perturbed 4D point to complex coordinates.
      3) Evaluating F via the provided model.
      4) Fitting a polynomial of total degree ≤ degree in (z1, z2) using least squares (optionally omitting constant 
         and linear terms if remove_linear is True).
      5) Adjusting the fitted polynomial so that it vanishes at the base point (P(0,0)=0 when shift_to_zero is True).
    
    Parameters:
    -----------
    model : torch.nn.Module
        A trained complex-valued neural network (e.g., SimpleComplexNet).
    xstar : numpy.ndarray, shape (4,)
        Base point specified as [Re(z1), Re(z2), Im(z1), Im(z2)].
    delta : float
        Half-width for uniform random perturbations along each dimension.
    degree : int
        Maximum total degree for polynomial terms.
    n_samples : int
        Number of samples generated for the least-squares fit.
    device : str or torch.device
        Device for PyTorch computations (e.g., 'cpu' or 'cuda').
    remove_linear : bool, optional
        If True, omits constant and linear terms from the polynomial fit (default True).
    shift_to_zero : bool, optional
        If True, shifts complex samples so that the base point maps to (0,0) and forces P(0,0)=0 (default True).
    
    Returns:
    --------
    expr : sympy.Expr
        A sympy expression in variables x and y representing the normalized local polynomial approximation of F.
    """
    x_star = xstar[0] + 1j * xstar[2]
    y_star = xstar[1] + 1j * xstar[3]
    
    z_samples = []
    Fvals = []
    for _ in range(n_samples):
        shift = (2 * delta) * np.random.rand(4) - delta
        x_loc = xstar + shift
        
        # Convert real data to complex z1, z2
        x = x_loc[0] + 1j * x_loc[2]
        y = x_loc[1] + 1j * x_loc[3]
        z_samples.append((x - x_star, y - y_star))

        # Evaluate F at x_loc using the model
        XY = torch.tensor(x_loc, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(XY)  # shape (1, 2*out_features)
            half = out.shape[1] // 2
            xr = out[:, :half]
            xi = out[:, half:]
            alpha = torch.sqrt(xr**2 + xi**2 + 1e-9)
            F_val = alpha[:, 0] - alpha[:, 1]
            F_val = F_val.item()
        Fvals.append(F_val)
    
    # Fit polynomial in complex variables (z1,z2)
    coeffs, expr = fit_polynomial_complex(z_samples, Fvals, degree=degree, remove_linear=remove_linear)
    
    # Enforce P(0,0)=0 by subtracting constant offset
    x_sym, y_sym = sympy.symbols('x y')
    
    expr = sympy.simplify(expr - expr.subs({x_sym: 0, y_sym: 0}))

    return expr


def puiseux_uncertain_point(F_hat_expr,prec=4, base_point=None):
    """
    Compute Newton-Puiseux expansions for a given local polynomial expression
    F_hat_expr in two variables (x, y). Save the expansions to a text file
    and return them as a list of strings.

    Parameters:
    -----------
    F_hat_expr : sympy.Expr
        The local polynomial approximation in x, y (complex variables).
    prec : int
        Precision (number of terms) for the Puiseux expansions
        in the function compute_puiseux.
    base_point : array-like, optional
        If provided, of shape (4,). This might be used to interpret expansions
        in terms of the original coordinates (z1*, z2*). Not strictly required.

    Returns:
    --------
    shifted_expansions : list of str
        A list of textual expansions from compute_puiseux,
        optionally adjusted if base_point is given.
    """
    x, y = sympy.symbols('x y')
    I = sympy.I
    
    expansions = puiseux_expansions(F_hat_expr,x,y,prec)
    if base_point is not None:
        x_sym = sympy.Symbol('x', complex=True)
        x_star = base_point[0] + 1j * base_point[2]
        y_star = base_point[1] + 1j * base_point[3]
        # For clarity, we do a naive substitution approach
        # but the usage depends on how expansions interpret 'x'
        shifted_expansions = [
            str(exp.subs(x_sym, x_sym - x_star) + y_star)
            for exp in expansions
        ]
    else:
        shifted_expansions = [str(exp) for exp in expansions]
    return shifted_expansions

