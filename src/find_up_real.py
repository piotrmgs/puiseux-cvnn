# Copyright (c) 2025 Piotr Migus
# This code is licensed under the MIT License.
# See the LICENSE file in the repository root for full license information.

"""
This module provides utility functions for preprocessing, modeling, training, and evaluating 
complex-valued neural networks, particularly applied to ECG classification using the MIT-BIH 
Arrhythmia Database.

Main functionalities include:
- Signal preprocessing and complex-valued feature extraction
- Dataset preparation and handling class imbalance
- Construction and training of complex-valued neural networks (e.g., SimpleComplexNet)
- Calibration using temperature scaling
- Evaluation with metrics, visualizations, and uncertainty analysis
- Ablation study support and plotting tools

This module is designed to support experiments involving complex-valued deep learning models
with a focus on interpretability, calibration, and robust evaluation.

Dependencies:
- PyTorch
- NumPy, SciPy, Scikit-learn
- Seaborn, Matplotlib
- MIT-BIH-specific preprocessing tools from `mit_bih_pre`
"""

import os
import logging
import copy
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.calibration import calibration_curve
import seaborn as sns           
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
import wfdb
from mit_bih_pre.pre_pro import load_mitbih_data

# Import your complex network components
from src.find_up_synthetic import SimpleComplexNet, complex_modulus_to_logits, find_uncertain_points, modrelu


# Configure module-level logger
logger = logging.getLogger(__name__)

import random  # NEW

# --- Reproducibility helpers (global) ---
DEFAULT_SEED = 12345

def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy, and PyTorch (CPU/CUDA) for deterministic behavior."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN settings for determinism (may be slightly slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        # Enforce deterministic cuBLAS matmul on CUDA
        # Must be set before the first CUDA matmul call; safe to set here at startup.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # Disable TF32 to avoid precision-related nondeterminism across GPUs
        if hasattr(torch.backends, "cuda") and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        torch.use_deterministic_algorithms(True)
    except Exception:
        pass  # not available on very old torch versions

def _make_worker_init_fn(base_seed: int):
    """Create a worker_init_fn that deterministically seeds each DataLoader worker."""
    def _seed_worker(worker_id: int):
        s = (base_seed + worker_id) % (2**32 - 1)
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
    return _seed_worker

def _make_generator(seed: int) -> torch.Generator:
    """Create a torch.Generator seeded for deterministic sampling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def tune_temperature(model: nn.Module, val_loader, device: str = "cpu") -> torch.Tensor:
    """
    Fit a single temperature parameter T >= 1 by minimizing cross-entropy on VALIDATION.
    We parameterize T = 1 + softplus(alpha), which guarantees T >= 1.
    Returns a 1-element CPU tensor with the optimal T.
    """
    model.eval()
    logits_list, labels_list = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = complex_modulus_to_logits(model(xb))
            logits_list.append(logits)
            labels_list.append(yb)

    logits = torch.cat(logits_list)   # [N, C]
    labels = torch.cat(labels_list)   # [N]

    # T = 1 + softplus(alpha)  => T>=1
    alpha = nn.Parameter(torch.zeros(1, device=logits.device))
    optimizer = optim.LBFGS([alpha], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        T = 1.0 + torch.nn.functional.softplus(alpha)
        loss = criterion(logits / T, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    T_opt = 1.0 + torch.nn.functional.softplus(alpha)
    return T_opt.detach().cpu()



def fit_isotonic_on_val(model: nn.Module, val_loader, device: str = "cpu"):
    """
    Fit an isotonic regression calibrator on VALIDATION for binary problems.
    Returns a callable f(p) that maps raw p1 -> calibrated p1, and a flag 'is_binary'.
    For multi-class this function returns (None, False).
    """
    from sklearn.isotonic import IsotonicRegression

    model.eval()
    probs1, labels = [], []
    C = None
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = complex_modulus_to_logits(model(xb))
            C = logits.size(1)
            if C != 2:
                return (None, False)
            p = torch.softmax(logits, dim=1)[:, 1]  # class-1 probability
            probs1.extend(p.cpu().numpy().tolist())
            labels.extend(yb.numpy().tolist())

    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(np.asarray(probs1, dtype=float), np.asarray(labels, dtype=int))
    def calibrate_batch(p_numpy: np.ndarray) -> np.ndarray:
        """Map Nx2 prob array -> calibrated Nx2 by isotonic on class 1."""
        p1 = p_numpy[:, 1]
        p1c = ir.predict(p1)
        p0c = 1.0 - p1c
        return np.stack([p0c, p1c], axis=1)
    return (calibrate_batch, True)

def fit_platt_on_val(model: nn.Module, val_loader, device: str = "cpu"):
    """
    Platt scaling for **binary** classification.

    We train a single-feature logistic regression on the score:
        s = logit_1 - logit_0
    where 'logit_k' are *pre-softmax* scores for class k.
    This avoids taking logs of probabilities (hence, no log(0/1) issues).

    Returns
    -------
    (apply_fn, True) on success, where:
        apply_fn(logits_np) -> calibrated probs (N, 2)
    or (None, False) if not applicable (non-binary or degenerate validation set).
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    model.eval()
    s_list, y_list = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = complex_modulus_to_logits(model(xb)).cpu().numpy()  # expected shape (B, 2)
            if logits.shape[1] != 2:
                return (None, False)
            s = logits[:, 1] - logits[:, 0]  # single scalar per sample
            s_list.append(s)
            y_list.append(yb.cpu().numpy())

    s_all = np.concatenate(s_list, axis=0).reshape(-1, 1)
    y_all = np.concatenate(y_list, axis=0).astype(int)

    if len(np.unique(y_all)) < 2 or s_all.shape[0] < 10:
        return (None, False)

    lr = LogisticRegression(solver="lbfgs", penalty="l2", C=1.0, max_iter=1000)
    lr.fit(s_all, y_all)

    def apply_platt(logits_np, **_):
        """
        Apply Platt scaling to a batch of *logits*.
        Parameters
        ----------
        logits_np : np.ndarray, shape (N, 2)
            Uncalibrated logits.

        Returns
        -------
        np.ndarray, shape (N, 2)
            Calibrated probabilities.
        """
        s = (logits_np[:, 1] - logits_np[:, 0]).reshape(-1, 1)
        p1 = lr.predict_proba(s)[:, 1]
        p1 = np.clip(p1, 1e-6, 1.0 - 1e-6)
        return np.stack([1.0 - p1, p1], axis=1)

    return (apply_platt, True)



def fit_beta_on_val(model: nn.Module, val_loader, device: str = "cpu", eps: float = 1e-6):
    """
    Beta calibration (Kull et al., 2019) for **binary** classification.

    We fit a logistic regression on the transformed features:
        X = [log(p), log(1-p)]
    where p = P(y=1 | x) are *uncalibrated* probabilities from the model.

    Numerical safety:
      - We clip p to [eps, 1-eps] before taking logs to avoid ±inf.
      - We guard against degenerate validation splits (single-class).

    Returns
    -------
    (apply_fn, True) on success, where:
        apply_fn(probs_np=...) -> calibrated probs of shape (N, 2)
    or (None, False) if not applicable (non-binary or degenerate validation set).
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    model.eval()
    p1_list, y_list = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = complex_modulus_to_logits(model(xb))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            C = probs.shape[1]
            if C != 2:
                return (None, False)  # Beta calibration is defined here for binary only
            p1_list.append(probs[:, 1])
            y_list.append(yb.cpu().numpy())

    p = np.concatenate(p1_list, axis=0).astype(np.float64)  # (N,)
    y = np.concatenate(y_list, axis=0).astype(int)

    # Both classes must be present; also require a minimal sample count
    if len(np.unique(y)) < 2 or p.size < 10:
        return (None, False)

    # Clip to avoid log(0) / log(1)
    p = np.clip(p, eps, 1.0 - eps)

    # Features for LR: [log p, log(1-p)]
    X = np.column_stack([np.log(p), np.log(1.0 - p)])
    assert np.isfinite(X).all(), "Non-finite encountered after clipping — unexpected."

    lr = LogisticRegression(
        solver="lbfgs", penalty="l2", C=1.0, max_iter=2000
    )
    lr.fit(X, y)

    def apply_beta(*, probs_np=None, **_):
        """
        Apply beta calibration to a batch of *probabilities*.
        Parameters
        ----------
        probs_np : np.ndarray, shape (N, 2)
            Uncalibrated probabilities for the binary problem.

        Returns
        -------
        np.ndarray, shape (N, 2)
            Calibrated probabilities.
        """
        assert probs_np is not None, "Beta calibration expects probs_np"
        p1 = np.asarray(probs_np, dtype=np.float64)[:, 1]
        p1 = np.clip(p1, eps, 1.0 - eps)
        Xte = np.column_stack([np.log(p1), np.log(1.0 - p1)])
        out = lr.predict_proba(Xte)[:, 1]
        out = np.clip(out, eps, 1.0 - eps)
        return np.stack([1.0 - out, out], axis=1)

    return (apply_beta, True)



def fit_vector_scaling_on_val(model: nn.Module, val_loader, device: str = "cpu", max_iter: int = 400):
    """
    Vector Scaling (Guo et al., 2017) for multi-class (also works for binary).

    We learn per-class scale 's ∈ R^C' and bias 'b ∈ R^C' such that:
        z' = diag(s) * z + b,
    and minimize cross-entropy on VALIDATION. Probabilities are obtained
    by softmax(z') at application time.

    Returns
    -------
    (apply_fn, True) on success, where:
        apply_fn(logits_np) -> calibrated probs (N, C)
    or (None, False) if not applicable (degenerate VAL etc.).
    """
    model.eval()
    Z_list, y_list = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            z = complex_modulus_to_logits(model(xb)).cpu()  # (B, C)
            Z_list.append(z)
            y_list.append(yb.cpu())

    Z = torch.cat(Z_list, dim=0)           # (N, C)
    y = torch.cat(y_list, dim=0).long()    # (N,)

    C = Z.shape[1]
    if C < 2 or len(torch.unique(y)) < 2 or Z.shape[0] < 10:
        return (None, False)

    # Parameters to learn: per-class scale and bias
    s = torch.ones(C, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(C, dtype=torch.float64, requires_grad=True)

    Z64 = Z.to(dtype=torch.float64)
    y64 = y.to(dtype=torch.long)

    opt = torch.optim.LBFGS([s, b], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        zprime = Z64 * s.unsqueeze(0) + b.unsqueeze(0)   # (N, C)
        loss = nll(zprime, y64)
        loss.backward()
        return loss

    opt.step(closure)

    s_f = s.detach().cpu().numpy()
    b_f = b.detach().cpu().numpy()

    def apply_vs(logits_np, **_):
        """
        Apply vector scaling to a batch of *logits*.
        Parameters
        ----------
        logits_np : np.ndarray, shape (N, C)
            Uncalibrated logits.

        Returns
        -------
        np.ndarray, shape (N, C)
            Calibrated probabilities after softmax(diag(s)*z + b).
        """
        z = np.asarray(logits_np, dtype=np.float64)
        zprime = z * s_f.reshape(1, -1) + b_f.reshape(1, -1)
        # Stable softmax
        zprime = zprime - zprime.max(axis=1, keepdims=True)
        expz = np.exp(zprime)
        probs = expz / expz.sum(axis=1, keepdims=True)
        probs = np.clip(probs, 1e-6, 1.0)
        return probs

    tag = f"s={np.round(s_f,3)}, b={np.round(b_f,3)}"
    return (apply_vs, True)


def get_calibrator(model: nn.Module, val_loader, method: str, device: str = "cpu"):
    """
    Calibration dispatcher.
    Returns
    -------
    (apply_fn, tag)
      - apply_fn: a callable to be used as:
          * for logit-based methods:   apply_fn(logits_np=...)
          * for prob-based methods:    apply_fn(probs_np=...)
      - tag: a short descriptor of the fitted parameters (e.g., 'T=1.23', 'isotonic', etc.)

    If a method is not applicable (e.g., isotonic/beta on non-binary), returns (None, False).
    """
    import numpy as np

    m = (method or "").strip().lower()

    # Identity: pass-through (compute softmax if logits are provided)
    if m in ("none", ""):
        def ident(**kwargs):
            if "logits_np" in kwargs and kwargs["logits_np"] is not None:
                z = kwargs["logits_np"]
                z = z - z.max(axis=1, keepdims=True)  # stable softmax
                ez = np.exp(z)
                return ez / ez.sum(axis=1, keepdims=True)
            elif "probs_np" in kwargs and kwargs["probs_np"] is not None:
                return kwargs["probs_np"]
            raise ValueError("Identity calibrator needs logits_np or probs_np")
        return (ident, "identity")

    # Temperature scaling (works for any C>=2)
    if m == "temperature":
        T = tune_temperature(model, val_loader, device=device).item()
        T = max(T, 1e-6)  # safety clamp
        def apply_T(logits_np, **_):
            z = logits_np / T
            z = z - z.max(axis=1, keepdims=True)
            ez = np.exp(z)
            return ez / ez.sum(axis=1, keepdims=True)
        return (apply_T, f"T={T:.3f}")

    # Isotonic regression (binary only; prob-space)
    if m == "isotonic":
        iso, ok = fit_isotonic_on_val(model, val_loader, device=device)
        if not ok:
            return (None, False)
        def apply_iso(*, probs_np=None, **_):
            assert probs_np is not None
            return iso(probs_np)
        return (apply_iso, "isotonic")

    # Platt scaling (binary; logit-space)
    if m == "platt":
        pl, ok = fit_platt_on_val(model, val_loader, device=device)
        return (pl, "platt") if ok else (None, False)

    # Beta calibration (binary; prob-space)
    if m == "beta":
        be, ok = fit_beta_on_val(model, val_loader, device=device)
        return (be, "beta") if ok else (None, False)

    # Vector scaling (multi-class; logit-space)
    if m == "vector":
        vs, ok = fit_vector_scaling_on_val(model, val_loader, device=device)
        return (vs, "vector") if ok else (None, False)

    # Unknown method → identity fallback
    def ident2(**kwargs):
        if "logits_np" in kwargs and kwargs["logits_np"] is not None:
            z = kwargs["logits_np"]
            z = z - z.max(axis=1, keepdims=True)
            ez = np.exp(z)
            return ez / ez.sum(axis=1, keepdims=True)
        elif "probs_np" in kwargs and kwargs["probs_np"] is not None:
            return kwargs["probs_np"]
        raise ValueError("Identity calibrator needs logits_np or probs_np")
    return (ident2, f"unknown:{m}")



def compress_to_C2(X: np.ndarray, window_size=128):
    """
    Compress a multichannel complex-valued signal into a compact 4-dimensional real feature vector.

    Each input sample is assumed to contain concatenated real and imaginary parts of a complex-valued 
    time signal across multiple channels. This function reconstructs the complex analytic signal 
    per channel, and extracts simple statistical descriptors:
        - Mean (mu) of the full signal (across all channels)
        - Mean slope (first difference) of the signal
    
    These are decomposed into real and imaginary parts to produce a final feature vector in ℝ⁴.

    Parameters
    ----------
    X : np.ndarray of shape (N, 2 * window_size * nchannels)
        Input matrix where each row is a sample containing real and imaginary parts concatenated.
    
    window_size : int, optional (default=128)
        Number of time steps per channel per signal (i.e., length of real/imag part per channel).

    Returns
    -------
    np.ndarray of shape (N, 4)
        Compressed feature matrix. Each row contains:
            [Re(mean), Re(slope), Im(mean), Im(slope)]
    """
    N, F = X.shape
    nch = F // (2 * window_size)
    C2 = np.zeros((N, 4), dtype=np.float32)

    for i in range(N):
        x = X[i]
        zs = []
        for ch in range(nch):
            off = ch * 2 * window_size
            real = x[off : off + window_size]
            imag = x[off + window_size : off + 2 * window_size]
            z = real + 1j * imag
            zs.append(z)
        z_all = np.hstack(zs)
        mu = z_all.mean()
        slope = np.mean(z_all[1:] - z_all[:-1])

        C2[i, 0] = mu.real
        C2[i, 1] = slope.real
        C2[i, 2] = mu.imag
        C2[i, 3] = slope.imag

    return C2



def prepare_complex_input(X: np.ndarray, method: str = 'split_pca', n_real: int = 2) -> np.ndarray:
    """
    Transform real-valued input features into a complex-valued representation for neural networks.

    Depending on the method chosen, this function reduces the dimensionality of the input and 
    outputs features arranged as pairs: [Re1, ..., ReN, Im1, ..., ImN]. It prepares data for 
    complex-valued neural networks where each feature is considered as a complex number.

    Supported methods:
      - 'pca':        Apply PCA to all features, assign zero imaginary parts.
      - 'split_pca':  Split input into real and imaginary halves, apply PCA separately.
      - 'identity':   Return input unchanged if it already has shape (N, 2 * n_real).
      - 'complex_stats': Extract 4D complex summary stats via `compress_to_C2`.

    Parameters
    ----------
    X : np.ndarray of shape (N, D)
        Real-valued input matrix.
    
    method : str, optional (default='split_pca')
        Dimensionality reduction and transformation method.

    n_real : int, optional (default=2)
        Number of real (and imaginary) components in the output. Total output dimension will be 2 * n_real.

    Returns
    -------
    X_complex : np.ndarray of shape (N, 2 * n_real)
        Transformed feature matrix ready for complex-valued neural networks.

    Raises
    ------
    ValueError
        If method is 'identity' and feature dimension does not match 2 * n_real.
        If an unknown method name is provided.
    """
    total_feats = X.shape[1]

    if method == 'pca':
        pca = PCA(n_components=n_real)
        X_r = pca.fit_transform(X)
        X_i = np.zeros_like(X_r)
        logger.info("PCA retained %d components (real part, imag zeros)", n_real)
        X_complex = np.hstack([X_r, X_i])

    elif method == 'split_pca':
        half = total_feats // 2
        X_r_full = X[:, :half]
        X_i_full = X[:, half:]
        pca_r = PCA(n_components=n_real)
        X_r = pca_r.fit_transform(X_r_full)
        pca_i = PCA(n_components=n_real)
        X_i = pca_i.fit_transform(X_i_full)
        logger.info("Split PCA retained %d components each for real and imag", n_real)
        X_complex = np.hstack([X_r, X_i])

    elif method == 'identity':
        expected = 2 * n_real
        if total_feats != expected:
            raise ValueError(f"For identity method, input features={total_feats} must equal 2*n_real={expected}")
        X_complex = X.copy()
        logger.info("Identity mapping used, features unchanged")

    elif method == 'complex_stats':
        X_complex = compress_to_C2(X, window_size=WINDOW_SIZE)
        logger.info("Compressed to C^2 using complex_stats")
        return X_complex
    
    elif method == 'stft_stats':
        # Dataset-agnostic STFT statistics from complex IQ/analytic signal.
        # X has shape (N, 2*L) where first half are real samples and second half imag samples.
        feats = []
        eps = 1e-12
        L = X.shape[1] // 2  # infer window length automatically

        for x in X:
            # Rebuild complex signal z(t) = I + jQ
            i_part = x[:L].astype(np.float32)
            q_part = x[L:].astype(np.float32)
            sig = i_part + 1j * q_part

            # Symmetric spectrum (±f), no boundary padding to avoid leakage artifacts
            f, t, S = scipy.signal.stft(
                sig,
                fs=1.0,
                nperseg=min(64, L),
                noverlap=max(1, min(64, L) // 2),  # explicit half-overlap for stability
                return_onesided=False,
                boundary=None
            )

            A = np.abs(S)  # (F, T)

            # 1) Mean and std of magnitude
            mu = float(A.mean())
            sigma = float(A.std())

            # 2) Frequency centroid (normalize frequencies to [-1,1] so it's dataset-agnostic)
            f_norm = f / (np.max(np.abs(f)) + eps)            # in [-1, 1]
            spec_mean = A.mean(axis=1)                        # average over time
            denom = spec_mean.sum() + eps
            f_centroid = float((f_norm * spec_mean).sum() / denom)

            # 3) Spectral entropy (normalize over full F×T grid)
            P = A / (A.sum() + eps)
            entropy = float(-(P * np.log2(P + eps)).sum())

            feats.append([mu, sigma, f_centroid, entropy])

        return np.asarray(feats, dtype=np.float32)

    else:
        raise ValueError(f"Unknown prepare_complex_input method '{method}'")

    return X_complex



def create_dataloaders(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       batch_size: int = 128, num_workers: int = 4,
                       pin_memory: bool = True,
                       seed: int | None = None):
    """
    Prepare PyTorch DataLoaders for training and testing with standardized input and class-balanced sampling.

    This function performs three main operations:
      1. Standardizes the input features using z-score normalization (mean=0, std=1).
      2. Converts NumPy arrays into PyTorch tensors.
      3. Builds DataLoaders with:
         - A `WeightedRandomSampler` for the training set to address class imbalance.
         - A deterministic sampler for the test set.

    Parameters
    ----------
    X_train : np.ndarray of shape (N_train, D)
        Training feature matrix.

    y_train : np.ndarray of shape (N_train,)
        Ground truth labels for training.

    X_test : np.ndarray of shape (N_test, D)
        Test feature matrix.

    y_test : np.ndarray of shape (N_test,)
        Ground truth labels for testing.

    batch_size : int, optional (default=128)
        Number of samples per batch.

    num_workers : int, optional (default=4)
        Number of subprocesses to use for data loading.

    pin_memory : bool, optional (default=True)
        Whether to use pinned memory for faster host-to-device transfer.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training set with balanced sampling.

    test_loader : torch.utils.data.DataLoader
        DataLoader for the test set (non-shuffled).
    """
    
    # Normalize features using training set statistics
    scaler = StandardScaler().fit(X_train)
    X_tr = scaler.transform(X_train)
    X_te = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    y_te_t = torch.tensor(y_test, dtype=torch.long)

    # Class-balanced sampling for TRAIN
    counts = np.bincount(y_train)
    inv = 1.0 / np.maximum(counts, 1)
    samp_weights = inv[y_train]
    gen = _make_generator(seed if seed is not None else DEFAULT_SEED)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(samp_weights, dtype=torch.double),
        num_samples=len(samp_weights),
        generator=gen,
        replacement=True
    )
    logger.info("WeightedRandomSampler (train) — class counts %s", counts.tolist())

    worker_fn = _make_worker_init_fn((seed if seed is not None else DEFAULT_SEED))

    train_loader = DataLoader(
        dataset=TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_fn,
        generator=gen
    )

    test_loader = DataLoader(
        dataset=TensorDataset(X_te_t, y_te_t),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_fn,
        generator=gen
    )


    return train_loader, test_loader


def create_train_val_test_loaders(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int | None = None,
):
    """
    Build deterministic DataLoaders for train/val/test using a scaler fit ONLY on TRAIN.
    - Standardize features with train statistics.
    - Use a class-balanced WeightedRandomSampler for TRAIN.
    - Use deterministic worker seeding + a torch.Generator for reproducible sampling.
    """

    # ---------- 1) Fit scaler on TRAIN only, then transform VAL/TEST ----------
    scaler = StandardScaler().fit(X_train)
    X_tr = scaler.transform(X_train)
    X_va = scaler.transform(X_val)
    X_te = scaler.transform(X_test)

    # ---------- 2) Convert to tensors ----------
    X_tr_t, y_tr_t = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    X_va_t, y_va_t = torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_val,   dtype=torch.long)
    X_te_t, y_te_t = torch.tensor(X_te, dtype=torch.float32), torch.tensor(y_test,  dtype=torch.long)

    # ---------- 3) Deterministic sampler & workers ----------
    # Class-balanced sampling for TRAIN: inverse frequency weights
    counts = np.bincount(y_train)
    inv = 1.0 / np.maximum(counts, 1)              # avoid division by zero if a class is empty
    samp_weights = inv[y_train]                     # shape: (N_train,)

    seed_eff = seed if seed is not None else DEFAULT_SEED
    gen = _make_generator(seed_eff)                 # deterministic generator for sampling order
    worker_fn = _make_worker_init_fn(seed_eff)      # deterministic seeding per worker

    sampler = WeightedRandomSampler(
        weights=torch.tensor(samp_weights, dtype=torch.double),
        num_samples=len(samp_weights),              # one epoch ~= one pass over (weighted) train set
        generator=gen,
        replacement=True
    )
    logger.info("WeightedRandomSampler (train) — class counts %s | seed=%d",
                counts.tolist(), seed_eff)

    # ---------- 4) DataLoaders (deterministic) ----------
    train_loader = DataLoader(
        dataset=TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size,
        sampler=sampler,                            # no shuffle when sampler is used
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_fn,
        generator=gen
    )

    val_loader = DataLoader(
        dataset=TensorDataset(X_va_t, y_va_t),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_fn,
        generator=gen
    )

    test_loader = DataLoader(
        dataset=TensorDataset(X_te_t, y_te_t),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_fn,
        generator=gen
    )

    return train_loader, val_loader, test_loader, scaler




def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                epochs: int = 10,
                lr: float = 1e-3,
                device: str = 'cpu',
                early_stopping_patience: int = 5):
    """
    Train a complex-valued neural network with early stopping and learning rate scheduling.

    This function handles the training loop for a model with complex-valued outputs, which are 
    converted to real-valued logits using `complex_modulus_to_logits`. It tracks training loss 
    and validation accuracy, applies early stopping based on validation performance, and 
    returns both the training history and the best-performing model weights.

    Key features:
    - Adam optimizer with weight decay regularization
    - ReduceLROnPlateau scheduler (halves LR if validation accuracy plateaus)
    - Early stopping with configurable patience
    - Best model checkpointing based on validation accuracy

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train.

    train_loader : DataLoader
        DataLoader providing batches of training data.

    val_loader : DataLoader
        DataLoader providing batches of validation data.

    epochs : int, optional (default=10)
        Maximum number of training epochs.

    lr : float, optional (default=1e-3)
        Initial learning rate for the Adam optimizer.

    device : str, optional (default='cpu')
        Device on which training will be performed ('cpu' or 'cuda').

    early_stopping_patience : int, optional (default=5)
        Number of epochs with no improvement on validation accuracy before stopping.

    Returns
    -------
    history : dict
        Dictionary containing:
            - 'train_loss': List of training losses per epoch.
            - 'val_acc': List of validation accuracies per epoch.

    best_state : dict
        State dictionary (`state_dict`) of the best-performing model (highest validation accuracy).
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Reduce LR when validation accuracy stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     factor=0.5, patience=2)

    history = {'train_loss': [], 'val_acc': []}
    best_acc = 0.0
    best_state = None
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        # --- Training loop ---
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            logits = complex_modulus_to_logits(out)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * Xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        # --- Validation loop ---
        model.eval()
        correct = 0
        total = 0
        y_true, y_pred, y_score = [], [], []

        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                logits = complex_modulus_to_logits(out)
                probs = nn.Softmax(dim=1)(logits)
                preds = torch.argmax(probs, dim=1)

                correct += (preds == yb).sum().item()
                total += yb.size(0)

                y_true.extend(yb.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
                y_score.extend(probs[:, 1].cpu().tolist())

        val_acc = correct / total
        history['val_acc'].append(val_acc)
        scheduler.step(val_acc)

        logger.info("Epoch %d: train_loss=%.4f, val_acc=%.4f", epoch, train_loss, val_acc)

        # --- Early stopping logic ---
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())  # <- DEEP COPY
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    return history, best_state

def save_plots(history: dict, output_folder: str, fold: str | int):
    """
    Save a plot of the training loss and validation accuracy over epochs for a single cross-validation fold.

    This function visualizes the training process for one fold by plotting:
      - Training loss vs. epoch
      - Validation accuracy vs. epoch

    The resulting plot is saved as a PNG file in the specified output folder. 
    It helps assess convergence, potential overfitting, and the general learning dynamics.

    Parameters
    ----------
    history : dict
        A dictionary containing:
            - 'train_loss': List of training losses per epoch.
            - 'val_acc': List of validation accuracies per epoch.

    output_folder : str
        Directory path where the plot will be saved.

    fold : int
        The fold index (used for naming the output file).

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(history['train_loss'], marker='o', label='Train Loss')
    ax.plot(history['val_acc'], marker='o', label='Val Acc')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.set_title(f'Fold {fold} Training History')
    ax.legend()

    path = os.path.join(output_folder, f'fold_{fold}_history.png')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    logger.info("Saved training history to %s", path)



def save_confusion_roc(y_true: list, y_pred: list, y_score: list, output_folder: str, fold: int):
    """
    Generate and save a confusion matrix and ROC curve for a single cross-validation fold.

    This function evaluates the classifier's performance using:
      - A confusion matrix (based on predicted class labels)
      - A ROC curve (based on predicted probabilities for the positive class)

    Both visualizations are saved as PNG files in the specified folder. Additionally, the
    Area Under the ROC Curve (AUC) is computed and returned.

    Parameters
    ----------
    y_true : list of int
        Ground truth binary class labels (0 or 1) for the validation set.

    y_pred : list of int
        Predicted class labels (0 or 1) from the model.

    y_score : list of float
        Predicted probabilities for the positive class (usually output[:,1] from softmax).

    output_folder : str
        Path to the directory where the plots will be saved.

    fold : int
        Fold index (used for naming the output files).

    Returns
    -------
    roc_auc : float
        Area Under the Curve (AUC) for the ROC curve.
    """
    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    cm_path = os.path.join(output_folder, f'fold_{fold}_confusion.png')
    fig.savefig(cm_path)
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", cm_path)

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
    ax2.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend()

    fig2.tight_layout()
    roc_path = os.path.join(output_folder, f'fold_{fold}_roc.png')
    fig2.savefig(roc_path)
    plt.close(fig2)
    logger.info("Saved ROC curve to %s", roc_path)

    return roc_auc


def save_overall_history(histories, output_folder):
    """
    Generate and save an aggregated training history plot across multiple cross-validation folds.

    This function visualizes the mean and standard deviation of training loss and validation accuracy
    across all folds of cross-validation. It plots:
      - Mean ± std of training loss (left Y-axis)
      - Mean ± std of validation accuracy (right Y-axis)

    Useful for analyzing the consistency and stability of training across different data splits.

    Parameters
    ----------
    histories : list of dict
        A list of dictionaries (one per fold), where each dict contains:
            - 'train_loss': List[float] – training loss per epoch
            - 'val_acc': List[float] – validation accuracy per epoch

    output_folder : str
        Path to the directory where the plot will be saved.

    Returns
    -------
    None
    """
    # Determine maximum number of epochs (in case folds ran for different lengths)
    max_len = max(len(h['train_loss']) for h in histories)

    # Pad each fold's history to the same length using edge values (repeat last epoch's values)
    tl = np.array([
        np.pad(h['train_loss'], (0, max_len - len(h['train_loss'])), mode='edge')
        for h in histories
    ])
    vacc = np.array([
        np.pad(h['val_acc'], (0, max_len - len(h['val_acc'])), mode='edge')
        for h in histories
    ])

    epochs = np.arange(1, max_len + 1)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()

    # Plot training loss (mean ± std)
    ax1.plot(epochs, tl.mean(0), label='Train loss (mean)', marker='o')
    ax1.fill_between(epochs, tl.mean(0) - tl.std(0), tl.mean(0) + tl.std(0), alpha=0.2)
    ax1.set_ylabel('Loss')

    # Plot validation accuracy (mean ± std)
    ax2.plot(epochs, vacc.mean(0), label='Val acc (mean)', marker='s', color='tab:orange')
    ax2.fill_between(epochs, vacc.mean(0) - vacc.std(0), vacc.mean(0) + vacc.std(0), 
                     alpha=0.2, color='tab:orange')
    ax2.set_ylabel('Accuracy')

    ax1.set_xlabel('Epoch')
    fig.tight_layout()

    path = os.path.join(output_folder, 'overall_history.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)

    logger.info("Saved overall learning-curve to %s", path)


def save_calibration_curve(
    y_true,
    y_prob,
    output_folder,
    suffix: str = "",
    n_bins: int = 10,
    strategy: str = "quantile",     # "quantile" (default) or "uniform"
    draw_confidence: bool = True,   # draw 95% binomial CIs per bin
):
    """
    Save a reliability diagram (calibration curve). If `suffix` is provided, it's appended
    to the filename and title.

    Improvements over the basic version:
      - quantile binning by default (more stable curves when probs are skewed),
      - 95% binomial Wilson confidence intervals per bin (optional).
    """
    import numpy as np
    from math import sqrt
    import os
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    # --- Compute bin edges ---
    if strategy.lower() == "quantile":
        # Guard against duplicate edges when many equal probabilities
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(y_prob, quantiles)
        # Ensure strictly increasing edges to avoid empty/undefined bins
        edges = np.clip(edges, 0.0, 1.0)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = min(1.0, edges[i - 1] + 1e-12)
    elif strategy.lower() == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")

    # --- Aggregate per-bin stats ---
    prob_true, prob_pred, counts = [], [], []

    def _wilson_interval(p_hat: float, n: int, z: float = 1.96):
        """Two-sided 95% Wilson interval (numerically stable near 0 or 1)."""
        if n <= 0:
            return (np.nan, np.nan)
        denom = 1.0 + (z**2) / n
        center = (p_hat + (z**2) / (2 * n)) / denom
        halfw = (z / denom) * sqrt(max(p_hat * (1 - p_hat) / n + (z**2) / (4 * n**2), 0.0))
        return (max(0.0, center - halfw), min(1.0, center + halfw))

    ci_low, ci_high = [], []

    for b in range(n_bins):
        left, right = edges[b], edges[b + 1]
        # Include the right edge only for the last bin to cover prob==1.0
        if b < n_bins - 1:
            mask = (y_prob >= left) & (y_prob < right)
        else:
            mask = (y_prob >= left) & (y_prob <= right)

        n = int(mask.sum())
        if n == 0:
            continue

        p_bin = float(y_prob[mask].mean())
        t_bin = float(y_true[mask].mean())

        prob_pred.append(p_bin)
        prob_true.append(t_bin)
        counts.append(n)

        lo, hi = _wilson_interval(t_bin, n)
        ci_low.append(lo)
        ci_high.append(hi)

    prob_pred = np.asarray(prob_pred)
    prob_true = np.asarray(prob_true)
    counts    = np.asarray(counts)
    ci_low    = np.asarray(ci_low)
    ci_high   = np.asarray(ci_high)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")

    label = f"Model ({strategy}, {len(prob_true)} bins)"
    if draw_confidence:
        yerr = np.vstack([prob_true - ci_low, ci_high - prob_true])
        ax.errorbar(prob_pred, prob_true, yerr=yerr, fmt="o", capsize=3, label=label)
    else:
        ax.plot(prob_pred, prob_true, marker="o", label=label)

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    title = "Reliability diagram" + (f" {suffix}" if suffix else "")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()
    fname = "calibration_curve" + (f"_{suffix}" if suffix else "") + ".png"
    fig.savefig(os.path.join(output_folder, fname), dpi=300)
    plt.close(fig)

    # Log some quick diagnostics for the rebuttal appendix
    if counts.size:
        logger.info(
            "Saved calibration curve to %s | strategy=%s | bins=%d | median/bin count=%.1f",
            os.path.join(output_folder, fname),
            strategy,
            len(counts),
            float(np.median(counts)),
        )
    else:
        logger.info("Saved calibration curve to %s (no non-empty bins)", os.path.join(output_folder, fname))


def save_uncertainty_hist(y_prob, threshold, output_folder, label_override=None):
    """
    Plot a histogram of maximum softmax probabilities (p_max).
    If `threshold` is None or <= 0, do not shade a region; instead, add a legend
    entry clarifying that only the margin condition is active.
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(y_prob, bins=30, alpha=0.7)

    if threshold is not None and threshold > 0:
        ax.axvspan(0, threshold, color='red', alpha=0.15,
                   label=label_override or f'Uncertain < {threshold:.3f}')
        ax.legend()
    else:
        # Draw a thin symbolic line at 0 and use a text-only legend entry
        ax.axvline(0, color='red', alpha=0.10)
        ax.legend([label_override or "tau* not used — only 'margin' condition active"])

    ax.set_xlabel('Max softmax probability')
    ax.set_ylabel('Number of samples')
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, 'uncertainty_hist.png'), dpi=300)
    plt.close(fig)

    logger.info("Saved uncertainty histogram")


def save_complex_pca_scatter(X_complex, y, output_folder, sample_sz=5000):
    """
    Generate and save a 2D scatter plot of complex-valued features to visualize class separation.

    This function takes complex-valued input features arranged as [Re1, ..., ReN, Im1, ..., ImN]
    and plots the first real and first imaginary dimensions (e.g., PC1_real vs. PC1_imag). It’s
    particularly useful after dimensionality reduction techniques such as PCA applied separately
    to real and imaginary parts.

    The plot shows how well different classes are separated in the complex feature space.

    Parameters
    ----------
    X_complex : np.ndarray of shape (N, 2 * d)
        Complex-valued feature matrix (real and imaginary parts concatenated along axis 1).
        Expected format: [Re1, ..., Re_d, Im1, ..., Im_d]

    y : array-like of shape (N,)
        Class labels corresponding to each sample in `X_complex`.

    output_folder : str
        Path to the directory where the scatter plot will be saved.

    sample_sz : int, optional (default=5000)
        Maximum number of points to plot (for performance). If the dataset is larger,
        a random subset is used.

    Returns
    -------
    None
    """
    # Subsample if needed to avoid overcrowding the plot
    if len(X_complex) > sample_sz:
        idx = np.random.choice(len(X_complex), sample_sz, replace=False)
        X, lbl = X_complex[idx], y[idx]
    else:
        X, lbl = X_complex, y

    # First real and imaginary components
    real_pc1 = X[:, 0]
    imag_pc1 = X[:, X.shape[1] // 2]  # First imaginary feature (after real ones)

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.scatterplot(
        x=real_pc1,
        y=imag_pc1,
        hue=lbl,
        palette='Set2',
        s=10,
        ax=ax,
        legend=False
    )
    ax.set_xlabel('Real PC1')
    ax.set_ylabel('Imag PC1')
    ax.set_title('Complex feature space')

    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, 'complex_PCA_scatter.png'), dpi=300)
    plt.close(fig)

    logger.info("Saved complex PCA scatter")



def save_ablation_barplot(results_dict, output_folder):
    """
    Generate and save a bar plot to visualize results of an ablation study.

    This function creates a bar chart comparing model performance (e.g., F1-score)
    across different experimental variants, such as baseline, model modifications, or
    feature removals. Useful for highlighting the contribution of each component
    in a modular architecture or preprocessing pipeline.

    Parameters
    ----------
    results_dict : dict[str, float]
        Dictionary mapping variant labels (e.g., 'baseline', 'no_dropout') to their
        corresponding performance scores (e.g., F1-score).

    output_folder : str
        Path to the directory where the bar plot will be saved.

    Returns
    -------
    None
    """
    labels = list(results_dict.keys())
    scores = [results_dict[k] for k in labels]

    fig, ax = plt.subplots(figsize=(5, 3))
    sns.barplot(x=labels, y=scores, ax=ax)

    ax.set_ylabel('F1-score')
    ax.set_xlabel('Variant')
    ax.set_ylim(0, 1)  # assumes score is normalized between 0 and 1

    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, 'ablation_barplot.png'), dpi=300)
    plt.close(fig)

    logger.info("Saved ablation barplot")

def negative_log_likelihood(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Binary NLL = - mean( y*log p + (1-y)*log(1-p) )."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probs, dtype=float)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Binary Brier score = mean( (p - y)^2 )."""
    y = np.asarray(y_true).astype(float)
    p = np.asarray(probs, dtype=float)
    return float(((p - y) ** 2).mean())

def mean_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Return (mean, halfwidth) for a normal-approx 100*confidence% CI."""
    import math
    v = np.array(values, dtype=float)
    m = float(v.mean())
    s = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    hw = 1.96 * s / math.sqrt(max(len(v), 1))
    return m, hw


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Expected Calibration Error (ECE) in the 'reliability diagram' convention:
    For each probability bin, compare the empirical frequency of positives
    (mean of y_true) with the mean predicted probability in that bin.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary labels {0,1}.
    y_prob : array-like of shape (n_samples,)
        Predicted probability for class 1.
    n_bins : int
        Number of bins in [0,1].

    Returns
    -------
    float
        Weighted average of |freq - conf| over bins (weights = bin fraction).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if m.any():
            conf = float(y_prob[m].mean())   # mean predicted prob in bin
            freq = float(y_true[m].mean())   # empirical positive rate in bin
            weight = float(m.mean())         # fraction of samples in bin
            ece += abs(freq - conf) * weight

    return float(ece)



def select_thresholds_budget_count(y_true, probs, X, budget_count=10, safety_mult=4):
    """
    Pick (tau, delta) so that the number of flagged samples on VALIDATION
    is <= budget_count, while maximizing error capture.
    Candidates are taken from the bottom-K order statistics of pmax and margin,
    so we can hit tiny budgets exactly (K ~ O(budget)).
    """
    import numpy as np

    y_true = np.asarray(y_true).astype(int)
    probs  = np.asarray(probs, dtype=float)
    X      = np.asarray(X, dtype=float)

    N = len(y_true)
    preds  = probs.argmax(axis=1)
    is_err = (preds != y_true)

    pmax   = probs.max(axis=1)
    top2   = np.partition(probs, -2, axis=1)[:, -2:]
    margin = np.abs(top2[:,1] - top2[:,0])

    K = min(N, max(50, safety_mult * max(1, budget_count)))

    # Candidates just *above* the bottom-K values
    p_cand = np.unique(np.nextafter(np.sort(pmax)[:K]  + 1e-8, 1.0))
    d_cand = np.unique(np.nextafter(np.sort(margin)[:K] + 1e-8, 1.0))
    p_cand = np.concatenate([[0.0], p_cand])
    d_cand = np.concatenate([[0.0], d_cand])

    best = None
    best_key = None

    # global spread for dispersion (optional tie-breaker)
    if X.ndim == 2 and X.shape[0] > 1:
        gcent = X.mean(axis=0)
        gspread = np.mean(np.linalg.norm(X - gcent, axis=1)) + 1e-8
    else:
        gspread = 1.0

    for tau in p_cand:
        for delta in d_cand:
            U = (pmax < tau) | (margin < delta)
            nU = int(U.sum())
            if nU <= budget_count:
                cap  = (is_err & U).sum() / max(int(is_err.sum()), 1)
                A    = ~U
                risk = (is_err & A).sum() / max(int(A.sum()), 1)
                prec = (is_err & U).sum() / max(nU, 1) if nU > 0 else 0.0
                if nU >= 2 and X.ndim == 2:
                    c = X[U].mean(axis=0)
                    disp = float(np.mean(np.linalg.norm(X[U] - c, axis=1)) / gspread)
                else:
                    disp = 0.0

                # priority: ↑capture, ↓risk, ↑precision, ↓dispersion, ↓nU
                key = (cap, -risk, prec, -disp, -nU)
                if best is None or key > best_key:
                    best_key = key
                    best = dict(tau=float(tau), delta=float(delta),
                                abstain=float(nU / max(N,1)), capture=float(cap),
                                precision=float(prec), dispersion=float(disp),
                                risk_accept=float(risk))

    if best is None:
        # Guaranteed fallback: exactly-k by order statistic on pmax
        k = min(budget_count, max(N-1, 0))
        tau = float(np.nextafter(np.sort(pmax)[k], 1.0))
        delta = 0.0
        U = (pmax < tau)
        nU = int(U.sum())
        cap  = (is_err & U).sum() / max(int(is_err.sum()), 1)
        A    = ~U
        risk = (is_err & A).sum() / max(int(A.sum()), 1)
        prec = (is_err & U).sum() / max(nU, 1) if nU > 0 else 0.0
        best = dict(tau=float(tau), delta=float(delta),
                    abstain=float(nU / max(N,1)), capture=float(cap),
                    precision=float(prec), dispersion=0.0, risk_accept=float(risk))

    return best


# ───────────────────────────────────────────────────────────
#  Sensitivity analysis for (tau, delta)
#  - tau: ceiling on max softmax P_max
#  - delta: margin between top-2 probs (|p1 - p2|)
#  Flags if (P_max < tau) OR (margin < delta)
#  Returns: dict with 'grid' (rows: tau,delta,abstain,capture,precision,dispersion)
#           and 'chosen' – the pair minimizing abstain s.t. capture >= target_capture
# ───────────────────────────────────────────────────────────

def sensitivity_analysis(
    y_true, probs, X, taus, deltas,
    target_capture=0.80,
    max_abstain=None,         # e.g., 0.20 for a 20% review budget
    target_risk=None,         # e.g., 0.05 for ≤5% error among accepted
    mode="budget",            # "capture" | "budget" | "risk" | "knee"
    min_flagged=10
):
    """
    Grid evaluation for (tau, delta).
    Metrics per grid point:
      - abstain: fraction flagged as uncertain
      - capture: fraction of *errors* that are flagged (recall on error set)
      - precision: precision of the error detector among flagged
      - dispersion: relative spread of flagged X
      - risk_accept: error rate among *accepted* (non-flagged) predictions

    Selection modes:
      - "capture":    min abstain s.t. capture >= target_capture
      - "budget":     max capture   s.t. abstain <= max_abstain
      - "risk":       min abstain   s.t. risk_accept <= target_risk
      - "knee":       knee point on normalized (abstain, capture) curve
    """
    import numpy as np

    y_true = np.asarray(y_true).astype(int)
    probs  = np.asarray(probs, dtype=float)
    X      = np.asarray(X, dtype=float)
    assert probs.ndim == 2 and probs.shape[1] >= 2

    N      = len(y_true)
    preds  = probs.argmax(axis=1)
    is_err = (preds != y_true)
    n_err  = int(is_err.sum())

    pmax   = probs.max(axis=1)
    top2   = np.partition(probs, -2, axis=1)[:, -2:]
    margin = np.abs(top2[:, 1] - top2[:, 0])

    # normalizer for dispersion
    if X.ndim == 2 and X.shape[0] > 1:
        glob_centroid = X.mean(axis=0)
        glob_spread = np.mean(np.linalg.norm(X - glob_centroid, axis=1)) + 1e-8
    else:
        glob_spread = 1.0

    rows = []
    for tau in taus:
        for delta in deltas:
            U = (pmax < tau) | (margin < delta)   # uncertain mask (OR)
            nU = int(U.sum())

            abstain = nU / max(N, 1)
            capture = (is_err & U).sum() / max(n_err, 1) if n_err > 0 else 0.0
            precision = (is_err & U).sum() / max(nU, 1) if nU > 0 else 0.0

            # selective risk on accepted set
            A = ~U
            risk_accept = (is_err & A).sum() / max(int(A.sum()), 1)

            if nU >= 2 and X.ndim == 2:
                c = X[U].mean(axis=0)
                disp = np.mean(np.linalg.norm(X[U] - c, axis=1)) / glob_spread
            else:
                disp = np.nan

            rows.append([float(tau), float(delta),
                         float(abstain), float(capture),
                         float(precision), float(disp),
                         float(risk_accept)])

    grid = np.array(rows, dtype=float)  # cols: tau,delta,abstain,capture,precision,dispersion,risk_accept
    chosen = {}

    # Helper views
    abst = grid[:,2]; capt = grid[:,3]; prec = grid[:,4]; disp = np.nan_to_num(grid[:,5], nan=1e9)
    risk = grid[:,6]

    # Feasibility masks
    enough_pts = (abst * N >= max(min_flagged, 1))

    if mode == "capture":
        mask = (capt >= float(target_capture)) & enough_pts
        if max_abstain is not None:
            mask &= (abst <= float(max_abstain))
        cand = grid[mask]
        if cand.size:
            # minimize abstain, then dispersion, then maximize precision
            order = np.lexsort((-cand[:,4], cand[:,5], cand[:,2]))
            best = cand[order[0]]
            chosen = dict(tau=best[0], delta=best[1],
                          abstain=best[2], capture=best[3],
                          precision=best[4], dispersion=best[5],
                          risk_accept=best[6])

    elif mode == "budget":
        assert max_abstain is not None, "Provide max_abstain for mode='budget'."
        mask = (abst <= float(max_abstain)) & enough_pts
        cand = grid[mask]
        if cand.size:
            # maximize capture, then minimize dispersion, then maximize precision
            order = np.lexsort((-cand[:,4], cand[:,5], -cand[:,3]))
            best = cand[order[0]]
            chosen = dict(tau=best[0], delta=best[1],
                          abstain=best[2], capture=best[3],
                          precision=best[4], dispersion=best[5],
                          risk_accept=best[6])

    elif mode == "risk":
        assert target_risk is not None, "Provide target_risk for mode='risk'."
        mask = (risk <= float(target_risk)) & enough_pts
        cand = grid[mask]
        if cand.size:
            # minimize abstain, then dispersion, then maximize precision
            order = np.lexsort((-cand[:,4], cand[:,5], cand[:,2]))
            best = cand[order[0]]
            chosen = dict(tau=best[0], delta=best[1],
                          abstain=best[2], capture=best[3],
                          precision=best[4], dispersion=best[5],
                          risk_accept=best[6])

    elif mode == "knee":
        # Normalize to [0,1] and pick the point farthest from diagonal (abstain vs capture)
        if grid.size:
            a = (abst - abst.min()) / (abst.max() - abst.min() + 1e-12)
            c = (capt - capt.min()) / (capt.max() - capt.min() + 1e-12)
            score = c - a   # knee-like criterion
            k = int(np.argmax(score))
            best = grid[k]
            chosen = dict(tau=best[0], delta=best[1],
                          abstain=best[2], capture=best[3],
                          precision=best[4], dispersion=best[5],
                          risk_accept=best[6])
        # --- Final safety net: if no candidate satisfied constraints, pick a knee point ---
    if not chosen and grid.size:
        a = (abst - abst.min()) / (abst.max() - abst.min() + 1e-12)
        c = (capt - capt.min()) / (capt.max() - capt.min() + 1e-12)
        k = int(np.argmax(c - a))
        best = grid[k]
        chosen = dict(tau=best[0], delta=best[1],
                      abstain=best[2], capture=best[3],
                      precision=best[4], dispersion=best[5],
                      risk_accept=best[6])

    return {'grid': grid, 'chosen': chosen}


def save_capture_abstain_curve(grid: np.ndarray, output_folder: str, filename: str = "capture_vs_abstain.png"):
    """
    Plot capture (y) vs abstain (x) from the sensitivity grid and save to file.
    Also overlays the Pareto frontier (upper envelope at minimal abstain per capture).
    """
    A = grid[:, 2]  # abstain
    C = grid[:, 3]  # capture

    # Pareto frontier: for any given A, take max C; then sort by A
    import pandas as pd
    df = pd.DataFrame({"abstain": A, "capture": C})
    df = df.groupby("abstain", as_index=False).max().sort_values("abstain")
    # Monotone envelope
    env = df.copy()
    for i in range(1, len(env)):
        env.loc[i, "capture"] = max(env.loc[i, "capture"], env.loc[i-1, "capture"])

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.scatter(A, C, alpha=0.15, s=10, label="grid")
    ax.plot(env["abstain"], env["capture"], linewidth=2, label="Pareto envelope")
    ax.set_xlabel("Abstain fraction")
    ax.set_ylabel("Error capture")
    ax.set_ylim(0, 1); ax.set_xlim(0, 1)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_folder, filename)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info("Saved capture–abstain curve to %s", path)


def save_sensitivity_heatmaps(grid: np.ndarray, output_folder: str, prefix: str = "sens"):
    """
    Save heatmaps for capture and abstain over the (tau, delta) grid returned by sensitivity_analysis.
    Grid columns: [tau, delta, abstain, capture, precision, dispersion, risk_accept]
    """
    taus   = np.unique(grid[:, 0])
    deltas = np.unique(grid[:, 1])

    def _pivot(col_idx: int, atol: float = 1e-12):
        M = np.zeros((len(taus), len(deltas)), dtype=float)
        col = col_idx
        for i, tau in enumerate(taus):
            for j, delt in enumerate(deltas):
                mask = (np.isclose(grid[:, 0], tau, atol=atol)) & (np.isclose(grid[:, 1], delt, atol=atol))
                row = grid[mask]
                if row.size == 0:
                    M[i, j] = np.nan
                else:
                    M[i, j] = row[0, col]
        return M
   

    cap = _pivot(3)
    absn = _pivot(2)

    for name, M in [("capture", cap), ("abstain", absn)]:
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(M, aspect='auto', origin='lower')
        ax.set_xticks(range(len(deltas)))
        ax.set_yticks(range(len(taus)))
        ax.set_xticklabels([f"{d:.2f}" for d in deltas], rotation=45, ha='right')
        ax.set_yticklabels([f"{t:.2f}" for t in taus])
        ax.set_xlabel("delta")
        ax.set_ylabel("tau")
        ax.set_title(f"Sensitivity heatmap: {name}")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        path = os.path.join(output_folder, f"{prefix}_{name}.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        logger.info("Saved sensitivity heatmap to %s", path)

        
def extract_penultimate_features(model: nn.Module, xb: torch.Tensor) -> torch.Tensor:
    """
    Return the activation after the first ComplexLinear + modReLU (penultimate representation).
    Shape: [N, 2*hidden_features]
    """
    with torch.no_grad():
        h = model.fc1(xb)
        h = modrelu(h, bias=model.bias_modrelu)
    return h

def save_feature_embedding_2d(model: nn.Module, loader, output_folder: str,
                              device: str = "cpu", method: str = "tsne",
                              max_points: int = 5000, seed: int = 12345,
                              filename: str = "feature_embedding_2d.png"):
    """
    Project penultimate features to 2D (t-SNE or UMAP) and save a scatter plot.
    This demonstrates non-linear separability learned by the CVNN.
    """
    from sklearn.manifold import TSNE
    try:
        import umap
        HAS_UMAP = True
    except Exception:
        HAS_UMAP = False

    model.eval()
    Xf, y = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            feat = extract_penultimate_features(model, xb)
            Xf.append(feat.cpu().numpy())
            y.extend(yb.numpy().tolist())

    Xf = np.vstack(Xf)
    y = np.asarray(y)

    if len(Xf) > max_points:
        idx = np.random.RandomState(seed).choice(len(Xf), max_points, replace=False)
        Xf = Xf[idx]; y = y[idx]

    if method.lower() == "umap" and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
        Z = reducer.fit_transform(Xf)
        title = "UMAP of penultimate features"
    else:
        reducer = TSNE(n_components=2, init="random", learning_rate="auto",
                       perplexity=30, random_state=seed)
        Z = reducer.fit_transform(Xf)
        title = "t-SNE of penultimate features"

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=y, palette="Set2", s=10, ax=ax, legend=False)
    ax.set_xlabel("dim-1"); ax.set_ylabel("dim-2")
    ax.set_title(title)
    fig.tight_layout()
    path = os.path.join(output_folder, filename)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info("Saved 2D feature embedding to %s", path)



# Constants imported here to be used by process_record
LOWCUT = 0.5
HIGHCUT = 40.0
ORDER = 2
FS = 360
WINDOW_SIZE = 128
PRE_SAMPLES = 50
