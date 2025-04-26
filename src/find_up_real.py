
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
import numpy as np
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
from src.find_up_synthetic import SimpleComplexNet, complex_modulus_to_logits, find_uncertain_points

# Configure module-level logger
logger = logging.getLogger(__name__)

def tune_temperature(model: nn.Module,
                     val_loader: torch.utils.data.DataLoader,
                     device: str = "cpu") -> torch.Tensor:
    """
    Tunes a single scalar temperature parameter for model calibration using temperature scaling.
    
    This function implements the method proposed by Guo et al. (2017) to calibrate the confidence
    of a classification model. It finds the optimal temperature `T` such that the softmax output
    probabilities become better aligned with the true correctness likelihood (i.e., calibrated).
    
    The tuning is performed on a validation set by minimizing the cross-entropy loss with 
    temperature-scaled logits: softmax(logits / T). The optimization is done using L-BFGS.

    Parameters:
    ----------
    model : nn.Module
        The trained PyTorch classification model.
    
    val_loader : torch.utils.data.DataLoader
        DataLoader for the validation dataset used to tune the temperature.
    
    device : str, optional (default="cpu")
        The device to perform computations on ("cpu" or "cuda").

    Returns:
    -------
    torch.Tensor
        A scalar tensor containing the optimal temperature value (on CPU).
    
    """
    model.eval()
    logits_list, labels_list = [], []

    # Collect all logits and true labels from the validation set
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = complex_modulus_to_logits(model(xb))
            logits_list.append(logits)
            labels_list.append(yb)

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    # Initialize temperature parameter (requires_grad=True for optimization)
    T = nn.Parameter(torch.ones(1, device=device) * 1.5)

    # Use L-BFGS optimizer for smooth scalar optimization
    optimizer = optim.LBFGS([T], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    # Define closure function required by L-BFGS
    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / T, labels)
        loss.backward()
        return loss

    # Run the optimization
    optimizer.step(closure)

    # Return calibrated temperature as CPU tensor (for portability)
    return T.detach().cpu()



def compress_to_C2(X: np.ndarray, window_size=128):
    """
    Compress a multichannel complex-valued signal into a compact 4-dimensional real feature vector.

    Each input sample is assumed to contain concatenated real and imaginary parts of a complex-valued 
    time signal across multiple channels. This function reconstructs the complex analytic signal 
    per channel, and extracts simple statistical descriptors:
        - Mean (mu) of the full signal (across all channels)
        - Mean slope (first difference) of the signal
    
    These are decomposed into real and imaginary parts to produce a final feature vector in ℝ⁴.

    Parameters:
    ----------
    X : np.ndarray of shape (N, 2 * window_size * nchannels)
        Input matrix where each row is a sample containing real and imaginary parts concatenated.
    
    window_size : int, optional (default=128)
        Number of time steps per channel per signal (i.e., length of real/imag part per channel).

    Returns:
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

    Parameters:
    ----------
    X : np.ndarray of shape (N, D)
        Real-valued input matrix.
    
    method : str, optional (default='split_pca')
        Dimensionality reduction and transformation method.

    n_real : int, optional (default=2)
        Number of real (and imaginary) components in the output. Total output dimension will be 2 * n_real.

    Returns:
    -------
    X_complex : np.ndarray of shape (N, 2 * n_real)
        Transformed feature matrix ready for complex-valued neural networks.

    Raises:
    ------
    ValueError:
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

    else:
        raise ValueError(f"Unknown prepare_complex_input method '{method}'")

    return X_complex



def create_dataloaders(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       batch_size: int = 128, num_workers: int = 4,
                       pin_memory: bool = True):
    """
    Prepare PyTorch DataLoaders for training and testing with standardized input and class-balanced sampling.

    This function performs three main operations:
      1. Standardizes the input features using z-score normalization (mean=0, std=1).
      2. Converts NumPy arrays into PyTorch tensors.
      3. Builds DataLoaders with:
         - A `WeightedRandomSampler` for the training set to address class imbalance.
         - A deterministic sampler for the test set.

    Parameters:
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

    Returns:
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

    # Compute class weights (inverse frequency) for balanced sampling
    counts = np.bincount(y_train)
    weights = 1.0 / counts
    samp_weights = weights[y_train]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(samp_weights, dtype=torch.double),
        num_samples=len(samp_weights)
    )

    logger.info("WeightedRandomSampler created: inverse class frequencies %s", counts)

    # Create DataLoaders
    train_loader = DataLoader(
        dataset=TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        dataset=TensorDataset(X_te_t, y_te_t),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, test_loader



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

    Parameters:
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

    Returns:
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
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    return history, best_state

def save_plots(history: dict, output_folder: str, fold: int):
    """
    Save a plot of the training loss and validation accuracy over epochs for a single cross-validation fold.

    This function visualizes the training process for one fold by plotting:
      - Training loss vs. epoch
      - Validation accuracy vs. epoch

    The resulting plot is saved as a PNG file in the specified output folder. 
    It helps assess convergence, potential overfitting, and the general learning dynamics.

    Parameters:
    ----------
    history : dict
        A dictionary containing:
            - 'train_loss': List of training losses per epoch.
            - 'val_acc': List of validation accuracies per epoch.

    output_folder : str
        Directory path where the plot will be saved.

    fold : int
        The fold index (used for naming the output file).

    Returns:
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

    Parameters:
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

    Returns:
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

    Parameters:
    ----------
    histories : list of dict
        A list of dictionaries (one per fold), where each dict contains:
            - 'train_loss': List[float] – training loss per epoch
            - 'val_acc': List[float] – validation accuracy per epoch

    output_folder : str
        Path to the directory where the plot will be saved.

    Returns:
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

    logger.info("Saved overall learning‑curve to %s", path)


def save_calibration_curve(y_true, y_prob, output_folder):
    """
    Generate and save a reliability diagram (calibration curve) for binary classification.

    A calibration curve compares predicted probabilities with actual frequencies of the 
    positive class. This function helps evaluate whether the model's predicted 
    confidence scores (e.g., softmax outputs) are well-calibrated.

    For example, among all predictions where the model outputs a probability of ~0.7,
    we would expect ~70% of those to actually be class 1. A well-calibrated model will
    lie close to the diagonal y = x line.

    Parameters:
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth binary labels (0 or 1).
    
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class (class 1).
    
    output_folder : str
        Path to the directory where the calibration plot will be saved.

    Returns:
    -------
    None

    
    """
    # Compute calibration statistics using scikit-learn
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

    # Plot reliability curve
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(prob_pred, prob_true, marker='o', label='Model')
    ax.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Perfect calibration')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed frequency')
    ax.set_title('Reliability diagram')
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, 'calibration_curve.png'), dpi=300)
    plt.close(fig)

    logger.info("Saved calibration curve")

def save_uncertainty_hist(y_prob, threshold, output_folder):
    """
    Generate and save a histogram of predicted max probabilities, highlighting the uncertainty zone.

    This function visualizes the distribution of maximum predicted class probabilities (i.e., model confidence).
    It shades the region below a given threshold to indicate predictions considered "uncertain".

    This is useful for:
    - Analyzing how confident the model is overall.
    - Visualizing how many samples fall into a low-confidence (uncertain) region.
    - Supporting selective prediction or abstention-based decision-making.

    Parameters:
    ----------
    y_prob : array-like of shape (n_samples,)
        Maximum softmax probabilities for each prediction. Typically:
        `y_prob = torch.max(softmax(logits), dim=1).values`.

    threshold : float
        Confidence threshold below which predictions are considered uncertain.

    output_folder : str
        Path to the directory where the histogram will be saved.

    Returns:
    -------
    None

   
    """
    fig, ax = plt.subplots(figsize=(5, 3))

    # Histogram of confidence scores
    ax.hist(y_prob, bins=30, alpha=0.7)

    # Shade area below threshold
    ax.axvspan(0, threshold, color='red', alpha=0.15, label=f'Uncertain < {threshold}')

    ax.set_xlabel('Max softmax probability')
    ax.set_ylabel('Number of samples')
    ax.legend()

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

    Parameters:
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

    Returns:
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

    Parameters:
    ----------
    results_dict : dict[str, float]
        Dictionary mapping variant labels (e.g., 'baseline', 'no_dropout') to their
        corresponding performance scores (e.g., F1-score).

    output_folder : str
        Path to the directory where the bar plot will be saved.

    Returns:
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



def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Compute the Expected Calibration Error (ECE) for a binary classifier.

    ECE is a scalar summary statistic that quantifies the difference between 
    confidence estimates and actual accuracy across predicted probability bins.

    For example, if a model predicts class 1 with 80% confidence on 100 samples,
    and only 70 of them are actually class 1, this bin has a 10% calibration gap.

    The lower the ECE, the better calibrated the model is.

    Parameters:
    ----------
    y_true : array-like of shape (n_samples,)
        True binary class labels (0 or 1).

    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class (class 1), typically
        the output of a softmax or sigmoid.

    n_bins : int, optional (default=10)
        Number of bins to divide the [0,1] probability range into.

    Returns:
    -------
    ece : float
        Expected Calibration Error — lower values indicate better calibration.

    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        # Identify samples in current bin
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.any():
            # Accuracy in bin (based on thresholding y_prob >= 0.5)
            acc_bin = (y_true[mask] == (y_prob[mask] >= 0.5)).mean()
            # Average predicted confidence in bin
            conf_bin = y_prob[mask].mean()
            # Weighted contribution to ECE
            ece += np.abs(acc_bin - conf_bin) * mask.mean()

    return float(ece)



# Constants imported here to be used by process_record
LOWCUT = 0.5
HIGHCUT = 40.0
ORDER = 2
FS = 360
WINDOW_SIZE = 128
PRE_SAMPLES = 50
