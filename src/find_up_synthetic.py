# Copyright (c) 2025 Piotr Migus
# This code is licensed under the MIT License.
# See the LICENSE file in the repository root for full license information.
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import csv
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

############################################################
# 1. Activation function: modReLU
############################################################
def modrelu(input, bias=0.0):
    """
    Implements modReLU as introduced by Arjovsky et al. (2016) or Trabelsi et al. (2018).
    
    This activation is designed for complex-valued inputs:
       - We compute |z| = sqrt(x_r^2 + x_i^2)
       - Shifted = |z| + bias
       - If (Shifted <= 0), output = 0
         else output = (Shifted / |z|) * z
       
    This allows us to preserve the phase information while applying a ReLU-like threshold
    on the magnitude. Conceptually, it zeros out small magnitudes (where |z|+bias <= 0),
    but if |z| + bias > 0, it rescales z by (|z| + bias)/|z|.
    
    Parameters
    ----------
    input : torch.Tensor
        A tensor of shape (batch_size, 2 * num_features). We interpret the first half
        of the features as the real part, and the second half as the imaginary part.
    bias : float or torch.Tensor
        A scalar (or possibly a learnable tensor) that shifts the magnitude threshold.
    
    Returns
    -------
    torch.Tensor
        The activated tensor of the same shape (batch_size, 2 * num_features).
        The real and imaginary parts are combined along the feature dimension.
    """
    half = input.size(1) // 2
    xr = input[:, :half]  # real part
    xi = input[:, half:]  # imaginary part

    # Ensure 'bias' is on the correct device as a tensor
    if not isinstance(bias, torch.Tensor):
        bias = torch.tensor(bias, device=input.device)

    # Compute magnitude safely
    mag = torch.sqrt(torch.clamp(xr**2 + xi**2, min=1e-9))

    # Shift the magnitude
    shifted = mag + bias

    # Build a mask where (shifted > 0)
    mask = (shifted > 0).float()

    # Scale factor for points that remain active
    sc = shifted / mag  # This is (|z| + bias)/|z|

    # Apply mask
    xr_act = sc * xr * mask
    xi_act = sc * xi * mask

    return torch.cat([xr_act, xi_act], dim=1)


############################################################
# 2. ComplexLinear layer
############################################################
class ComplexLinear(nn.Module):
    """
    A linear (fully-connected) layer designed for complex inputs. Concretely:
        z_in = x_r + i*x_i
        w    = w_r + i*w_i
    The layer output is given by:
        output = (w_r + i*w_i)*(x_r + i*x_i) + (b_r + i*b_i)
                = (w_r*x_r - w_i*x_i) + i (w_r*x_i + w_i*x_r)
                
    This implementation internally represents each complex number as two real values:
    (real_part, imaginary_part). Hence, the input shape is (batch_size, 2*in_features).
    
    The forward pass partitions the input into real and imaginary parts, performs
    the appropriate complex multiplication via matrix operations, and optionally
    adds a complex bias.
    """
    def __init__(self, in_features, out_features, bias=True):
        """
        Constructs a ComplexLinear layer.

        Parameters
        ----------
        in_features : int
            The number of input complex features (each complex feature is effectively
            2 real numbers).
        out_features : int
            The number of output complex features.
        bias : bool
            If True, a complex bias is added (with real and imaginary parts).
        """
        super(ComplexLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # We store separate real and imaginary weights
        self.weight_real = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias_real = nn.Parameter(torch.Tensor(out_features))
            self.bias_imag = nn.Parameter(torch.Tensor(out_features))
        else:
            self.bias_real = None
            self.bias_imag = None

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes weights (and bias if present) using Xavier initialization for
        the real and imaginary parts, and sets bias terms to zero.
        """
        nn.init.xavier_uniform_(self.weight_real)
        nn.init.xavier_uniform_(self.weight_imag)
        if self.bias_real is not None:
            nn.init.zeros_(self.bias_real)
            nn.init.zeros_(self.bias_imag)

    def forward(self, input):
        """
        Performs a forward pass of the ComplexLinear layer.
        
        Parameters
        ----------
        input : torch.Tensor
            Shape (batch_size, 2*in_features). We interpret the first 'in_features'
            dimensions as the real part (xr) and the next 'in_features' as the
            imaginary part (xi).

        Returns
        -------
        torch.Tensor
            Shape (batch_size, 2*out_features). The output combines real and
            imaginary parts along the feature dimension.
        """
        batch_size = input.size(0)
        half = input.size(1) // 2
        xr = input[:, :half]   # real part
        xi = input[:, half:]   # imaginary part

        # Real and imaginary matrix multiplication
        # real = W_real*xr - W_imag*xi
        real_part = torch.matmul(xr, self.weight_real.T) - torch.matmul(xi, self.weight_imag.T)
        # imag = W_real*xi + W_imag*xr
        imag_part = torch.matmul(xi, self.weight_real.T) + torch.matmul(xr, self.weight_imag.T)

        # Add bias if present
        if self.bias_real is not None:
            real_part = real_part + self.bias_real
            imag_part = imag_part + self.bias_imag

        # Concatenate real and imaginary parts along the last dimension
        output = torch.cat([real_part, imag_part], dim=1)
        return output


############################################################
# 3. Minimal Complex Neural Network
############################################################
class SimpleComplexNet(nn.Module):
    """
    A minimal feed-forward neural network for complex inputs, composed of:
      1) A ComplexLinear hidden layer + modReLU,
      2) A ComplexLinear output layer (optionally without activation).
    
    This network expects an input with shape (batch_size, 2*in_features),
    e.g. 2 complex features = 4 real dimensions. The hidden layer similarly
    handles complex activations, and the final output has 'out_features' complex units.
    """
    def __init__(self, in_features=4, hidden_features=8, out_features=2, bias=0.0):
        """
        Initialize a simple two-layer complex neural network.

        Parameters
        ----------
        in_features : int
            Number of input complex features (each complex feature => 2 real dims).
        hidden_features : int
            Size of the hidden layer (complex). So effectively 2D real dims * hidden_features.
        out_features : int
            Number of output complex features. Usually 2 => (2 real + 2 imag) = 4 outputs total.
        bias : float
            The bias offset used in modReLU. (Could be a scalar or param.)
        """
        super(SimpleComplexNet, self).__init__()
        self.fc1 = ComplexLinear(in_features, hidden_features)
        self.fc2 = ComplexLinear(hidden_features, out_features)
        self.bias_modrelu = bias  # this offset is used in modReLU

    def forward(self, x):
        """
        Forward pass:
          1) The first ComplexLinear layer + modReLU activation,
          2) The second ComplexLinear layer (no further activation by default).
        """
        # Step 1: hidden layer with modReLU
        x = modrelu(self.fc1(x), bias=self.bias_modrelu)
        # Step 2: final linear layer
        x = self.fc2(x)
        return x


############################################################
# 4. Decision function: from complex outputs to logits
############################################################
def complex_modulus_to_logits(output):
    """
    Converts complex outputs (in shape (batch_size, 2*out_features))
    into modulus-based logits with shape (batch_size, out_features).
    
    For each complex output c_k = (re_k, im_k), we compute its modulus sqrt(re_k^2 + im_k^2).
    If out_features = 2, then we produce two values (|c_1|, |c_2|).
    
    Parameters
    ----------
    output : torch.Tensor
        Shape (batch_size, 2*out_features). The first half of features is the real part,
        the second half is the imaginary part.
    
    Returns
    -------
    torch.Tensor
        Shape (batch_size, out_features). Each entry is the modulus of the corresponding
        complex output.
    """
    half = output.size(1) // 2
    real_part = output[:, :half]
    imag_part = output[:, half:]
    mod = torch.sqrt(real_part**2 + imag_part**2 + 1e-9)
    return mod  # e.g. shape (batch_size, out_features)


############################################################
# 5. Synthetic data generation in C^2
############################################################
def generate_synthetic_data(num_samples_per_class=2000):
    """
    Generate a synthetic dataset of two classes (0,1) in C^2 (4 real dimensions).

    Each sample represents (z1, z2) => (Re(z1), Re(z2), Im(z1), Im(z2)).
    The function returns:
       X: a tensor of shape (num_samples, 4)
       y: a tensor of shape (num_samples,) with labels (0 or 1)

    We define two classes:
      Class 0:
        z1 ~ amplitude ~ 1 +/- 0.1,   phase ~ 0   +/- 0.3
        z2 ~ amplitude ~ 2 +/- 0.2,   phase ~ pi/4 +/- 0.3
      Class 1:
        z1 ~ amplitude ~ 1.5 +/- 0.15, phase ~ pi/2  +/- 0.3
        z2 ~ amplitude ~ 2   +/- 0.2,  phase ~ pi    +/- 0.3
    """
    # local helper
    def generate_complex(amplitude_mean, amplitude_std, phase_mean, phase_std, n):
        r = torch.normal(mean=amplitude_mean, std=amplitude_std, size=(n,))
        phi = torch.normal(mean=phase_mean, std=phase_std, size=(n,))
        x_real = r * torch.cos(phi)
        x_imag = r * torch.sin(phi)
        return x_real, x_imag

    # Class 0
    x1r0, x1i0 = generate_complex(1.0, 0.1, 0.0, 0.3, num_samples_per_class)
    x2r0, x2i0 = generate_complex(2.0, 0.2, np.pi/4, 0.3, num_samples_per_class)

    # Class 1
    x1r1, x1i1 = generate_complex(1.5, 0.15, np.pi/2, 0.3, num_samples_per_class)
    x2r1, x2i1 = generate_complex(2.0, 0.2, -np.pi, 0.3, num_samples_per_class)

    # Stack data for each class
    X0 = torch.stack([x1r0, x2r0, x1i0, x2i0], dim=1)  # shape: (num_samples_per_class, 4)
    y0 = torch.zeros(num_samples_per_class, dtype=torch.long)

    X1 = torch.stack([x1r1, x2r1, x1i1, x2i1], dim=1)
    y1 = torch.ones(num_samples_per_class, dtype=torch.long)

    # Combine and shuffle
    X = torch.cat([X0, X1], dim=0)
    y = torch.cat([y0, y1], dim=0)

    indices = torch.randperm(X.size(0))
    X = X[indices]
    y = y[indices]

    return X, y


############################################################
# 6. Example training loop
############################################################
def train_model(model, X_train, y_train, epochs=50, lr=1e-3):
    """
    Train the given complex-valued network on (X_train, y_train).
    We interpret X_train as shape (num_samples, 4) if in_features=2.

    Steps:
      1) Forward pass => compute logit moduli (via complex_modulus_to_logits).
      2) CrossEntropyLoss => standard classification loss on real logits.
      3) Adam optimizer step.

    Parameters
    ----------
    model : nn.Module
        The SimpleComplexNet or similar model.
    X_train : torch.Tensor
        Training samples (num_samples, 4).
    y_train : torch.Tensor
        Training labels of shape (num_samples,).
    epochs : int
        Number of epochs for gradient descent training.
    lr : float
        Learning rate for Adam.

    Returns
    -------
    None
        Prints intermediate results every 10 epochs.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # Forward
        out = model(X_train)
        logits = complex_modulus_to_logits(out)
        loss = criterion(logits, y_train)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print stats every 10 epochs
        if (epoch+1) % 10 == 0:
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                acc = (preds == y_train).float().mean()
            print(f"Epoch {epoch+1:02d}/{epochs}, Loss={loss.item():.4f}, Accuracy={acc.item():.4f}")


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on a test set.

    Parameters
    ----------
    model : nn.Module
        The trained model.
    X_test : torch.Tensor
        Test samples, shape (num_test, 4).
    y_test : torch.Tensor
        Test labels, shape (num_test,).

    Returns
    -------
    float
        Accuracy on the test set.
    """
    with torch.no_grad():
        out = model(X_test)
        logits = complex_modulus_to_logits(out)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y_test).float().mean()
    return acc.item()


############################################################
# 7. Finding "uncertain" points
############################################################
def find_uncertain_points(
    model: nn.Module,
    X_data: torch.Tensor,
    y_data: torch.Tensor,
    prob_thr: float | None = 0.6,
    margin_delta: float | None = None,
    temperature: torch.Tensor | None = None,
    threshold: float | None = None,   # backward-compat alias for margin_delta
    device: str = 'cpu'
) -> list[dict]:
    """
    Flag samples as 'uncertain' if:
      (max softmax < prob_thr) OR (top-2 probability margin < margin_delta).

    Works for binary and multi-class. Supports optional temperature scaling.
    If only one of prob_thr / margin_delta is provided, uses that one.
    If 'threshold' is given, it's treated as margin_delta for backward compatibility.
    """

    # --- Backward compatibility: map old 'threshold' to 'margin_delta'
    if margin_delta is None and threshold is not None:
        margin_delta = threshold

    model.eval()
    X_data = X_data.to(device).float()

    with torch.no_grad():
        logits = complex_modulus_to_logits(model(X_data))
        if temperature is not None:
            logits = logits / temperature.to(logits.device)
        probs = torch.softmax(logits, dim=1)

    # Max probability and top-2 margin (multi-class safe)
    topk_vals = torch.topk(probs, k=min(2, probs.size(1)), dim=1).values  # (N, K)
    max_p = topk_vals[:, 0]
    if probs.size(1) > 1:
        margin = topk_vals[:, 0] - topk_vals[:, 1]  # >= 0
    else:
        # degenerate 1-class case: define margin to keep API intact
        margin = 2 * max_p - 1.0

    # Build uncertainty mask using OR over the available criteria
    mask = torch.zeros_like(max_p, dtype=torch.bool)
    eps = 1e-6
    if prob_thr is not None:
        mask |= (max_p <= prob_thr + eps)
    if margin_delta is not None:
        mask |= (margin <= margin_delta + eps)

    # Debug / quick stats
    print(f"[INFO] min max-p={max_p.min():.3f} | max max-p={max_p.max():.3f} | "
          f"mean margin={margin.mean():.3f} | n_uncertain={int(mask.sum().item())}")

    # Pack results
    idxs = torch.nonzero(mask, as_tuple=False).flatten().cpu().tolist()
    unsure = []
    for i in idxs:
        unsure.append({
            "index":      int(i),
            "X":          X_data[i].detach().cpu().tolist(),
            "true_label": int(y_data[i]),
            "pred":       int(probs[i].argmax().item()),
            "prob":       probs[i].detach().cpu().tolist(),
            "logits":     logits[i].detach().cpu().tolist(),
            "maxp":       float(max_p[i].item()),
            "margin":     float(margin[i].item()),
        })
    return unsure


def save_uncertain_to_csv(uncertain_points, folder, filename='uncertain_points.csv'):
    """
    Save the list of uncertain points to a CSV file with columns:
       index, X, true_label, p1, p2

    Parameters
    ----------
    uncertain_points : list of dict
        Output from 'find_uncertain_points'.
    folder : str
        Directory to place the CSV file. Will be created if not existing.
    filename : str
        CSV filename. Defaults to 'uncertain_points.csv'.

    Returns
    -------
    None
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    full_path = os.path.join(folder, filename)

    fields = ['index', 'X', 'true_label', 'p1', 'p2']
    with open(full_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fields)
        for up in uncertain_points:
            row = [
                up['index'],
                up['X'],
                up['true_label'],
                up['prob'][0],
                up['prob'][1]
            ]
            writer.writerow(row)


############################################################
# 8. Visualization of results (new functionality)
############################################################
def visualize_uncertainty(model, X_data, y_data, uncertain_points, folder, threshold=0.6):
    """
    Generates three plots:
      - Distribution of test points in 2D space (PCA reduction) colored according to predicted labels.
      - Distribution of test points colored according to maximum probability values, highlighting uncertain points.
      - Histogram of maximum probability distribution with uncertainty threshold marked.
    """
    model.eval()
    with torch.no_grad():
        out = model(X_data.float())
        half = out.size(1) // 2
        xr = out[:, :half]
        xi = out[:, half:]
        logits = torch.sqrt(xr**2 + xi**2 + 1e-9)
        probs = torch.softmax(logits, dim=1)
        predicted = torch.argmax(probs, dim=1).numpy()
        max_prob = torch.max(probs, dim=1).values.numpy()

    # Dimensionality reduction of inputs (4D) to 2D using PCA
    X_np = X_data.numpy()
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_np)

    # 1. Scatter plot - Predicted labels
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
                          c=predicted, cmap='viridis', alpha=0.7)
    plt.title('Test Data: Predicted Labels')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(scatter, label='Predicted Label')
    plt.grid(True)
    pred_plot_path = os.path.join(folder, 'predicted_labels.png')
    plt.savefig(pred_plot_path)
    plt.close()

    # 2. Scatter plot - Maximum probability with highlighting of uncertain points
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
                          c=max_prob, cmap='coolwarm', alpha=0.7)
    plt.title('Test Data: Maximum Probability')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(scatter, label='Max Probability')
    # Highlighting uncertain points (indices from uncertain_points)
    uncertain_indices = [pt['index'] for pt in uncertain_points]
    if uncertain_indices:
        plt.scatter(X_reduced[uncertain_indices, 0],
                    X_reduced[uncertain_indices, 1],
                    marker='*', s=150, c='red', label='Uncertain Points')
        plt.legend()
    plt.grid(True)
    prob_plot_path = os.path.join(folder, 'max_probability.png')
    plt.savefig(prob_plot_path)
    plt.close()

    # 3. Histogram of maximum probabilities distribution
    plt.figure(figsize=(8, 6))
    plt.hist(max_prob, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
    plt.title('Distribution of Maximum Probability')
    plt.xlabel('Max Probability')
    plt.ylabel('Number of samples')
    plt.legend()
    plt.grid(True)
    hist_plot_path = os.path.join(folder, 'max_probability_distribution.png')
    plt.savefig(hist_plot_path)
    plt.close()
    
    print(f"[INFO] Visualizations saved in folder: {folder}")
