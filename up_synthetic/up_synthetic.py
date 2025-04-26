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

    
from src.find_up_synthetic import modrelu
from src.find_up_synthetic import ComplexLinear
from src.find_up_synthetic import SimpleComplexNet
from src.find_up_synthetic import complex_modulus_to_logits
from src.find_up_synthetic import generate_synthetic_data
from src.find_up_synthetic import train_model
from src.find_up_synthetic import evaluate_model
from src.find_up_synthetic import find_uncertain_points
from src.find_up_synthetic import save_uncertain_to_csv
from src.find_up_synthetic import visualize_uncertainty



############################################################
# Main section (test + identification of uncertain points)
############################################################
if __name__ == "__main__":
    """
    Example usage:
      1) Generate a synthetic dataset in C^2.
      2) Split into train/test.
      3) Build a SimpleComplexNet model, train it.
      4) Evaluate on test set.
      5) Find uncertain points (where max(prob) < threshold).
      6) Save them to CSV (uncertain_synthetic.csv).
      7) Save model parameters as well.
    """
    

    # 1. Generate synthetic data
    X, y = generate_synthetic_data(num_samples_per_class=200)
    print("Data shape:", X.shape, "Labels shape:", y.shape)

    # 2. Train/test split
    train_size = int(0.8 * X.size(0))
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test  = X[train_size:]
    y_test  = y[train_size:]

    # 3. Build the complex network model
    #    Note: in_features=2 means 2 complex dims => 4 real dims
    model = SimpleComplexNet(in_features=2, hidden_features=16, out_features=2, bias=0.1)

    # 4. Training
    train_model(model, X_train, y_train, epochs=50, lr=1e-3)

    # 5. Evaluation
    test_acc = evaluate_model(model, X_test, y_test)
    print(f"[INFO] Test accuracy = {test_acc:.4f}")

    # 6. Identify uncertain points in test set
    uncertain_pts = find_uncertain_points(model, X_test, y_test, threshold=0.1)
    print(f"[INFO] Number of uncertain points (threshold=0.1): {len(uncertain_pts)}")

    # 7. Save to CSV
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if len(uncertain_pts) > 0:
        save_uncertain_to_csv(uncertain_pts, current_dir, 'uncertain_synthetic.csv')
        print("[INFO] Saved uncertain points to 'uncertain_synthetic.csv'.")
    else:
        print("[INFO] No uncertain points found at threshold=0.1.")

    # Also save model parameters to a .pt file
    model_save_path = os.path.join(current_dir, 'model_parameters.pt')
    torch.save(model.state_dict(), model_save_path)
    print(f"[INFO] Model parameters saved to {model_save_path}")
    
    # Optional: quick preview of the first few uncertain points
    for i in range(min(5, len(uncertain_pts))):
        print(uncertain_pts[i])

    visualize_uncertainty(model, X_test, y_test, uncertain_pts, current_dir, threshold=0.1)
