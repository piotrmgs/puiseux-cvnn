
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

from src.local_analysis import benchmark_local_poly_approx_and_puiseux
from src.local_analysis import evaluate_poly_approx_quality
from src.local_analysis import polynomial_features_complex
from src.local_analysis import load_uncertain_points
from src.local_analysis import local_poly_approx_complex   
from src.local_analysis import puiseux_uncertain_point


    


if __name__ == "__main__":
    """
    Main processing pipeline:
    1. Load pretrained model and uncertain points from CSV.
    2. For selected uncertain points, construct local polynomial approximations
        and compute Puiseux expansions.
    3. Benchmark the process, evaluate approximation quality, and save results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # Load pretrained neural model
    model = SimpleComplexNet(in_features=2, hidden_features=16, out_features=2, bias=0.1).to(device)

    # Load model parameters
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'up_synthetic', 'model_parameters.pt')
    model_path = os.path.abspath(model_path)

    
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[INFO] Loaded parameters from {model_path}")

    # Load uncertain points from CSV
    csv_path = os.path.join(current_dir, '..', 'up_synthetic', 'uncertain_synthetic.csv')
    csv_path = os.path.abspath(csv_path)
    
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV with uncertain points not found: {csv_path}")
    up_list = load_uncertain_points(csv_path)
    print(f"[INFO] Loaded {len(up_list)} uncertain points from {csv_path}")

    # Process uncertain points
    
    
    for i, up in enumerate(up_list):  
        print(f"\n=== POINT # {i} ===")
        print("[DATA]",up)
        xstar = np.array(up['X'], dtype=np.float32)  # 4D

        # Construct local polynomial approximation and Puiseux expansions
        delta = 0.1
        polynom_degree = 4
        nsamples = 200
        F_hat_expr = local_poly_approx_complex(
            model,
            xstar,
            delta=delta,
            degree=polynom_degree,
            n_samples=nsamples,
            device=device,
            remove_linear=True
        )
 
        # Benchmark and compute expansions
        times_dict, expr_poly, expansions_list = benchmark_local_poly_approx_and_puiseux(
            model=model,
            xstar=xstar,
            local_poly_func=local_poly_approx_complex,   # or your function name
            puiseux_func=F_hat_expr,
            delta=delta,
            degree=polynom_degree,
            n_samples=nsamples,
            device=device,
            do_factor=True,
            do_simplify=True,
            puiseux_prec=4
        )

        expansions_list_uncertain = puiseux_uncertain_point(
            F_hat_expr,
            prec=4,
            base_point=xstar
        )

        # Evaluate polynomial approximation quality
        metrics_dict = evaluate_poly_approx_quality(
            model=model,
            poly_expr=expr_poly,
            xstar=xstar,
            delta=delta,
            n_samples=300,   # e.g. more samples for testing
            device=device
        )

        # Save results (timing, expansions, metrics) to text file
        #out_txt_path = f"/Users/piotrmigus/Desktop/research/ML-puis/synt-real/benchmark_point{i}.txt"
        out_txt_path = os.path.join(current_dir, f'benchmark_point{i}.txt')
        with open(out_txt_path, "w") as ff:
            ff.write(f"=== Local Polynomial + Puiseux Benchmark for point #{i} ===\n")
            ff.write(f"Point coordinates (xstar): {xstar.tolist()}\n")
            ff.write(f"degree={polynom_degree}, n_samples={nsamples}, delta={delta}\n\n")

            ff.write("---- 1) Timing results ----\n")
            for k, v in times_dict.items():
                ff.write(f"{k}: {v:.6f} sec\n")
            
            ff.write("\n---- 2) Final polynomial expression ----\n")
            ff.write(str(expr_poly) + "\n")

            ff.write("\n---- 3) Puiseux expansions at origin ----\n")
            for idx, exp_item in enumerate(expansions_list):
                ff.write(f"Expansion {idx+1}: {exp_item}\n")

            ff.write("\n---- 4) Puiseux expansions at uncertain point ----\n")
            for idx, exp_item in enumerate(expansions_list_uncertain):
                ff.write(f"Expansion {idx+1}: {exp_item}\n")

            ff.write("\n---- 5) Approximation metrics ----\n")
            for mk, mv in metrics_dict.items():
                ff.write(f"{mk}: {mv:.6f}\n")

        print(f"[INFO] Results saved to: {out_txt_path}")
        print("[INFO] Times:", times_dict)
        print("[INFO] Metrics:", metrics_dict)

        print("\n[INFO] Newton-Puiseux expansions for this point:")
        for e in expansions_list_uncertain:
            print("  ", e)

    print("\n[INFO] Completed the 4D (2 complex variables) analysis.")
