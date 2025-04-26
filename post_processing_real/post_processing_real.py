import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import shap
import sympy
import os

from lime.lime_tabular import LimeTabularExplainer

# It is assumed that the following functions are defined in separate modules.
# For clarity we import them as shown below:
from src.find_up_synthetic import SimpleComplexNet, complex_modulus_to_logits
from mit_bih_pre.pre_pro import load_mitbih_data, WINDOW_SIZE, PRE_SAMPLES, FS

from up.real.up_real import record_names, parse_args
from src.local_analysis import (
    local_poly_approx_complex,
    puiseux_uncertain_point,
    load_uncertain_points,
    evaluate_poly_approx_quality
)
from src.find_up_real import compress_to_C2
from src.post_processing import (
    interpret_puiseux_expansions,
    find_adversarial_directions,
    test_adversarial_impact,
    model_to_explain,
    compute_lime_explanation,
    compute_shap_explanation,
    plot_local_contour_2d,
    plot_robustness_curve,
    scalarize
)


########################################################################
# MAIN SCRIPT
########################################################################
if __name__ == "__main__":
    # Set the computation device: use GPU if available, otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # ----------------------------------------------------------------------
    # 1) Load the trained model from a file.
    # ----------------------------------------------------------------------
    # The model is an instance of SimpleComplexNet with specified input, hidden, and output sizes.
    model = SimpleComplexNet(
            in_features=2,
            hidden_features=64,    
            out_features=2,
            bias=0.1
    ).to(device)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'up_real', 'best_model_full.pt')
    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    # Load the saved parameters and update the model's state.
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    T_path = os.path.join(current_dir, '..', 'up_real', 'T_calib.pt')
    if os.path.isfile(T_path):
        T = torch.load(T_path, map_location=device)
        print(f"[INFO] Loaded temperature scaling T={T.item():.3f}")
    else:
        T = None
        print("[WARN] No temperature file – model will be uncalibrated")

    model.eval()
    print(f"[INFO] Loaded MIT‑BIH parameters {model_path}")

    # ----------------------------------------------------------------------
    # 2) Load uncertain points from a CSV file.
    # ----------------------------------------------------------------------
    csv_path = os.path.join(current_dir, '..', 'up_real', 'uncertain_full.csv')
    csv_path = os.path.abspath(csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV with uncertain points not found: {csv_path}")
    # The function load_uncertain_points loads and parses uncertain points from the CSV.
    up_list = load_uncertain_points(csv_path)
    print(f"[INFO] Loaded {len(up_list)} uncertain points from {csv_path}")

    # Additionally, load training data for LIME (used as background data).
    args   = parse_args()
    X_full, y_full = load_mitbih_data(
            args.data_folder,
            record_names,
            WINDOW_SIZE,
            PRE_SAMPLES,
            FS
    )
    # Additionally, load training data for LIME (used as background data).
    idx_bg = np.random.choice(len(X_full), size=512, replace=False)
    X_train_full = X_full[idx_bg]    
    X_train =  compress_to_C2(X_train_full) # shape (512, 4)

    # Define symbolic variables to be used in the analysis (e.g., for Puiseux expansions).
    x_sym, y_sym = sympy.symbols('x y')

    # Process each uncertain point in the loaded list.
    for i, up in enumerate(up_list):
        print(f"\n=== POINT # {i} ===")
        print("[DATA]", up)

        # Prepare the base point for local analysis.
        xstar = np.array(up['X'], dtype=np.float32)

        # ------------------------------------------------------------------
        # (A) Fit a local polynomial approximation around xstar.
        # ------------------------------------------------------------------
        # The function local_poly_approx_complex computes a polynomial approximation of the model's function
        # near the base point. The polynomial is "zeroed" at (0,0) via the remove_linear parameter.
        F_expr_zero = local_poly_approx_complex(
            model, xstar, delta=0.05, degree=4, n_samples=300, 
            device=device, remove_linear=True,
        )

        # Evaluate the approximation quality by comparing the polynomial with the model's output.
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

        # Add a warning if Pearson correlation is very low.
        if metrics['corr_pearson'] < 0.2:
            print("[WARNING] Low Pearson correlation detected (%.3f). "
                "The local approximation might be imprecise—this may indicate a highly nonlinear region. "
                "Consider increasing the number of samples or the polynomial degree." % metrics['corr_pearson'])

        # ------------------------------------------------------------------
        # (B) Compute Puiseux expansions and provide an interpretation.
        # ------------------------------------------------------------------
        # puiseux_uncertain_point computes the Puiseux series expansion near the uncertain point.
        expansions_np = puiseux_uncertain_point(F_expr_zero, prec=4, base_point=xstar)
        # Interpret the Puiseux expansions using the provided helper function.
        interpret_results = interpret_puiseux_expansions(expansions_np, x_sym, y_sym)

        print("\n[PUISEUX EXPANSIONS & INTERPRETATION]")
        for idx_e, ir in enumerate(interpret_results):
            print(f"  Expansion {idx_e}:")
            print("    ", ir["puiseux_expr"])
            print("    =>", ir["comment"])

        # ------------------------------------------------------------------
        # (C) Robustness Analysis Along Multiple Directions
        # ------------------------------------------------------------------
        # For this analysis, the same polynomial approximation is used.
        polynom = F_expr_zero

        # Identify promising adversarial directions by analyzing the phase of the polynomial.
        # Here, we sample 20 random directions and choose the top 3 where the phase is closest to ±pi.
        best_dirs_info = find_adversarial_directions(polynom, x_sym, y_sym, num_random=20, radius=0.01)
        print("\n[ROBUSTNESS] Checking top directions from the polynomial's phase analysis:")
        all_adv_curves=[]
        # Collect robustness results in a table.
        results_table = []
        for d_id, (dir_radians, phase_val) in enumerate(best_dirs_info):
            adv_list, changed_class, changed_radius = test_adversarial_impact(
                model, xstar, dir_radians, radius=0.02, steps=20, device=device
            )
            # Append the results for this direction.
            all_adv_curves.append(adv_list)
            results_table.append({
                "direction_id": d_id,
                "direction_radians": dir_radians,
                "phase": phase_val,
                "changed_class": changed_class,
                "changed_radius": changed_radius
            })

        # Print a formatted table of the adversarial direction analysis.
        print("DirID | (thx, thy)        | phase     | changed_class | changed_radius")
        for row in results_table:
            print("{:5d} | ({:.3f}, {:.3f}) | {:.3f}  | {}           | {}".format(
                row["direction_id"], 
                row["direction_radians"][0], row["direction_radians"][1],
                row["phase"],
                row["changed_class"],
                f"{row['changed_radius']:.4f}" if row["changed_radius"] is not None else None
            ))

        # Plot robustness curves (new visualization)
        robustness_plot_path = os.path.join(current_dir, f'robustness_curves_point{i}.png')
        plot_robustness_curve(all_adv_curves, save_path=robustness_plot_path)


        # ------------------------------------------------------------------
        # (D) Explanation Using SHAP and LIME
        # ------------------------------------------------------------------
        # Reshape the base point to be consistent with the expected input shape.
        xstar_reshaped = xstar.reshape(1, -1).astype(np.float32)

        # --- LIME Explanation ---
        lime_exp = compute_lime_explanation(model, X_train, xstar_reshaped[0], device)
        lime_list = lime_exp.as_list()
        print("\n[LIME Explanation]")
        for feat, val in lime_list:
            print(f" {feat}: {val:.3f}")
        
        # --- SHAP Explanation ---
        shap_vals = compute_shap_explanation(model, xstar_reshaped, device, background=X_train[:10], num_samples=100)
        # If shap_vals contains more than one array, extract the per-class SHAP values.
        if isinstance(shap_vals, list) and len(shap_vals) >= 2:
            shap_class0 = shap_vals[0][0]
            shap_class1 = shap_vals[1][0]
        else:
            if isinstance(shap_vals, list) and len(shap_vals) > 0:
                shap_class0 = shap_vals[0][0]
            else:
                shap_class0 = np.array(shap_vals)[0]
            shap_class1 = None
            print("[INFO] SHAP returned only one element; possibly model_to_explain returns a scalar value.")

        def scalarize(x):
            """Returns a scalar value extracted from a numpy array."""
            sx = np.squeeze(x)
            return sx.item() if np.isscalar(sx) or sx.size == 1 else float(sx.flat[0])

        print("\n[SHAP Explanation] per feature:")
        for fid, feat_name in enumerate(["Re(z1)", "Re(z2)", "Im(z1)", "Im(z2)"]):
            if shap_class1 is None:
                print("  {} => shap: {:.3f}".format(feat_name, scalarize(shap_class0[fid])))
            else:
                print("  {} => shap0: {:.3f}, shap1: {:.3f}".format(
                    feat_name, scalarize(shap_class0[fid]), scalarize(shap_class1[fid])
                ))

        # ------------------------------------------------------------------
        # (E) 2D Visualization of the Local Decision Contour
        # ------------------------------------------------------------------
        # Generate two contour plots by fixing different pairs of dimensions.
        # For example, fixing dimensions (1,3) [i.e., Re(z2), Im(z2)]
        # to visualize variations over (Re(z1), Im(z1)).
        save_dim_1_3 =os.path.join(current_dir, f'ontour_point{i}_fix_dim=[1,3].png')
        plot_local_contour_2d(model, xstar, fix_dims=(1, 3), delta=0.05, steps=50, device=device, save_path=save_dim_1_3)
        save_dim_0_2 =os.path.join(current_dir, f'ontour_point{i}_fix_dim=[0,2].png')
        plot_local_contour_2d(model, xstar, fix_dims=(0, 2), delta=0.05, steps=50, device=device, save_path=save_dim_0_2)

        
        # ------------------------------------------------------------------
        # (F) Save the Analysis Report to a File
        # ------------------------------------------------------------------
        # Save the comprehensive results of the analysis in a formatted text file.
        out_txt_path = os.path.join(current_dir, f'benchmark_point{i}.txt')
        with open(out_txt_path, "w") as f_out:
            f_out.write("=" * 80 + "\n")
            f_out.write(f"Local Analysis Report for Uncertain Point #{i}\n")
            f_out.write("=" * 80 + "\n\n")
            
            # Report the base point and the quality metrics of the approximation.
            f_out.write("1. Base Point (xstar):\n")
            f_out.write(f"   {xstar.tolist()}\n\n")
            f_out.write("2. Approximation Quality Metrics:\n")
            f_out.write(f"   RMSE             : {metrics['RMSE']:.3f}\n")
            f_out.write(f"   MAE              : {metrics['MAE']:.3f}\n")
            f_out.write(f"   Pearson Corr     : {metrics['corr_pearson']:.3f}\n")
            f_out.write(f"   Sign Agreement   : {metrics['sign_agreement']:.3f}\n\n")
            
            # Record the Puiseux expansions and their interpretations.
            f_out.write("3. Puiseux Expansions and Their Interpretation:\n")
            for idx_e, ir in enumerate(interpret_results):
                f_out.write(f"   >> Expansion {idx_e}:\n")
                f_out.write(f"      Puiseux Expression: {ir['puiseux_expr']}\n")
                f_out.write(f"      Interpretation  : {ir['comment']}\n")
            f_out.write("\n")
            
            # Output the robustness analysis results.
            f_out.write("4. Robustness Analysis Results:\n")
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
            
            # Write the LIME explanation results.
            f_out.write("5. LIME Explanation (Local Feature Importance):\n")
            for feat, val in lime_list:
                f_out.write(f"   {feat}: {val:.3f}\n")
            f_out.write("\n")
            
            # Write the SHAP explanation results.
            f_out.write("6. SHAP Explanation (Feature Contributions per Class):\n")
            for fid, feat_name in enumerate(["Re(z1)", "Re(z2)", "Im(z1)", "Im(z2)"]):
                if shap_class1 is None:
                    f_out.write(f"  {feat_name} => Class: {scalarize(shap_class0[fid]):.3f}\n")
                else:
                    f_out.write(f"  {feat_name} => Class 0: {scalarize(shap_class0[fid]):.3f}, Class 1: {scalarize(shap_class1[fid]):.3f}\n")
            
            f_out.write("\n" + "=" * 80 + "\n")
            f_out.write("End of Report\n")
            f_out.write("=" * 80 + "\n")

    print("\n[INFO] Completed analysis for all uncertain points.")
