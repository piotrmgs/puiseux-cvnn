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
    Interprets Puiseux series expansions by extracting the dominant coefficients
    and providing heuristic comments on their significance. The input is typically
    a list of expansion expressions (e.g., "x**4(...)+ x**3(...) + ...").

    Parameters
    ----------
    expansions : list[str]
        A list of strings or sympy expressions representing the Puiseux expansions.
    x_sym : sympy.Symbol
        The symbol corresponding to the first variable (e.g., shifted z1).
    y_sym : sympy.Symbol
        The symbol corresponding to the second variable (e.g., shifted z2).

    Returns
    -------
    interpretation : list[dict]
        A list of dictionaries where each dictionary contains:
            - "puiseux_expr": the original Puiseux expansion (as a string),
            - "dominant_terms": a list of the top 3 terms (tuple of exponent tuple and coefficient),
            - "comment": a textual explanation noting the dominant coefficients.
            
    Notes
    -----
    The function expands the expression using sympy and collects individual terms.
    It then sorts the terms by the magnitude of their coefficients (using the absolute value).
    A heuristic commentary is built to indicate potential sensitivities (e.g., if a term like x^3 has
    a particularly large coefficient, it might suggest strong non-linearity in that region).
    """
    results = []
    for exp_str in expansions:
        # Convert the string to a sympy expression if it is not already.
        expr_sym = sympy.sympify(exp_str)

        # Expand the expression and convert it to a polynomial form with respect to x_sym and y_sym.
        poly_expr = sympy.expand(expr_sym)
        poly_expanded = sympy.Poly(poly_expr, x_sym, y_sym)

        # Retrieve all terms as a list of tuples: ((power_x, power_y), coefficient)
        terms = poly_expanded.terms()
        # Sort terms in descending order based on the absolute value of the coefficient.
        sorted_terms = sorted(terms, key=lambda t: abs(t[1]), reverse=True)
        top_terms = sorted_terms[:3]

        # Construct a commentary string that lists the dominant terms and their coefficients.
        commentary = "Dominant coefficients: " + ", ".join([
            f"{str(term[0])}: {float(sympy.N(sympy.re(term[1]))):.2e} {float(sympy.N(sympy.im(term[1]))):+.2e}i"
            for term in top_terms
        ])

        result_item = {
            "puiseux_expr": str(expr_sym),
            "dominant_terms": top_terms,
            "comment": commentary
        }
        results.append(result_item)
    
    return results


########################################################################
# 2. EXPANDED ROBUSTNESS ANALYSIS OVER MULTIPLE DIRECTIONS
########################################################################
def find_adversarial_directions(expansion_expr, x_sym, y_sym, num_random=10, radius=0.01):
    """
    Samples multiple random directions in the complex plane and evaluates an 
    expansion expression along those directions to identify adversarial perturbation paths.
    
    Parameters
    ----------
    expansion_expr : sympy.Expr
        A sympy expression that represents the expansion (e.g., a Puiseux series) to be evaluated.
    x_sym : sympy.Symbol
        The symbol corresponding to the first dimension.
    y_sym : sympy.Symbol
        The symbol corresponding to the second dimension.
    num_random : int, optional
        The number of random directional samples to generate (default is 10).
    radius : float, optional
        The magnitude (radius) for the perturbation in the complex plane (default is 0.01).
        
    Returns
    -------
    top_directions : list[tuple]
        A list of tuples containing:
            - the pair of randomly chosen angles (thx, thy) in radians,
            - the phase (angle) of the evaluated expression at the corresponding point.
            
    Details
    -------
    For each random direction, a complex perturbation is generated for both dimensions based on a
    randomly chosen phase. The expansion expression is evaluated at this point. We then sort the directions
    according to how close the phase is to ±pi, which may indicate potential sign changes (a heuristic to 
    identify adversarial behaviour).
    
    The function returns the top three directions (by default) with the phases closest to ±pi.
    """
    best_dirs = []
    for _ in range(num_random):
        thx = 2 * np.pi * random.random()
        thy = 2 * np.pi * random.random()
        # Create complex numbers with the specified radius and random phase.
        z1 = radius * np.exp(1j * thx)
        z2 = radius * np.exp(1j * thy)
        val = expansion_expr.subs({x_sym: z1, y_sym: z2})
        cval = complex(val.evalf())
        phase = np.angle(cval)
        best_dirs.append(((thx, thy), phase))
    
    # Sort the directions by how close their phase is to ±pi (i.e., potential sign reversal).
    best_dirs = sorted(best_dirs, key=lambda x: abs(abs(x[1]) - np.pi))
    
    # Return the top 3 adversarial directions.
    top_k = 3
    return best_dirs[:top_k]


def test_adversarial_impact(model, base_point, direction_radians, radius=0.01, steps=20, device='cpu'):
    """
    Tests the impact of an adversarial perturbation in a given direction on a model's prediction.

    Starting from a base input point, the function applies a perturbation that grows linearly
    (in a specified direction given by a pair of angles) and checks at what radius the model’s
    predicted class changes.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to be tested.
    base_point : np.ndarray
        The base input data point (assumed to be a 1D numpy array) from which to perturb.
    direction_radians : tuple(float, float)
        A tuple of two angles (thx, thy) in radians that define the direction of the perturbation.
    radius : float, optional
        The maximum radius for the perturbation (default is 0.01).
    steps : int, optional
        The number of intermediate steps in the perturbation (default is 20).
    device : str, optional
        The device to perform computation on (e.g., 'cpu' or 'cuda').
        
    Returns
    -------
    out_list : list[tuple]
        A list of tuples, each containing:
            (current radius, computed function value F, predicted class).
    changed_class : bool
        A boolean indicating whether the model's predicted class changed along the direction.
    changed_radius : float or None
        The radius at which the predicted class first changed, or None if no change occurred.

    Details
    -------
    The model's output is assumed to be composed of two halves corresponding to two classes.
    The function computes the difference in the aggregated logits (using a simple combination via
    the Euclidean norm) between the two classes along the perturbation path.
    """
    out_list = []
    thx, thy = direction_radians
    changed_class = False
    base_class = None
    changed_radius = None

    for i in range(steps + 1):
        r = (i / steps) * radius
        # Perturbation along the specified complex directions
        z1 = r * np.exp(1j * thx)
        z2 = r * np.exp(1j * thy)
        # Create a copy of the base point and apply real and imaginary perturbations 
        x_loc = base_point.copy()
        x_loc[0] += z1.real
        x_loc[2] += z1.imag
        x_loc[1] += z2.real
        x_loc[3] += z2.imag

        # Convert the modified point to a tensor and pass it through the model.
        x_ten = torch.tensor(x_loc, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            out_ = model(x_ten)
            half_ = out_.shape[1] // 2
            # Compute the modulus for the first and second halves.
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
def model_to_explain(x_numpy, model, device):
    """
    Converts a numpy array input into a model prediction suitable for explanation techniques.
    
    The function converts input data into a Torch tensor, performs a forward pass through the model,
    computes the Euclidean norm of the two halves of the output (corresponding to two classes), and
    returns a probability vector (after applying softmax).

    Parameters
    ----------
    x_numpy : np.ndarray
        The input data as a numpy array.
    model : torch.nn.Module
        The model which outputs logits for each class.
    device : str
        The computation device ('cpu' or 'cuda').

    Returns
    -------
    probs : np.ndarray
        The class probabilities computed from the logits (2D array: [batch_size, num_classes]).

    Notes
    -----
    This implementation ensures that the output is in a format compatible with SHAP and LIME,
    which generally expect a probability distribution over classes.
    """
    x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to(device)
    with torch.no_grad():
        out = model(x_tensor)
        half = out.shape[1] // 2
        # Compute Euclidean norm for both halves of the output.
        logits = torch.sqrt(out[:, :half]**2 + out[:, half:]**2 + 1e-9)
        # Convert logits to probabilities using softmax.
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


def compute_lime_explanation(model, training_data, point_np, device, class_names=["class0", "class1"], num_samples=500):
    """
    Computes an explanation for a given prediction using the LIME framework.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to be explained.
    training_data : np.ndarray
        Data to be used as the background (training) dataset for LIME.
    point_np : np.ndarray
        The data point (as a numpy array) for which the explanation is to be computed.
    device : str
        The computation device ('cpu' or 'cuda').
    class_names : list[str], optional
        Names of the classes (default are "class0" and "class1").
    num_samples : int, optional
        Number of samples to generate for LIME (default is 500).
        
    Returns
    -------
    exp : LIME explanation object
        The explanation generated by LIME. The explanation can be explored by calling exp.as_list()
        to retrieve feature contributions.
    """
    explainer = LimeTabularExplainer(
        training_data,
        feature_names=["Re(z1)", "Re(z2)", "Im(z1)", "Im(z2)"],
        class_names=class_names
    )
    
    def predict_fn(x):
        return model_to_explain(x, model, device)
    
    exp = explainer.explain_instance(point_np, predict_fn, num_samples=num_samples)
    return exp


def compute_shap_explanation(model, single_point_np, device, background=None, num_samples=100):
    """
    Computes a SHAP explanation for a given model prediction.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model for which the explanation is generated.
    single_point_np : np.ndarray
        A single data point as a numpy array of shape (1, 4).
    device : str
        The computation device ('cpu' or 'cuda').
    background : np.ndarray or None, optional
        The background dataset used for SHAP. If None, a zero array is used.
    num_samples : int, optional
        The number of samples to use for the Kernel SHAP estimation (default is 100).
        
    Returns
    -------
    shap_values : list or np.ndarray
        The SHAP values computed for each feature.
    
    Notes
    -----
    The SHAP KernelExplainer is used here with a background distribution. A zero baseline is used 
    if no background is provided.
    """
    if background is None:
        background = np.zeros((1, 4))
    
    explainer = shap.KernelExplainer(lambda x: model_to_explain(x, model, device), background)
    shap_values = explainer.shap_values(single_point_np, nsamples=num_samples)
    return shap_values


########################################################################
# 4. PLOTTING THE 2D LOCAL CONTOUR OF THE FUNCTION F = alpha0 - alpha1
########################################################################
def plot_local_contour_2d(model, xstar, fix_dims=(1, 3), delta=0.02, steps=50, device='cpu', save_path=None):
    """
    Plots a 2D contour of the function F = alpha0 - alpha1 around an uncertain data point.
    
    This function fixes two dimensions of the input data (given by fix_dims) and varies the remaining
    two dimensions (var_dims) to evaluate how the model's output changes. A contour plot is generated
    showing the level sets of F.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be used for evaluating the output function.
    xstar : np.ndarray
        The base input data point (a 1D numpy array with 4 elements).
    fix_dims : tuple of int, optional
        Indices of the dimensions to be held constant (default is (1, 3)). The other two dimensions
        will be varied.
    delta : float, optional
        The range of variation around the xstar value for the variable dimensions (default is 0.02).
    steps : int, optional
        The number of points per axis in the contour grid (default is 50).
    device : str, optional
        The computation device ('cpu' or 'cuda') used for inference.
    save_path : str or None, optional
        If provided, the plot will be saved to the given file path; otherwise, the plot is displayed.
        
    Raises
    ------
    ValueError
        If the number of variable dimensions is not equal to 2.
        
    Returns
    -------
    None

    Details
    -------
    The function creates a mesh grid for the two variable dimensions around the base point.
    It then computes the output F for each grid point by running the model. A contour plot is
    generated with a filled color map, a contour line for F=0 and the base point marked.
    """
    # Determine which dimensions will be varied.
    var_dims = [d for d in [0, 1, 2, 3] if d not in fix_dims]
    if len(var_dims) != 2:
        raise ValueError("plot_local_contour_2d requires exactly 2 variable dims.")
    
    x_index = var_dims[0]
    y_index = var_dims[1]

    # Clone xstar to maintain the constant dimensions.
    xstar_clone = xstar.copy()

    # Create a grid of points to vary the two dimensions.
    x_lin = np.linspace(xstar[x_index] - delta, xstar[x_index] + delta, steps)
    y_lin = np.linspace(xstar[y_index] - delta, xstar[y_index] + delta, steps)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = np.zeros_like(X)

    # Evaluate the model for each point in the grid.
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
            Z[i, j] = F_
    
    # Create a contour plot with filled color levels.
    plt.figure(figsize=(6, 5))
    cs = plt.contourf(X, Y, Z, levels=30, cmap='RdBu')
    plt.colorbar(cs)
    # Overlay a dashed contour line at the decision boundary F=0.
    plt.contour(X, Y, Z, levels=[0], colors='black', linestyles='--')
    # Mark the base point with a yellow dot.
    plt.scatter(xstar[x_index], xstar[y_index], color='yellow', edgecolor='k', s=100)
    plt.title(f"Local contour around uncertain point (variable dims={var_dims})")
    plt.xlabel(f"Dimension {x_index}")
    plt.ylabel(f"Dimension {y_index}")
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


###############################################
# 5. Plot robustness curves
###############################################

def plot_robustness_curve(all_adv_curves, save_path=None):
    """
    Plots the decision function value (F) versus perturbation radius for each adversarial direction.
    
    Parameters
    ----------
    all_adv_curves : list[list[tuple]]
        Each element is a list of tuples (radius, F, predicted_class) from test_adversarial_impact.
    save_path : str or None
        If provided, the plot is saved to the specified file path.
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
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Robustness curve saved to {save_path}")
    else:
        plt.show()

def scalarize(x):
    """
    Converts a numpy array containing a single element into its scalar value.

    Parameters
    ----------
    x : np.ndarray
        The input array that is expected to contain only a single element.

    Returns
    -------
    scalar : float or int
        The scalar value extracted from the array.
        
    Notes
    -----
    If the array is not inherently scalar, the function will attempt to extract the first element.
    """
    sx = np.squeeze(x)
    return sx.item() if np.isscalar(sx) or sx.size == 1 else float(sx.flat[0])

