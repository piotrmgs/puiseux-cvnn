# Copyright (c) 2025 Piotr Migus
# This code is licensed under the MIT License.
# See the LICENSE file in the repository root for full license information.
"""
Description:
    This module computes Puiseux series expansions for algebraic functions defined by a polynomial f(x, y).
    Puiseux series are generalized power series that allow fractional exponents and are useful in studying
    singular points in algebraic geometry. The module leverages symbolic computation via sympy to perform
    the expansion.

Functions:
    cross(o, a, b):
        Computes the 2D cross product (determinant) of vectors defined by the points o, a, and b.
        This is used for determining the orientation of three points and is fundamental in computing the convex hull.

    lower_convex_hull(points):
        Computes the lower convex hull (or lower envelope) of a set of 2D points.
        The lower convex hull is used to extract the contributing monomials from the polynomial's support
        for determining the initial series branch.

    get_support(expr, x, y):
        Extracts the support (the exponent set) of a bivariate polynomial expression.
        The support is returned as a list of tuples (i, j, coeff) representing monomials x^i * y^j with their coefficients.

    initial_branches(f, x, y, deg=0):
        Determines the initial Puiseux branches of the polynomial f(x, y) by analyzing its support.
        It computes the lower convex hull of the exponent points and then extracts candidate series branches
        in the form u*x^(theta). These candidates serve as starting points for the series expansion.

    puiseux_expansions(f, x, y, max_terms=5):
        Iteratively computes the Puiseux series expansions of f(x, y) up to a specified maximum number
        of terms. At each iteration, the method refines the current expansion by substituting the previous
        approximation, then extracting and appending the next term in the series.
        
Usage Example:
    1. Define the symbols x and y using sympy.symbols.
    2. Set up a polynomial f(x, y).
    3. Call puiseux_expansions(f, x, y, max_terms) to generate the series.
    4. Display the results and save them to a text file with attractive formatting.

Output File:
    The generated file "puiseux_expansions.txt" begins with a decorative header, lists the numbered expansions
    each separated by a horizontal line, and ends with a footer.
"""

import sympy
import os
from sympy import nroots, symbols, expand, nsimplify, I

def cross(o, a, b):
    """
    Compute the 2D cross product (determinant) of the vectors OA and OB.

    Parameters:
        o, a, b : tuple
            Points in the 2D plane, each represented as a tuple (x, y).

    Returns:
        The scalar cross product which is proportional to the signed area of the parallelogram formed by OA and OB.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def lower_convex_hull(points):
    """
    Compute the lower convex hull (lower envelope) from a given set of 2D points.

    This function sorts the points and uses a variation of the Andrew's monotone chain algorithm
    to determine the lower convex hull.

    Parameters:
        points : list of tuples
            A list of (x, y) coordinates.

    Returns:
        A list of points forming the lower convex hull.
    """
    pts = sorted(dict.fromkeys(points).keys(), key=lambda p: (p[0], p[1]))
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    return lower

def get_support(expr, x, y):
    """
    Extract the support of a bivariate polynomial expression.

    The support is defined as the set of monomials with their exponents and coefficients,
    i.e., a list of tuples (i, j, coeff) corresponding to monomials x^i * y^j.

    Parameters:
        expr : sympy expression
            The polynomial expression in variables x and y.
        x, y : sympy.Symbol
            The symbols used in the polynomial.

    Returns:
        List of tuples (exp_x, exp_y, coeff) representing the support.
    """
    expr = expand(expr)
    terms = expr.as_ordered_terms()
    support = []
    for term in terms:
        d = term.as_powers_dict()
        exp_x = d.get(x, 0)
        exp_y = d.get(y, 0)
        coeff = nsimplify(term / (x ** exp_x * y ** exp_y))
        support.append((nsimplify(exp_x), nsimplify(exp_y), coeff))
    return support

def initial_branches(f, x, y, deg=0):
    """
    Determine the initial Puiseux branches for the polynomial f(x,y).

    This function computes the lower convex hull of the support of f and identifies edges
    corresponding to potential leading terms in the Puiseux expansion. For each edge, an edge
    polynomial is formed and solved to obtain a candidate coefficient 'u' and exponent 'theta' 
    (the Puiseux exponent).

    Parameters:
        f : sympy expression
            The polynomial f(x, y).
        x, y : sympy.Symbol
            The symbols appearing in the polynomial.
        deg : int or sympy expression, optional
            A degree threshold to filter the branches (default is 0).

    Returns:
        List of tuples (u_val, theta) where u_val is the branch coefficient and theta is the exponent.
    """
    support = get_support(f, x, y)
    pts = [(pt[0], pt[1]) for pt in support]
    pts = sorted(dict.fromkeys(pts).keys(), key=lambda p: (p[0], p[1]))
    hull = lower_convex_hull(pts)
    branches = []
    for i in range(len(hull) - 1):
        p_point = hull[i]
        q_point = hull[i + 1]
        dx = q_point[0] - p_point[0]
        dy = q_point[1] - p_point[1]
        if dy == 0:
            continue  # Skip horizontal edges, the branch y = 0 is handled separately.
        # Compute the normal vector n = (-dy, dx)
        n1 = -dy
        n2 = dx
        theta = nsimplify(n2 / n1)
        if theta <= deg:
            continue
        else:
            d_val = nsimplify(n1 * p_point[0] + n2 * p_point[1])
            edge_terms = [(i_, j_, coeff)
                          for (i_, j_, coeff) in support
                          if nsimplify(n1 * i_ + n2 * j_) == d_val]
            u = sympy.symbols('u', complex=True)
            poly_expr = sum(coeff * u ** j_ for (i_, j_, coeff) in edge_terms)
            try:
                sols = sympy.nroots(poly_expr)
            except:
                sols = sympy.solve(poly_expr, u)
            for sol in sols:
                if sol != 0:
                    branches.append((sol, theta))
    return branches

def puiseux_expansions(f, x, y, max_terms=5):
    """
    Compute the Puiseux series expansions of f(x,y) up to max_terms.

    The expansion is built iteratively by computing an initial approximation (the first term)
    and then refining it by substituting the current approximation into the polynomial and
    extracting further terms.

    Parameters:
        f : sympy expression
            The polynomial f(x, y) for which the series is computed.
        x, y : sympy.Symbol
            The symbols representing the variables.
        max_terms : int, optional
            The maximum number of terms to compute in the Puiseux expansion (default is 5).

    Returns:
        A list of unique Puiseux series expansions.
    """
    num_terms = 1
    expansions = []
    expansions_fin = []
    while num_terms <= max_terms:
        if num_terms == 1:
            if nsimplify(f.subs(y, 0)) == 0:
                expansions_fin.append(0)
            branches = initial_branches(f, x, y)
            for (u_val, theta) in branches:
                expansions.append([u_val * x**(theta), theta, u_val * x**(theta), f])
            if max_terms == 1:
                return [row[0] for row in expansions] + expansions_fin
            else:
                num_terms += 1
        else:
            expansions_hat = []
            for exp in expansions:
                # Substitute the current approximation into f to get the residual
                f1 = nsimplify(expand(exp[3].subs(y, y + exp[2])), tolerance=1e-5)
                if nsimplify(f1.subs(y, 0)) == 0:
                    expansions_fin.append(exp[0])
                branches = initial_branches(f1, x, y, exp[1])
                if not branches:
                    expansions_hat.append(exp)
                else:
                    for (u_val, theta) in branches:
                        new_term = nsimplify(exp[0] + u_val * x**(theta), tolerance=1e-14).evalf()
                        expansions_hat.append([new_term, theta, u_val * x**(theta), f1])
            if max_terms == num_terms:
                expressions_unique = []
                for item in [row[0] for row in expansions_hat] + expansions_fin:
                    if item not in expressions_unique:
                        expressions_unique.append(item)
                return expressions_unique
            else:
                expansions = expansions_hat
                num_terms += 1

