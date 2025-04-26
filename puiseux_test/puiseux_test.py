import sympy
import os
from sympy import nroots, symbols, expand, nsimplify, I

from src.puiseux import cross
from src.puiseux import lower_convex_hull
from src.puiseux import get_support
from src.puiseux import initial_branches
from src.puiseux import puiseux_expansions
   

if __name__ == '__main__':
    # Define the symbols
    x, y = sympy.symbols('x y')
    I = sympy.I

    # Example polynomial f(x, y). Modify f as needed.
    f = (y**3) * (y - x**2 - x**4 + x**5) * (y - x**2 - x**3 - x**4 - x**5)**2
    # Alternative examples (uncomment one to test):
    # f = y**6 - 3*x*y**4 + 3*x**2*y**2 - x**3 - 2*x*y**5 + 4*x**2*y**3 - 2*x**3*y + 8*x**5
    # f = y*(y - x**2 - x**3)*(y - x**3)
    # f = x**5 - 2.3324234*I*y**3
    #f=x**4*(31.8678326856002 - 9.63632020093375*I) + x**3*y*(26.5922869736171 + 56.6595267593413*I) + x**3*(2.95837145830588 + 19.7064951036128*I) + x**2*y**2*(-42.6726264798206 + 81.914967246129*I) + x**2*y*(7.48183521954347 - 6.25333431849867*I) + x**2*(-0.857786312915825 + 0.242658197527829*I) - x*y**3*(95.8955483696551 + 54.3438865342842*I) + x*y**2*(1.35669471910996 - 1.39669404709715*I) - x*y*(0.26489001722476 + 0.237768731578301*I) + y**4*(1.97703048522467 - 38.2008327802488*I) + y**3*(11.6432159624287 - 11.1131255841797*I) + y**2*(0.198694450266951 - 0.186767416998532*I)


    # Generate Puiseux expansions (default: 5 terms).
    exps = puiseux_expansions(f, x, y, max_terms=5)
    
    # Display the expansions in the console with a numeric approximation.
    print("Below are the Puiseux expansions (using 'x' as the variable):")
    for exp in exps:
        print(exp)
        #print(expand(exp).evalf(chop=True))
        print("--------------------------------------------------")

    # Define the output directory where the results will be saved.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "puiseux_expansions.txt")
    

    # Write the results to a text file with improved formatting.
    with open(file_path, "w") as outfile:
        header = "=" * 60 + "\n"
        header += "PUISEUX SERIES EXPANSIONS\n"
        header += "=" * 60 + "\n\n"
        outfile.write(header)
        
        outfile.write(f"The computed Puiseux expansions for f(x, y)={f} are:\n\n")
        for idx, exp in enumerate(exps, start=1):
            outfile.write(f"Expansion {idx}:\n")
            outfile.write(f"{expand(exp).evalf(chop=True)}\n")
            outfile.write("-" * 60 + "\n")
        
        footer = "\n" + "=" * 60 + "\n"
        footer += "End of Puiseux Series Expansions\n"
        footer += "=" * 60 + "\n"
        outfile.write(footer)

    print(f"Puiseux expansions have been saved to: {file_path}")
