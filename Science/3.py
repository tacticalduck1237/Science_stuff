import sympy as sp
import numpy as np

def nsolve_derivative(expr, x, guesses, tol=1e-6):
    solutions = []
    for guess in guesses:
        try:
            sol = sp.nsolve(expr, x, guess)
            sol_val = float(sol)
            # Add solution if it is not already found (within a tolerance)
            if not any(abs(sol_val - s) < tol for s in solutions):
                solutions.append(sol_val)
        except Exception:
            # Skip guesses that fail
            pass
    return solutions

def solve_derivatives_numerical(a_val, b_val, k_val):
    # Define the variable and parameters; assume x > 0 for this function
    x = sp.symbols('x', positive=True)
    a, b, k = sp.symbols('a b k', real=True, positive=True)
    
    # Define f(x) = a/(x^b * exp(k/x)) = a * x^(-b) * exp(-k/x)
    f = a * x**(-b) * sp.exp(-k/x)
    # Substitute the given values
    f_numeric = f.subs({a: a_val, b: b_val, k: k_val})
    
    # Compute the first, second, and third derivatives
    f1 = sp.simplify(sp.diff(f_numeric, x))
    f2 = sp.simplify(sp.diff(f1, x))
    f3 = sp.simplify(sp.diff(f2, x))
    
    print("f(x) =")
    sp.pprint(f_numeric)
    print("\nFirst derivative f'(x) =")
    sp.pprint(f1)
    print("\nSecond derivative f''(x) =")
    sp.pprint(f2)
    print("\nThird derivative f'''(x) =")
    sp.pprint(f3)
    
    # Define a range of initial guesses for x; adjust as needed
    guesses = np.linspace(0.1, 20, 200)
    
    sol_f1 = nsolve_derivative(f1, x, guesses)
    sol_f2 = nsolve_derivative(f2, x, guesses)
    sol_f3 = nsolve_derivative(f3, x, guesses)
    
    print("\nNumerical solutions for f'(x)=0:")
    print(sol_f1)
    print("\nNumerical solutions for f''(x)=0:")
    print(sol_f2)
    print("\nNumerical solutions for f'''(x)=0:")
    print(sol_f3)

# Example usage: Replace these with your known values for a, b, and k.
a_val = 8.61E-01  # Your value for a
b_val = -1.34E-01   # Your value for b
k_val = 1.29E+01   # Your value for k

solve_derivatives_numerical(a_val, b_val, k_val)