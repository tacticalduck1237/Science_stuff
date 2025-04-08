import sympy as sp
import numpy as np

def filter_real(sol_list):
    """
    Filters a list of solutions, returning only those with a negligible imaginary part.
    """
    real_sol = []
    for sol in sol_list:
        sol_eval = sp.N(sol)
        if abs(sp.im(sol_eval)) < 1e-6:
            real_sol.append(sp.re(sol_eval))
    return real_sol

def scan_find_roots(f_expr, var, x_min, x_max, num_points, epsilon):
    """
    Scans the interval [x_min, x_max] using a fixed number of points for sign changes in f_expr,
    then uses sp.nsolve to locate a root when a sign change is detected.
    
    Parameters:
      f_expr   : sympy expression for the function to scan.
      var      : sympy symbol (typically x).
      x_min    : start of the interval.
      x_max    : end of the interval.
      num_points: number of sample points in the interval.
      epsilon  : minimum distance from zero to avoid singular points.
      
    Returns:
      List of numerical roots found.
    """
    # Generate sample points.
    x_values = np.linspace(x_min, x_max, num_points)
    f_num = sp.lambdify(var, f_expr, 'numpy')
    roots = []
    
    for i in range(len(x_values) - 1):
        x1 = x_values[i]
        x2 = x_values[i+1]
        # Skip points too close to zero
        if abs(x1) < epsilon or abs(x2) < epsilon:
            continue
        try:
            f1_val = float(f_num(x1))
            f2_val = float(f_num(x2))
        except Exception:
            continue
        if f1_val * f2_val < 0:
            try:
                # Use x1 as initial guess
                root = sp.nsolve(f_expr, var, x1)
                # Check for duplicates
                if not any(abs(root - r) < 1e-6 for r in roots):
                    roots.append(root)
                    print(f"Found root between {x1} and {x2}: {root}")
            except Exception:
                continue
    return roots

def get_derivative_roots(f_expr, var, derivative_expr, scan_interval=(-100, 100), num_points=1000, epsilon=1e-6):
    """
    Attempts to solve derivative_expr = 0 symbolically.
    If no solutions are found, scans numerically in scan_interval using num_points.
    Returns a list of real roots.
    """
    sols = sp.solve(derivative_expr, var, domain=sp.S.Reals)
    sols = filter_real(sols)
    if not sols:
        print("No symbolic solution; scanning numerically for roots...")
        sols = scan_find_roots(derivative_expr, var, scan_interval[0], scan_interval[1], num_points, epsilon)
    else:
        print("Symbolic solution found.")
    return [sp.N(r) for r in sols]

def compute_derivatives(A_val, B_val, K_val):
    """
    Defines f(x)=A/(x^B*exp(K/x)), computes its first three derivatives,
    and attempts to find zeros for each derivative.
    Prints the solutions for f'(x)=0, f''(x)=0, and f'''(x)=0.
    """
    x = sp.symbols('x', real=True)
    # Define f(x)
    f_expr = A_val / (x**B_val * sp.exp(K_val/x))
    
    # Compute derivatives.
    f1 = sp.diff(f_expr, x)
    f2 = sp.diff(f1, x)
    f3 = sp.diff(f2, x)
    
    # Optionally simplify f3
    f3_simpl = sp.simplify(f3)
    
    print("Simplified f'''(x):")
    sp.pprint(f3_simpl)
    print("\nSolving for roots...\n")
    
    sol_f1 = get_derivative_roots(f_expr, x, f1)
    sol_f2 = get_derivative_roots(f_expr, x, f2)
    sol_f3 = get_derivative_roots(f_expr, x, f3_simpl)
    
    print("\nResults:")
    print("Solutions for f'(x)=0:", sol_f1)
    print("Solutions for f''(x)=0:", sol_f2)
    print("Solutions for f'''(x)=0:", sol_f3)

if __name__ == "__main__":
    # Example usage; change these parameters as needed.
    compute_derivatives(0.976, -0.039, 9.32)
