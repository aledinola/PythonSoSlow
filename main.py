# Solve neoclassical growth model via value function iteration
# using Brent's method for optimization from scipy.optimize
# and linear interpolation, from numpy

import numpy as np
from scipy.optimize import minimize_scalar, bisect
from types import SimpleNamespace
import matplotlib.pyplot as plt

# ---- parameters and grids ----

par = SimpleNamespace(
    beta  = 0.96,
    gamma = 2.0,
    alpha = 0.33,
    n_k   = 600
)


# ---- utility function ----
def f_util(c, gamma):
    # avoid log(0) or negative consumption
    c = np.maximum(c, 1e-10)
    if abs(gamma - 1.0) < 1e-8:
        util = np.log(c)
    else:
        util = c**(1.0 - gamma) / (1.0 - gamma)
    return util

# ---- RHS of Bellman equation ----
def fun_rhs(kprime, k, k_grid,V_array,par):
    """
    Returns minus the RHS of the Bellman equation
    for given k' and k.
    """
    beta  = par.beta
    gamma = par.gamma
    alpha = par.alpha

    # resource constraint: c = k^alpha - k'
    c = k**alpha - kprime

    # value at k' via interpolation
    v_interp = np.interp(kprime, k_grid, V_array)

    rhs = f_util(c, gamma) + beta * v_interp
    return -rhs

# ---- Value function iteration
k_ss = (par.alpha*par.beta)**(1.0/(1.0-par.alpha))
# Build capital grid around steady state
k_grid = np.linspace(0.1*k_ss,2*k_ss,par.n_k)
V0 = np.zeros(par.n_k)

tol = 1e-5
err = 1.0+tol

print("Start VFI....")
while err>tol:

    V1 = np.zeros(par.n_k)
    pol_kprime = np.zeros(par.n_k)
    for k_c in range(par.n_k):
        k_val = k_grid[k_c]
        # k'<=kprime_max makes sure consumption is positive 
        kprime_max = k_val**par.alpha
        objective = lambda kprime: fun_rhs(kprime, k_val,par,k_grid,V0)
        result = minimize_scalar(objective, bounds=(k_grid[1], kprime_max-1e-8), method='bounded')
        pol_kprime[k_c] = result.x 
        V1[k_c]         = -result.fun

    # Compute error
    err = max(abs(V1-V0))
    print(err)

    # Update
    V0 = V1

print("VFI done!")

# Plot results
plt.plot(k_grid, k_grid, label="45-degree line (k' = k)")
plt.plot(k_grid, pol_kprime, label="Policy function k'(k)")
plt.xlabel("Capital today k")
plt.ylabel("Capital tomorrow k'")
plt.legend()
plt.grid(True)

plt.show()
