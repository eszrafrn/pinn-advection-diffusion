# Test the CN matrices and solver
import numpy as np
from cn_matrices import build_CN_matrices, solve_tridiagonal

Nx=10
dx=0.1
dt=0.01
u=1.0
D=0.01

A, B = build_CN_matrices(Nx, dt, dx, u, D, bc_type='Dirichlet')
assert A.shape == (Nx, Nx)
print(A[0:3, 0:3])
solve = solve_tridiagonal(A.diagonal(-1), A.diagonal(0), A.diagonal(1), B[:, 0])
print(solve)
solve_compare = np.linalg.solve(A, B[:, 0])
print(solve_compare)