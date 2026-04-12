from cn_solver import CNSolver
from initial_conditions import gaussian_pulse
from boundary_conditions import Dirichlet_BC

# Setup parameters
L = 1.0
T = 0.5
Nx = 100
Nt = 100
v = 1.0
D = 0.01

# Initial condition
IC = lambda x: gaussian_pulse(x, A=1.0, x0=0.5, sigma=0.1)
BC = Dirichlet_BC(left=0.0, right=0.0)

solver = CNSolver(L, T, Nx, Nt, v, D, IC, BC)
c_final = solver.solve()

