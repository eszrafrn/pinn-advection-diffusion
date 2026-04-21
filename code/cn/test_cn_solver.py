from cn_solver import CNSolver
from initial_conditions import gaussian_pulse
from boundary_conditions import Neumann_BC
import numpy as np
import matplotlib.pyplot as plt
import plotting

# Setup parameters
L = 1.0
T = 1.0
Nx = 400
Nt = 400
v = 0.01
D = 0.1

# Initial condition
IC = lambda x: gaussian_pulse(x, A=1.0, x0=0.5, sigma=0.1)
BC = Neumann_BC(flux_left=0.0, flux_right=0.0)

solver = CNSolver(L, T, Nx, Nt, v, D, IC, BC)
c_final = solver.solve(save_history=True)

# Check for NaN values
assert not np.any(np.isnan(c_final)), "Error: Terdapat nilai NaN pada solusi!"
print("Check for NaN values: no NaN")
# Check for negative values
assert np.all(c_final >= -1e-10), "Error: Terdapat nilai negatif pada solusi!"
print("Check for negative values: all positives")
# Check for values greater than 1 (assuming concentration should be between 0 and 1)
assert np.all(c_final >= -1e-10) and np.all(c_final <= 1.0 + 1e-10), "Error: Magnitudo di luar batas fisik 0 < c < 1!"
print("Check solution magnitude reasonable: Physical range")


# plot solusi akhir
plotting.plot_solution(solver.x, c_final, title='Profil Solusi Akhir Crank-Nicolson', save_path='figures/CN_Final_Solution.png')

# plot evolusi solusi pada beberapa waktu sampel
indices = [0, Nt//4, Nt//2, 3*Nt//4, Nt]
t_samp = [solver.t[i] for i in indices]
c_samp = [solver.solution_history[i] for i in indices]
plotting.plot_solution_evolution(solver.x, t_samp, c_samp, save_path='figures/CN_Solution_Evolution.png')

# plot evolusi massa
plotting.plot_mass_vs_time(solver.t, solver.mass_history, save_path='figures/CN_Mass_vs_Time.png') 

