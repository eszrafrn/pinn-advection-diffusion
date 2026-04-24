import os
import time
import numpy as np
from cn_solver import CNSolver
from initial_conditions import gaussian_pulse
from boundary_conditions import Dirichlet_BC, Neumann_BC
from io_utils import save_solution
from utils import compute_mass, compute_mass_error
from plotting import plot_solution_evolution, plot_mass_vs_time

# SETUP PARAMETER
L = 1.0
T = 0.5
Nx = 1000
Nt = 1000
D = 0.01

Peclet = [0.25, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0]
bc_dict = {
    'Dirichlet': Dirichlet_BC(left=0.0, right=0.0),
    'Neumann': Neumann_BC(flux_left=0.0, flux_right=0.0)
}

IC = lambda x: gaussian_pulse(x, A=1.0, x0=0.5, sigma=0.1)

save_dir = 'data/reference'
os.makedirs(save_dir, exist_ok=True)

# LOOPING UNTUK SEMUA KOMBINASI PÉCLET DAN BOUNDARY CONDITIONS
total_cases = 0

for bc_name, bc_obj in bc_dict.items():
    for pe in Peclet:
        #D = v*L/pe
        v = D*pe/L 
        print(f"\n[{bc_name.upper()}] Running Pe = {pe} (v = {v}, D = {D})")

        start = time.time()
        solver = CNSolver(L, T, Nx, Nt, v, D, IC, bc_obj)
        c_final = solver.solve(save_history=True)
        elapsed = time.time() - start

        # Cek Instabilitas
        if np.any(np.isnan(c_final)):
            print("Warning: Solusi mengandung NaN, terjadi instabilitas!")
            continue

        # (Neumann) cetak info evaluasi massa
        if bc_name == 'Neumann':
            mass_initial = compute_mass(solver.IC(solver.x), solver.dx)
            mass_final = compute_mass(c_final, solver.dx)
            mass_error = compute_mass_error(mass_initial, mass_final)
            print(f"Mass Conservation Error: {mass_error:.4e}")
        
        # Simpan
        c_exact = np.array(solver.solution_history)
        filepath = os.path.join(save_dir, f"reference_{bc_name}_Pe={pe:.2f}.npz")
        params = {'v': v, 'D': D, 'Peclet': pe, 'L': L, 'T': T, 'Nx': Nx, 'Nt': Nt, 'BC': bc_name}
        save_solution(filepath, solver.x, solver.t, c_exact, params)

        print(f"Eksekusi dalam {elapsed:.2f} s")
        total_cases += 1
        x = solver.x
        t = solver.t
        c = np.array(solver.solution_history)
        plot_solution_evolution(x, t, c, save_path=f'figures/reference_{bc_name}_Pe={pe:.2f}_evolution.png', title=f'Evolusi Solusi CN ({bc_name} Pe={pe:.2f})')
        if bc_name == 'Neumann':
            plot_mass_vs_time(t, solver.mass_history, save_path=f'figures/reference_{bc_name}_Pe={pe:.2f}_mass.png', title=f'Evolusi Massa CN ({bc_name} Pe={pe:.2f})')
        else:
            continue
print("\n" + "="*50)
print(f"SUMMARY: ALL {total_cases}/14 REFERENCE SOLUTIONS COMPLETE!")




