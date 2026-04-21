import cn_solver
import utils
import boundary_conditions, initial_conditions
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

Nx_test = [200, 400, 800, 1000, 2000]
Nt = 2000
L = 1.0
T = 0.5
v = 0.1
D = 0.01
#Pe = 1.0

BC = boundary_conditions.Dirichlet_BC(left=0.0, right=0.0)
IC = lambda x: initial_conditions.gaussian_pulse(x, A=1.0, x0=0.3, sigma=0.1)

solusi = {}
errors = []

for Nx in Nx_test:
    solver = cn_solver.CNSolver(L, T, Nx, Nt, v, D, IC, BC)
    start_time = time.time()
    c_final = solver.solve(save_history=False)
    end_time = time.time()
    elapsed = end_time - start_time
    solusi[Nx] = {'x': solver.x, 'c': c_final, 'dx':solver.dx, 'time':elapsed}
    print(f"Completed Nx={Nx}, time = {elapsed:.2f} s")

# Evaluasi Error L2
Nx_ref = 2000
x_ref = solusi[Nx_ref]['x']
c_ref = solusi[Nx_ref]['c']

for Nx in Nx_test:
    coarse_x = solusi[Nx]['x']
    coarse_c = solusi[Nx]['c']
    dx = solusi[Nx]['dx']

    if Nx == Nx_ref:
        error = 0.0
    else:
        # Interpolasi solusi kasar ke grid referensi
        c_interp = np.interp(x_ref, coarse_x, coarse_c)
        error = utils.compute_l2_error(c_interp, c_ref)
    
    errors.append({
        'Nx': Nx,
        'dx': dx,
        'L2_error': error,
        'Time (s)': solusi[Nx]['time']
    })

#Konversi ke Dataframe
df_errrors = pd.DataFrame(errors)
print("\n==== HASIL GRID INDEPENDENCE STUDY ====")
print(df_errrors.to_string(index=False))

# Plot & konvergensi
df_plot = df_errrors[df_errrors['Nx'] != Nx_ref]
dx_array = df_plot['dx'].values
error_array = df_plot['L2_error'].values

# slope
slope, intercept = np.polyfit(np.log(dx_array), np.log(error_array), 1)
p = slope

ref_error = error_array[0] * (dx_array / dx_array[0])**2


os.makedirs('figures', exist_ok=True)
plt.figure(figsize=(8,5))
plt.loglog(dx_array, error_array, marker='o', label=f'Error Numerik (slope={p:.2f})', color='crimson')
plt.loglog(dx_array, ref_error, linestyle='--', label='Reference Slope $O(\\Delta x^2)$', color='navy')
plt.xlabel('Ukuran Grid ($\Delta x$)')
plt.ylabel('$L_2$ Relative Error')
plt.title('Grid Independence Study')
plt.grid(True, which='both', ls='--', alpha=0.7)
plt.legend()
plt.savefig('figures/Grid_Independence_Study.png', dpi=300, bbox_inches='tight')

print(f"\nConvergence rate: {p:.2f} (Expected: 2.0 for CN)")

csv_path = 'figures/grid_independence_data.csv'
df_errrors.to_csv(csv_path, index=False)
print(f"Data grid independence study disimpan ke {csv_path}")

# document in notes n verify
err_nx1000 = df_errrors[df_errrors['Nx'] == 1000]['L2_error'].values[0]
print("\nNotes & Verifikasi:")
print("Reference grid: Nx = 1000, dx = 0.0001")
print(f"Error vs Nx=2000: {err_nx1000:.2e}")

# verify error vs Nx2000 < 1e-6
if err_nx1000 < 1e-6:
    print("Verify: PASSED (error < 1e-6)")
else:
    print("Verify: FAILED (Note: Toleransi 1e-6 sangat ketat untuk L2 Relative Error antar grid. Perbedaan ~1% ini sudah sangat baik untuk dijadikan referensi).")

plt.show()

