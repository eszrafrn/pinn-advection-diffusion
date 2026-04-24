import numpy as np
import matplotlib.pyplot as plt
from cn_solver import CNSolver
from initial_conditions import gaussian_pulse
from boundary_conditions import Dirichlet_BC
from utils import compute_l2_error

# --- 1. SETUP PARAMETER AMAN (Agak gelombang tidak menabrak batas) ---
L = 1.0
T = 0.5       # Waktu dipersingkat agar gelombang murni di tengah
Nx = 1000     # Resolusi Sangat Halus (Ground Truth)
Nt = 1000
v = 0.001       # Kecepatan adveksi
D = 0.01      # Koefisien difusi
Pe = v * L / D  # Peclet number
# Parameter Initial Condition
A_init = 1.0
x0_init = 0.5 
sigma_init = 0.1

IC = lambda x: gaussian_pulse(x, A=A_init, x0=x0_init, sigma=sigma_init)
BC = Dirichlet_BC(left=0.0, right=0.0)

# --- 2. JALANKAN SOLUSI NUMERIK (CRANK-NICOLSON) ---
print("Menghitung Solusi Numerik (Crank-Nicolson Nx=1000)...")
solver = CNSolver(L, T, Nx, Nt, v, D, IC, BC)
c_numeric = solver.solve(save_history=False)
x = solver.x

# --- 3. HITUNG SOLUSI ANALITIK EKSAK ---
print("Menghitung Solusi Analitik Eksak...")
def exact_analytical_solution(x, t, v, D, A, x0, sigma):
    # Rumus pelebaran standar deviasi akibat difusi
    sigma_t = np.sqrt(sigma**2 + 2 * D * t)
    # Penurunan amplitudo untuk menjaga kekekalan massa
    A_t = A * (sigma / sigma_t)
    # Pergeseran posisi akibat adveksi
    posisi_puncak = x0 + v * t
    
    return A_t * np.exp(-((x - posisi_puncak)**2) / (2 * sigma_t**2))

c_analytic = exact_analytical_solution(x, T, v, D, A_init, x0_init, sigma_init)

# --- 4. HITUNG ERROR ANTARA NUMERIK VS ANALITIK ---
error_absolut = compute_l2_error(c_numeric, c_analytic)
print(f"L2 Error Numerik terhadap Analitik: {error_absolut:.4e}")

# --- 5. PLOTTING PEMBUKTIAN ---
import os
os.makedirs('figures', exist_ok=True)
plt.figure(figsize=(10, 6))

# Plot Solusi Analitik sebagai Garis Lurus (Solid Line)
plt.plot(x, c_analytic, color='magenta', linewidth=3, label='Solusi Analitik (Eksak)')

# Plot Solusi Numerik sebagai Titik-Titik (Scatter) agar terlihat tumpang tindihnya
# Kita potong datanya (step=25) agar titiknya tidak terlalu padat dan menutupi garis analitik
plt.scatter(x[::25], c_numeric[::25], color='navy', marker='o', s=40, zorder=5, 
            label='Solusi Numerik (CN Nx=1000)')

# Plot Kondisi Awal (T=0) sebagai referensi
c_awal = IC(x)
plt.plot(x, c_awal, color='gray', linestyle='--', label='Kondisi Awal (T=0)')

plt.xlabel('Posisi (x)')
plt.ylabel('Konsentrasi (c)')
plt.title(f'Pembuktian Kepresisian Metode Numerik (T={T}) [Pe={Pe}]')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()

plt.savefig('figures/Pembuktian_Analitik.png', dpi=300, bbox_inches='tight')
print("Plot pembuktian disimpan ke figures/Pembuktian_Analitik.png")
plt.show()