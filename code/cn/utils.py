## Fungsi Utilitas untuk Simulasi Numerik ##
### sebagai fungsi untuk membuat grid, perhitungan massa, dan kesalahan metrik.

import numpy as np
#from scipy.integrate import simpson

def create_grid(L, Nx):
    return np.linspace(0, L, Nx)

def compute_mass(c, dx):
    #return simpson(y=c, dx=dx)
    return np.trapezoid(c, dx=dx)

def compute_l2_error(c_numeric, c_ref):
    num = np.sqrt(np.sum((c_numeric - c_ref) ** 2))
    denom = np.sqrt(np.sum(c_ref ** 2))

    if denom < 1e-15:  # Hindari pembagian dengan nol
        return 0.0 if num < 1e-15 else np.inf
    
    return num / denom

def compute_linf_error(c_numeric, c_ref): #kalau perlu
    diff_max = np.max(np.abs(c_numeric - c_ref))
    ref_max = np.max(np.abs(c_ref))
    
    if ref_max < 1e-15:  # Hindari pembagian dengan nol
        return 0.0 if diff_max < 1e-15 else np.inf
    
    return diff_max / ref_max

def compute_mass_error(Mt, M0):
    return np.abs(Mt - M0) / np.abs(M0)

