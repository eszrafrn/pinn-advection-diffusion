## Kondisi Awal untuk Persamaan Adveksi-Difusi ##
import numpy as np

def gaussian_pulse(x, A=1.0, x0= 0.5, sigma=0.1):
    """
    Fungsi untuk menghasilkan pulsa Gaussian sebagai kondisi awal.
    
    Parameters:
    x : array_like
        Array posisi di mana pulsa Gaussian dihitung.
    A : float, optional
        Amplitudo pulsa Gaussian (default: 1.0).
    x0 : float, optional
        Posisi pusat pulsa Gaussian (default: 0.0).
    sigma : float, optional
        Lebar standar dari pulsa Gaussian (default: 0.1).
    
    Returns:
    array_like
        Nilai pulsa Gaussian pada posisi x.
    """
    return A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
