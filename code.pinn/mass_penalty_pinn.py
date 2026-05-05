import torch
import numpy as np

# gaussian quadrature
def gaussian_quadrature(n=21, L=1.0, device='cpu'):
    # titik dan bobot untuk integrasi numerik
    # titik standar di rentang [-1, 1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(n)

    #transformasi titik ke interval [0, L]
    x_mapped = 0.5 * (x_gl + 1) * L
    w_mapped = 0.5 * w_gl * L
    
    x_tensor = torch.tensor(x_mapped, dtype=torch.float32, device=device).unsqueeze(1)  # (n, 1)
    w_tensor = torch.tensor(w_mapped, dtype=torch.float32, device=device).unsqueeze(1)  # (n, 1)
    return x_tensor, w_tensor

# array waktu yang tersebar merata dari 0 hingga T
def sample_time_points(n_t=21, T=0.5, device='cpu'):
    t = np.linspace(0, T, n_t)
    t_tensor = torch.tensor(t, dtype=torch.float32, device=device).unsqueeze(1)  # (n_t, 1)
    return t_tensor

def compute_mass_penalty(model, x_quad, w_quad, t_samples, M0):
    # menghitung loss massa total dengan integrasi numerik
    loss_mass = 0.0
    n_t = t_samples.shape[0]

    for i in range(n_t):
        t_current = t_samples[i].repeat(x_quad.shape[0], 1)  # (n_quad, 1)
        input = torch.cat([x_quad, t_current], dim=1)  # (n_quad, 2)
        c_pred = model(input)  # (n_quad, 1)

        # hitung massa total dengan integrasi numerik
        mass = torch.sum(c_pred * w_quad)
        loss_mass += (mass - M0) ** 2

    return loss_mass

def lambda_mass(epoch):
    # fungsi pembobotan untuk penalti massa berdasarkan epoch
    if epoch <= 100:
        return 10.0
    else:
        return 1.0   


    