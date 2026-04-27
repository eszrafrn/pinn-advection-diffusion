# Menerapkan Latin Hypercube Sampling (LHS) untuk titik kolokasi (PDE residual) di dalam domain [0, L] x [0, T]
import numpy as np
import torch
from scipy.stats import qmc
import matplotlib.pyplot as plt
import os

def sample_collocation(N_r, L, T):
    sampler = qmc.LatinHypercube(d=2)
    sample = sampler.random(n=N_r)

    # skala sample ke domain [0, L] x [0, T]
    x_r = sample[:, 0] * L
    t_r = sample[:, 1] * T

    return x_r.reshape(-1, 1), t_r.reshape(-1, 1)

def sample_IC(N_IC, L):
    x_ic = np.random.uniform(0, L, N_IC)
    t_ic = np.zeros(N_IC)
    return x_ic.reshape(-1, 1), t_ic.reshape(-1, 1)

def sample_BC_Dirichlet(N_BC, T, L):
    x_bc = np.random.choice([0.0, L], N_BC)
    t_bc = np.random.uniform(0, T, N_BC)
    return x_bc.reshape(-1, 1), t_bc.reshape(-1, 1)

def sample_BC_Neumann(N_BC, T, L):
    x_bc = np.random.choice([0.0, L], N_BC)
    t_bc = np.random.uniform(0, T, N_BC)
    return x_bc.reshape(-1, 1), t_bc.reshape(-1, 1)

# ubah array numpy ke tensor PyTorch
def to_tensor(points, requires_grad=False, device='cpu'):
    tensor = torch.tensor(points, dtype=torch.float32, device=device)
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor

def prepare_training_data(N_r = 10000, N_IC = 5000, N_BC = 1600, L=1.0, T=0.5, bc_type='Dirichlet', device='cpu'):
    # sampling (dalam numpy array)
    x_r_np, t_r_np = sample_collocation(N_r, L, T)
    x_ic_np, t_ic_np = sample_IC(N_IC, L)
    if bc_type == 'Dirichlet':
        x_bc_np, t_bc_np = sample_BC_Dirichlet(N_BC, T, L)
    elif bc_type == 'Neumann':
        x_bc_np, t_bc_np = sample_BC_Neumann(N_BC, T, L)
    else:
        raise ValueError("Invalid BC type. Choose 'Dirichlet' or 'Neumann'.")
    
    # convert ke tensor PyTorch
    data = {
        'x_r': to_tensor(x_r_np, requires_grad=True, device=device),
        't_r': to_tensor(t_r_np, requires_grad=True, device=device),
        'x_ic': to_tensor(x_ic_np, device=device),
        't_ic': to_tensor(t_ic_np, device=device),
        'x_bc': to_tensor(x_bc_np, device=device),
        't_bc': to_tensor(t_bc_np, device=device)
    }

    return data

# TEST
L_test = 1.0
T_test = 0.5
data_test = prepare_training_data(N_r=1000, N_IC=500, N_BC=250, L=L_test, T=T_test, bc_type='Dirichlet')
print(f"Collocation Points (x_r, t_r): {data_test['x_r'].shape}, {data_test['t_r'].shape}")
print(f"Initial Condition Points (x_ic, t_ic): {data_test['x_ic'].shape}, {data_test['t_ic'].shape}")
print(f"Boundary Condition Points (x_bc, t_bc): {data_test['x_bc'].shape}, {data_test['t_bc'].shape}")

'''
# visualisasi
os.makedirs('figures', exist_ok=True)
plt.figure(figsize=(8, 6))
plt.scatter(data_test['t_r'].detach().cpu().numpy(), data_test['x_r'].detach().cpu().numpy(), s=10, label='Collocation Points', color = 'black')
plt.scatter(data_test['t_ic'].detach().cpu().numpy(), data_test['x_ic'].detach().cpu().numpy(), s=10, label='Initial Condition Points', color = 'blue', marker='x')
plt.scatter(data_test['t_bc'].detach().cpu().numpy(), data_test['x_bc'].detach().cpu().numpy(), s=10, label='Boundary Condition Points', color = 'red', marker='x')
plt.xlabel('Waktu (t)')
plt.ylabel('Posisi (x)')
plt.title('Distribusi Titik Sampling')
plt.legend()
plt.grid(True,alpha=0.5)

save_path = 'figures/sampling_distribution.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Distribusi titik sampling disimpan ke: {save_path}")
plt.show()

# test transfer ke GPU (jika tersedia)
if torch.cuda.is_available():
    gpu_tensor = to_tensor(data_test['x_r'], device='cuda')
    print(f"Tensor x_r di GPU: {gpu_tensor.device}")
else:
    print("GPU tidak tersedia, menggunakan CPU.")
    '''