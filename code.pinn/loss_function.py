import torch
import torch.nn as nn

from network import PINN_Network
from pde_residual import compute_residual, compute_bc_neumann

# MSE PyTorch
mse_loss = nn.MSELoss()

# fungsi loss untuk initial conditions (IC)
def loss_ic(model, x_ic, t_ic, c_ic_true):
    c_pred = model(torch.cat([x_ic, t_ic], dim=1))
    loss = mse_loss(c_pred, c_ic_true)
    return loss

# fungsi loss untuk boundary conditions (BC)
def loss_bc_dirichlet(model, x_bc, t_bc, c_bc_true):
    c_pred = model(torch.cat([x_bc, t_bc], dim=1))
    loss = mse_loss(c_pred, c_bc_true)
    return loss

def loss_bc_neumann(model, x_bc, t_bc):
    dc_dx_pred = compute_bc_neumann(model, x_bc, t_bc)
    target_grad = torch.zeros_like(dc_dx_pred)  
    loss = mse_loss(dc_dx_pred, target_grad)
    return loss

# fungsi loss untuk PDE residual
def loss_pde(model, x_r, t_r, v, D):
    residual = compute_residual(model, x_r, t_r, v, D)
    target_residual = torch.zeros_like(residual)  
    loss = mse_loss(residual, target_residual)
    return loss

# fungsi total loss (gabungan IC, BC, dan PDE)
def total_loss_vanilla(model, data, params):
    # ekstrak data
    x_ic, t_ic = data['x_ic'], data['t_ic']
    x_bc, t_bc = data['x_bc'], data['t_bc']
    x_r, t_r = data['x_r'], data['t_r']
     
    # ekstrak parameter dan ground truth
    v, D = params['v'], params['D']
    bc_type = params.get('bc_type', 'Dirichlet')
    c_ic_true = params['c_ic_true']
    c_bc_true = params.get('c_bc_true', None)  # Hanya untuk Dirichlet

    # ekstrak pembobotan  (set default: IC = 100, BC = 10, PDE = 1)
    lambda_ic = params.get('lambda_ic', 100.0)
    lambda_bc = params.get('lambda_bc', 10.0)
    lambda_pde = params.get('lambda_pde', 1.0)

    # hitung masing-masing komponen loss
    L_IC = loss_ic(model, x_ic, t_ic, c_ic_true)
    L_PDE = loss_pde(model, x_r, t_r, v, D)
    if bc_type == 'Dirichlet':
        L_BC = loss_bc_dirichlet(model, x_bc, t_bc, c_bc_true)
    elif bc_type == 'Neumann':
        L_BC = loss_bc_neumann(model, x_bc, t_bc)
    else:
        raise ValueError("Invalid BC type. Choose 'Dirichlet' or 'Neumann'.")
    
    # hitung total loss
    total_loss = lambda_ic * L_IC + lambda_bc * L_BC + lambda_pde * L_PDE
    return total_loss, L_IC, L_BC, L_PDE

# TEST
if __name__ == '__main__':
    print("="*60)
    print("MENGUJI FUNGSI LOSS PINN (VANILLA)")
    print("="*60)
    
    # 1. Setup Dummy Data
    torch.manual_seed(42)
    model = PINN_Network()
    
    # Dummy tensors (requires_grad=True untuk PDE/Neumann)
    x_dummy = torch.rand(10, 1, requires_grad=True)
    t_dummy = torch.rand(10, 1, requires_grad=True)
    c_true_dummy = torch.zeros(10, 1) # Target dummy = 0.0
    
    # 2. Uji Komponen Loss Individu
    l_ic = loss_ic(model, x_dummy, t_dummy, c_true_dummy)
    print(f"IC Loss (Scalar)        : {l_ic.item():.6f}")
    
    l_bc_dir = loss_bc_dirichlet(model, x_dummy, t_dummy, c_true_dummy)
    print(f"BC Dirichlet Loss       : {l_bc_dir.item():.6f}")
    
    l_bc_neu = loss_bc_neumann(model, x_dummy, t_dummy)
    print(f"BC Neumann Loss         : {l_bc_neu.item():.6f}")
    
    l_pde = loss_pde(model, x_dummy, t_dummy, v=0.05, D=0.01)
    print(f"PDE Residual Loss       : {l_pde.item():.6f}")
    
    # 3. Uji Total Loss Function
    dummy_data = {
        'x_ic': x_dummy, 't_ic': t_dummy,
        'x_bc': x_dummy, 't_bc': t_dummy,
        'x_r': x_dummy,  't_r': t_dummy
    }
    
    dummy_params = {
        'v': 0.05, 'D': 0.01, 
        'bc_type': 'Neumann',
        'c_ic_true': c_true_dummy,
        'lambda_ic': 100.0, 'lambda_bc': 10.0, 'lambda_pde': 1.0
    }
    
    total_loss, L_IC, L_BC, L_PDE = total_loss_vanilla(model, dummy_data, dummy_params)
    print("\n--- KOMPUTASI TOTAL LOSS ---")
    print(f"L_IC  * {dummy_params['lambda_ic']:>5.1f} = {L_IC.item() * dummy_params['lambda_ic']:.6f}")
    print(f"L_BC  * {dummy_params['lambda_bc']:>5.1f} = {L_BC.item() * dummy_params['lambda_bc']:.6f}")
    print(f"L_PDE * {dummy_params['lambda_pde']:>5.1f} = {L_PDE.item() * dummy_params['lambda_pde']:.6f}")
    print("-" * 35)
    print(f"TOTAL LOSS (Vanilla)  = {total_loss.item():.6f}")
    
    if total_loss.dim() == 0:
        print("\n-> Status: KOMPUTASI FUNGSI LOSS BERHASIL DAN SKALAR!")