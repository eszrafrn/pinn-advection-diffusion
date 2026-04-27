import torch
import torch.nn as nn
from network import PINN_Network
def compute_residual(model, x, t, v, D):
    input = torch.cat([x,t], dim=1)  # gabungkan x dan t menjadi input (batch_size, 2)
    c_pred = model(input)  # prediksi konsentrasi dari PINN (batch_size, 1)

    # hitung turunan pertama (dc/dt) menggunakan autograd
    dc_dt = torch.autograd.grad(c_pred, t, grad_outputs=torch.ones_like(c_pred), create_graph=True, retain_graph=True)[0]

    # hitung turunan pertama (dc/dx) menggunakan autograd
    dc_dx = torch.autograd.grad(c_pred, x, grad_outputs=torch.ones_like(c_pred), create_graph=True, retain_graph=True)[0]

    # hitung turunan kedua (d^2c/dx^2) menggunakan autograd
    d2c_dx2 = torch.autograd.grad(dc_dx, x, grad_outputs=torch.ones_like(dc_dx), create_graph=True, retain_graph=True)[0]

    # hitung residual PDE
    residual = dc_dt + v*dc_dx - D*d2c_dx2

    return residual

def compute_bc_neumann(model, x, t):
    input = torch.cat([x,t], dim=1)
    c_pred = model(input)

    # hitung turunan pertama (dc/dx) menggunakan autograd
    dc_dx = torch.autograd.grad(c_pred, x, grad_outputs=torch.ones_like(c_pred), create_graph=True, retain_graph=True)[0]

    return dc_dx


'''
# test
from network import PINN_Network
if __name__ == '__main__':
    print("="*55)
    print("MENGUJI AUTOGRAD VS FINITE DIFFERENCE (FD)")
    print("="*55)
    
    torch.manual_seed(42)
    model = PINN_Network()
    
    model = model.double() 
    
    x_val, t_val = 0.5, 0.3
    x = torch.tensor([[x_val]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([[t_val]], dtype=torch.float64, requires_grad=True)
    v_test, D_test = 0.05, 0.01
    
    # --- AUTOGRAD ---
    c_ag = model(torch.cat([x, t], dim=1))
    
    dc_dt_ag = torch.autograd.grad(c_ag, t, grad_outputs=torch.ones_like(c_ag), create_graph=True)[0]
    dc_dx_ag = torch.autograd.grad(c_ag, x, grad_outputs=torch.ones_like(c_ag), create_graph=True)[0]
    d2c_dx2_ag = torch.autograd.grad(dc_dx_ag, x, grad_outputs=torch.ones_like(dc_dx_ag), create_graph=True)[0]
    
    val_dt_ag = dc_dt_ag.item()
    val_dx_ag = dc_dx_ag.item()
    val_dx2_ag = d2c_dx2_ag.item()
    
    # --- FINITE DIFFERENCE ---
    epsilon = 1e-4
    
    def get_c(x_in, t_in):
        return model(torch.cat([torch.tensor([[x_in]], dtype=torch.float64), 
                                torch.tensor([[t_in]], dtype=torch.float64)], dim=1)).item()

    c_center = c_ag.item()
    c_plus_x = get_c(x_val + epsilon, t_val)
    c_minus_x = get_c(x_val - epsilon, t_val)
    c_plus_t = get_c(x_val, t_val + epsilon)
    c_minus_t = get_c(x_val, t_val - epsilon)
    
    dc_dt_fd = (c_plus_t - c_minus_t) / (2 * epsilon)
    dc_dx_fd = (c_plus_x - c_minus_x) / (2 * epsilon)
    d2c_dx2_fd = (c_plus_x - 2 * c_center + c_minus_x) / (epsilon**2)
    
    # --- EVALUASI ---
    err_dt = abs(val_dt_ag - dc_dt_fd)
    err_dx = abs(val_dx_ag - dc_dx_fd)
    err_dx2 = abs(val_dx2_ag - d2c_dx2_fd)
    
    print(f"{'Turunan':<15} | {'Autograd':<12} | {'Finite Diff':<12} | {'Error (Selisih)':<15} | {'Status'}")
    print("-" * 75)
    
    print(f"{'dc/dt':<15} | {val_dt_ag:>12.6f} | {dc_dt_fd:>12.6f} | {err_dt:>15.2e} | {'PASSED' if err_dt < 1e-5 else 'FAILED'}")
    print(f"{'dc/dx':<15} | {val_dx_ag:>12.6f} | {dc_dx_fd:>12.6f} | {err_dx:>15.2e} | {'PASSED' if err_dx < 1e-5 else 'FAILED'}")
    print(f"{'d2c/dx2':<15} | {val_dx2_ag:>12.6f} | {d2c_dx2_fd:>12.6f} | {err_dx2:>15.2e} | {'PASSED' if err_dx2 < 1e-4 else 'FAILED'}")
    
    print("-" * 75)
    bc_grad = compute_bc_neumann(model, x, t).item()
    print(f"\nUji Boundary Gradient (Neumann) di x={x_val}, t={t_val}:")
    print(f"Nilai dc/dx dari fungsi batas : {bc_grad:.6f}")
'''