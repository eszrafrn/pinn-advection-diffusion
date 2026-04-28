import numpy as np
import torch

def compute_l2_relative_error(c_pred, c_ref):
    error = np.linalg.norm(c_pred - c_ref) / np.linalg.norm(c_ref)
    return error

def compute_mass(c_pred, dx):
    mass = np.sum(c_pred) * dx
    return mass

def mass_error_percent(current_mass, initial_mass):
    return abs((current_mass - initial_mass) / initial_mass) * 100

def evaluate_pinn(model, test_grid_x, t_val, c_ref, dx):
    model.eval()

    t_tensor = torch.full_like(test_grid_x, t_val)
    input_tensor = torch.cat([test_grid_x, t_tensor], dim=1)

    with torch.no_grad():
        c_pred = model(input_tensor).cpu().numpy()
    
    l2_error = compute_l2_relative_error(c_pred, c_ref)
    current_mass = compute_mass(c_pred, dx)
    return c_pred, l2_error, current_mass

'''
#TEST
if __name__ == '__main__':
    print("="*60)
    print("MENGUJI FUNGSI EVALUASI (L2 ERROR & MASS BALANCE)")
    print("="*60)
    
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from network import PINN_Network
    
    # 1. Setup Dummy Model & Grid
    torch.manual_seed(42)
    dummy_model = PINN_Network()
    
    Nx = 100
    L = 1.0
    dx = L / (Nx - 1)
    
    # Grid spasial (x) sebagai PyTorch Tensor
    x_np = np.linspace(0, L, Nx).reshape(-1, 1)
    test_grid_x = torch.tensor(x_np, dtype=torch.float32)
    
    t_test = 0.5 # Evaluasi pada detik ke 0.5
    
    # 2. Simulasi Data Referensi (Seolah-olah ini data Crank-Nicolson)
    # Kita asumsikan solusi referensi adalah fungsi Gaussian konstan
    c_ref_dummy = np.exp(-((x_np - 0.5)**2) / 0.05)
    
    # Hitung massa referensi awal
    initial_mass_ref = compute_mass(c_ref_dummy, dx)
    
    # 3. Eksekusi Evaluasi
    c_pred_dummy, l2_err, current_mass_pred = evaluate_pinn(
        dummy_model, test_grid_x, t_test, c_ref_dummy, dx
    )
    
    # 4. Tampilkan Hasil
    print(f"Evaluasi pada t = {t_test} detik:")
    print(f"-> Relative L2 Error      : {l2_err:.6e}")
    print(f"-> Massa Awal (Referensi) : {initial_mass_ref:.6e}")
    print(f"-> Massa PINN saat ini    : {current_mass_pred:.6e}")
    
    # Hitung Persentase Error Massa
    mass_error_percent = mass_error_percent(current_mass_pred, initial_mass_ref)
    print(f"-> Error Neraca Massa (%) : {mass_error_percent:.4e} %")
    
    print("\n-> Status: FUNGSI EVALUASI L2 & NERACA MASSA BERHASIL!")
'''