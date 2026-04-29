import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def compute_l2_relative_error(c_pred, c_ref):
    error = np.linalg.norm(c_pred - c_ref) / np.linalg.norm(c_ref)
    return error

def compute_mass(c_pred, dx):
    mass = np.sum(c_pred) * dx
    return mass

def mass_error(current_mass, initial_mass):
    return abs((current_mass - initial_mass) / initial_mass)

def evaluate_pinn(model, test_grid_x, t_val, c_ref, dx):
    model.eval()

    t_tensor = torch.full_like(test_grid_x, t_val)
    input_tensor = torch.cat([test_grid_x, t_tensor], dim=1)

    with torch.no_grad():
        c_pred = model(input_tensor).cpu().numpy()
    
    l2_error = compute_l2_relative_error(c_pred, c_ref)
    current_mass = compute_mass(c_pred, dx)
    return c_pred, l2_error, current_mass

def plot_solution_evolution(x, t_array, c_array, save_path=None, title='Evolusi Solusi'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.05)
    
    # Cari nilai batas Y (maksimal dan minimal) agar skala ketiga grafik seragam
    y_min = min([np.min(c) for c in c_array])
    y_max = max([np.max(c) for c in c_array])
    margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
    
    # Loop untuk menggambar di masing-masing kotak (ax)
    for i, (t, c) in enumerate(zip(t_array, c_array)):
        ax = axes[i]
        ax.plot(x, c, color='indigo', linewidth=2.5)
        ax.set_title(f'Waktu (t) = {t:.2f} s', fontsize=12)
        ax.set_xlabel('Posisi (x)', fontsize=11)
        
        # Label Y hanya ditampilkan di grafik paling kiri agar lebih rapi
        if i == 0:
            ax.set_ylabel('Konsentrasi (c)', fontsize=11)
            
        ax.set_ylim(y_min - margin, y_max + margin) # Kunci skala Y agar seragam
        ax.grid(True, alpha=0.4, linestyle='--')
    
   
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" -> Plot evolusi (3 subplots) disimpan: {save_path}")
    plt.show()
    plt.close()

def plot_mass_vs_time(t, mass, save_path=None, title='Evolusi Mass vs Waktu'):
    plt.figure(figsize=(8, 5))
    plt.plot(t, mass, marker='o', color='navy', label = 'Total Massa')
    plt.axhline(y=mass[0], color='cornflowerblue', linestyle='--', label='Massa awal ($M_0$)')
    plt.xlabel('Waktu (t)')
    plt.ylabel('Total Massa')
    plt.title(title)
    plt.ylim(0, max(mass)*1.2)
    plt.grid(True)
    plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot evolusi massa disimpan ke {save_path}")
    
    plt.show()
    plt.close()

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
    mass_error_percent = mass_error(current_mass_pred, initial_mass_ref)
    print(f"-> Error Neraca Massa (%) : {mass_error_percent:.4e} %")
    
    print("\n-> Status: FUNGSI EVALUASI L2 & NERACA MASSA BERHASIL!")
'''