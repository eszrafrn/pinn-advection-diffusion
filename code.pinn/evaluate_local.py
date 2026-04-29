import sys
import os
import matplotlib
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from network import PINN_Network
from evaluate import compute_l2_relative_error, compute_mass, mass_error

sys.path.insert(0, os.path.abspath('code.pinn'))

def evaluate_and_log(model_path, ref_path):
    # load data CN
    data_ref = np.load(ref_path)
    c_cn_all = data_ref['c']
    x_np = data_ref['x']
    t_np = data_ref['t']

    L = data_ref['L']
    dx = L/(len(x_np) - 1)
    t_eval_idx = -1
    t_eval = t_np[t_eval_idx]
    c_ref = c_cn_all[t_eval_idx].reshape(-1, 1)

    # load model PINN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PINN_Network().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # prediksi PINN
    x_tensor = torch.tensor(x_np, dtype=torch.float32).to(device).reshape(-1, 1)
    t_tensor = torch.full_like(x_tensor, t_eval).to(device).reshape(-1, 1)
    input_tensor = torch.cat([x_tensor, t_tensor], dim=1)
    with torch.no_grad():
        c_pred = model(input_tensor).cpu().numpy()

    # evaluasi
    l2_error = compute_l2_relative_error(c_pred, c_ref)
    l_inf_error = float(np.max(np.abs(c_pred - c_ref)))

    mass_cn = compute_mass(c_ref, dx)
    mass_pinn = compute_mass(c_pred, dx)
    mass_err = mass_error(mass_pinn, mass_cn)
    print(f"Evaluasi pada t={t_eval}s:")
    print(f"  L2 Relative Error: {l2_error:.6e}")
    print(f"  L-infinity Error: {l_inf_error:.6e}")
    print(f"  Mass CN: {mass_cn:.6e}, Mass PINN: {mass_pinn:.6e}, Mass Error: {mass_err:.6e}")

    # save JSON log
    result_log = {
        'Pe': 1.0,
        'BC_Type': 'Dirichlet',
        't_eval': float(t_eval),
        'L2_Error': float(l2_error),
        'L_inf_Error': float(l_inf_error),
        'Mass_CN': float(mass_cn),
        'Mass_PINN': float(mass_pinn),
        'Mass_Error': float(mass_err)
    }

    os.makedirs('results', exist_ok=True)
    json_path = 'results/pinn_vanilla_Pe1_Dirichlet_results.json'
    with open(json_path, 'w') as f:
        json.dump(result_log, f, indent=4)
    print(f"\nHasil evaluasi telah disimpan di {json_path}")

    # visualisasi perbandingan
    plt.figure(figsize=(10, 6))
    plt.plot(x_np, c_ref, label='CN Reference', color='navy', linewidth=2)
    plt.plot(x_np, c_pred, label='PINN Prediction', color='red', linestyle='--', linewidth=2)
    plt.xlabel('Posisi (x)')
    plt.ylabel('Konsentrasi (c)')
    plt.title(f'CN vs PINN Vanilla pada Waktu t = {t_eval}s dengan Pe = 1.0 (Dirichlet BC)')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    os.makedirs('figures', exist_ok=True)
    fig_path = 'figures/pinn_vanilla_Pe1_Dirichlet_comparison.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Perbandingan solusi CN dan PINN disimpan di: {fig_path}")
    plt.show()

if __name__ == '__main__':
    MODEL_FILE = 'models/PINN_Vanilla_Pe1_Dirichlet.pth'
    REF_FILE = 'data/reference/reference_Dirichlet_Pe=1.00.npz'
    
    if os.path.exists(REF_FILE):
        evaluate_and_log(MODEL_FILE, REF_FILE)
    else:
        print(f"ERROR: File referensi '{REF_FILE}' tidak ditemukan!")