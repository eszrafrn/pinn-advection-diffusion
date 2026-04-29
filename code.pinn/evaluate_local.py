import sys
import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from network import PINN_Network
from evaluate import compute_l2_relative_error, compute_mass, mass_error, plot_solution_evolution, plot_mass_vs_time

sys.path.insert(0, os.path.abspath('code.pinn'))

def evaluate_and_log(model_path, ref_path, pe_val):
    # load data CN
    data_ref = np.load(ref_path)
    c_cn_all = data_ref['c']
    x_np = data_ref['x']
    t_np = data_ref['t']

    L = data_ref['L']
    dx = L/(len(x_np) - 1)

    # load model PINN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PINN_Network().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # prediksi PINN

    # mengekstrak PINN sepanjang waktu
    mass_pinn_history = []
    c_pred_evolution = []
    c_pred_full = []

    target_times = [0.0, 0.25, 0.5]  # waktu evaluasi yang diinginkan
    t_indices = [np.abs(t_np - t_target).argmin() for t_target in target_times]  # indeks waktu terdekat dalam data CN
    t_eval = t_np[t_indices] 
    
    x_tensor = torch.tensor(x_np, dtype=torch.float32).to(device).reshape(-1, 1)
    
    with torch.no_grad():
        for i, t_val in enumerate(t_np):
            t_tensor = torch.full_like(x_tensor, float(t_val)).reshape(-1, 1)
            input_tensor = torch.cat([x_tensor, t_tensor], dim=1)
            c_pred = model(input_tensor).cpu().numpy()

            c_pred_full.append(c_pred.flatten())
            mass_pinn_history.append(compute_mass(c_pred, dx))

            if i in t_indices:
                c_pred_evolution.append(c_pred.flatten())

    # ubah list ke bentuk matriks 2d
    c_pred_matrix = np.array(c_pred_full)

    # evaluasi
    # error di t = 0.5
    c_ref_final = c_cn_all[-1].reshape(-1, 1)
    c_pred_final = c_pred_evolution[-1].reshape(-1, 1)
    t_final = t_np[-1]
    
    l2_error = compute_l2_relative_error(c_pred_final, c_ref_final)
    l_inf_error = float(np.max(np.abs(c_pred_final - c_ref_final)))

    # global error
    global_l2_error = compute_l2_relative_error(c_pred_matrix.flatten(), c_cn_all.flatten())

    # mass error
    mass_cn_final = compute_mass(c_ref_final, dx)
    mass_pinn_final = mass_pinn_history[-1]
    mass_err = mass_error(mass_pinn_final, mass_cn_final)
   
    print(f"\n***** Evaluasi PINN Vanilla pada Bilangan Peclet {pe_val} *****")
    print(f"Global L2 Error: {global_l2_error:.6e}")
    print(f"  L2 Relative Error (t = 0.5s): {l2_error:.6e}")
    print(f"  L-infinity Error: {l_inf_error:.6e}")
    print(f"  Mass CN: {mass_cn_final:.6e}, Mass PINN: {mass_pinn_final:.6e}, Mass Error: {mass_err:.6e}")

    # save JSON log
    result_log = {
        'Pe': pe_val,
        'BC_Type': 'Dirichlet',
        't_eval': float(t_final),
        'Global_L2_Error': float(global_l2_error),
        'L2_Error_at_t_final': float(l2_error),
        'L_inf_Error_at_t_final': float(l_inf_error),
        'Mass_CN': float(mass_cn_final),
        'Mass_PINN': float(mass_pinn_final),
        'Mass_Error': float(mass_err)
    }

    os.makedirs('results/pinn_vanilla/', exist_ok=True)
    json_path = f'results/pinn_vanilla/pinn_vanilla_Pe={pe_val}_{result_log["BC_Type"]}.json'
    with open(json_path, 'w') as f:
        json.dump(result_log, f, indent=4)
    print(f"\nHasil evaluasi telah disimpan di {json_path}")

    # visualisasi perbandingan

    os.makedirs('figures/pinn_vanilla/', exist_ok=True)
    plot_solution_evolution(x_np.flatten(), t_eval, c_pred_evolution, save_path=f'figures/pinn_vanilla/Evolution_Pinn_vanilla_Pe={pe_val}_{result_log["BC_Type"]}.png', title=f'Evolusi Solusi PINN Vanilla (Pe={pe_val} [{result_log["BC_Type"]} BC]')
    plot_mass_vs_time(t_np, mass_pinn_history, save_path=f'figures/pinn_vanilla/Mass_Pinn_vanilla_Pe={pe_val}_{result_log["BC_Type"]}.png', title=f'Evolusi Massa PINN Vanilla (Pe={pe_val} [{result_log["BC_Type"]}BC]')

    plt.figure(figsize=(10, 6))
    plt.plot(x_np, c_ref_final, label='CN (Referensi)', color='blue', linewidth=2.5)
    plt.plot(x_np, c_pred_final, label='PINN (Prediksi)', color='red', linestyle='--', linewidth=2.2)
    plt.xlabel('Posisi (x)')
    plt.ylabel('Konsentrasi (c)')
    plt.title(f'CN vs PINN Vanilla at Pe={pe_val} (t={t_final} s) [{result_log["BC_Type"]}] BC]', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig(f'figures/pinn_vanilla/Comparison_Pinn_vanilla_Pe={pe_val}_{result_log["BC_Type"]}.png')
    plt.show()

if __name__ == '__main__':
    PE_VALUE = 50.0
    MODEL_FILE = f'models\PINN_Vanilla\PINN_VANILLA_Pe={PE_VALUE}_Dirichlet.pth'
    REF_FILE = f'data/reference/reference_Dirichlet_Pe={PE_VALUE:.2f}.npz'
    if os.path.exists(REF_FILE):
        evaluate_and_log(MODEL_FILE, REF_FILE, PE_VALUE)
    else:
        print(f"ERROR: File referensi '{REF_FILE}' tidak ditemukan!")