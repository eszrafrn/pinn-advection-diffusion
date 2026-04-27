import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from network import PINN_Network
from sampling import prepare_training_data
from trainer import train_adam

if __name__ == '__main__':
    # 1. Parameter Fisika (Uji Coba)
    L = 1.0
    T = 0.5
    v = 0.05 
    D = 0.01 
    
    # 2. Persiapkan Data Sampling (Skala Kecil)
    #menggunakan Neumann BC di pengujian loss terakhir
    print("Generating collocation points...")
    data = prepare_training_data(N_r=500, N_IC=200, N_BC=100, L=L, T=T, bc_type='Neumann')
    
    # 3. Mendefinisikan Target Kondisi Awal (IC) (Gaussian Pulse di x=0.5)
    x_ic = data['x_ic']
    sigma = 0.05
    c_ic_true = torch.exp(-((x_ic - 0.5)**2) / (2 * sigma**2))
    
    # 4. Bungkus ke dalam Dictionary Parameters
    params = {
        'v': v, 'D': D,
        'bc_type': 'Neumann',
        'c_ic_true': c_ic_true,
        'lambda_ic': 100.0, 'lambda_bc': 10.0, 'lambda_pde': 1.0
    }
    
    # 5. Inisialisasi Model AI
    torch.manual_seed(42)
    model = PINN_Network()
    
    # 6. Eksekusi Training (1000 Epoch saja untuk tes CPU laptop)
    epochs = 1000
    history = train_adam(model, data, params, epochs=epochs, lr=1e-3)
    
    # 7. Visualisasi Kurva Loss
    os.makedirs('figures', exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    # Gunakan skala logaritmik (log scale) karena nilai loss akan turun drastis
    plt.yscale('log')
    plt.plot(history['total_loss'], label='Total Loss', color='navy', linewidth=2)
    plt.plot(history['L_IC'], label='IC Loss', alpha=0.7, color='yellow')
    plt.plot(history['L_BC'], label='BC Loss', alpha=0.7, color='red')
    plt.plot(history['L_PDE'], label='PDE Loss', alpha=0.7, color='purple')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value (Log Scale)')
    plt.title(f'Kurva Pembelajaran PINN (Uji Skala Kecil - {epochs} Epochs)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path = 'figures/test_loss_curve.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nKurva loss berhasil disimpan ke: {save_path}")
    plt.show()