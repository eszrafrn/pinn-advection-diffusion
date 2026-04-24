import matplotlib.pyplot as plt
import numpy as np
import os

def plot_solution(x, c, title='Solusi', save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(x, c, label='c(x)', color='navy')
    plt.xlabel('Posisi (x)')
    plt.ylabel('Konsentrasi (c)')
    plt.title(title)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot solusi disimpan ke {save_path}")
    
    plt.show()
    plt.close()

def plot_solution_evolution(x, t_array, c_array, save_path=None, title='Evolusi Solusi'):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.cividis(np.linspace(0, 1, len(t_array)))
    for i, (t,c) in enumerate(zip(t_array, c_array)):
        plt.plot(x, c, color=colors[i])
    plt.xlabel('Posisi (x)')
    plt.ylabel('Konsentrasi (c)')
    plt.title(title)
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot evolusi solusi disimpan ke {save_path}")
    
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