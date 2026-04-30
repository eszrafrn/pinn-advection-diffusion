from ast import If

import torch
import os
from tqdm import tqdm
from loss_function import total_loss_vanilla
from mass_penalty_pinn import gaussian_quadrature, sample_time_points, compute_mass_penalty


def train_adam (model, data, params, epochs=1000, lr=1e-3, save_dir='checkpoints'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #direktori untuk menyimpan model dan log
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    history = {'epoch': [], 'total_loss': [], 'L_IC': [], 'L_BC': [], 'L_PDE': []}
    
    # header log
    with open('logs/training_log.csv', 'w') as log_file:
        log_file.write('Epoch\tTotal_Loss\tLoss_IC\tLoss_BC\tLoss_PDE\n')
    
    print(f"\nMemulai Training ADAM dengan {epochs} epochs dan learning rate {lr:.1e}...")
    
    #setup progress bar
    progress_bar = tqdm(range(1, epochs+1), desc='Training Progress')
    for epoch in progress_bar:
        optimizer.zero_grad()
        total_loss, L_IC, L_BC, L_PDE = total_loss_vanilla(model, data, params)
        total_loss.backward()       #backpropagation
        optimizer.step()
        
        history['epoch'].append(epoch)
        history['total_loss'].append(total_loss.item())
        history['L_IC'].append(L_IC.item())
        history['L_BC'].append(L_BC.item())
        history['L_PDE'].append(L_PDE.item())
            
        #update progress bar
        progress_bar.set_postfix({
            'Total Loss': f"{total_loss.item():.6e}",
            'L_IC': f"{L_IC.item():.6e}",
            'L_BC': f"{L_BC.item():.6e}",
            'L_PDE': f"{L_PDE.item():.6e}"
        })
        
        #simpan log setiap 100 epochs
        if epoch % 100 == 0 or epoch == epochs:
            with open('logs/training_log.csv', 'a') as log_file:
                log_file.write(f"{epoch}\t{total_loss.item():.6e}\t{L_IC.item():.6e}\t{L_BC.item():.6e}\t{L_PDE.item():.6e}\n")

        # model checkpoint setiap 1/4 total epochs    
        if epoch > 0 and epoch % (epochs // 4) == 0:    
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch}.pth'))
        
    # simpan mode final
    torch.save(model.state_dict(), os.path.join(save_dir, 'model_final.pth'))
    print(f"\nTraining selesai. Model dan log telah disimpan di {save_dir}/model_final.pth.")
    
    return history
        

# trainer untuk optimasi LBFGS
def train_lbfgs(model, data, params, max_iter=1000, save_dir='checkpoints'):
    
    print(f"\nMemulai Training LBFGS dengan max iterasi {max_iter}...")

    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=max_iter, max_eval=max_iter*1.25, history_size=50, tolerance_grad=1e-7, tolerance_change=1e-9, line_search_fn='strong_wolfe')

    history = {'iter': [], 'total_loss': [], 'L_IC': [], 'L_BC': [], 'L_PDE': []}

    iter_count = 0

    def closure():
        nonlocal iter_count
        optimizer.zero_grad()

        # hitung loss
        total_loss, L_IC, L_BC, L_PDE = total_loss_vanilla(model, data, params)

        # backpropagation
        total_loss.backward()

        # simpan history
        history['iter'].append(iter_count)
        history['total_loss'].append(total_loss.item())
        history['L_IC'].append(L_IC.item())
        history['L_BC'].append(L_BC.item())
        history['L_PDE'].append(L_PDE.item())
        
        if iter_count % 100 == 0:
            print(f"L-BFGS Iterasi {iter_count:4d} | Total Loss: {total_loss.item():.6e}, L_IC: {L_IC.item():.6e}, L_BC: {L_BC.item():.6e}, L_PDE: {L_PDE.item():.6e}")
            with open('logs/lbfgs_training_log.csv', 'a') as log_file:
                log_file.write(f"L-BFGS_{iter_count}\t{total_loss.item():.6e}\t{L_IC.item():.6e}\t{L_BC.item():.6e}\t{L_PDE.item():.6e}\n")
       
        iter_count += 1
        return total_loss

    # Eksekusi optimasi L-BFGS dengan closure
    model.train()
    optimizer.step(closure)

    # simpan model final L-BFGS
    torch.save(model.state_dict(), f"{save_dir}/model_lbfgs_final.pth")
    print(f"\nTraining L-BFGS selesai. Model disimpan di {save_dir}/model_lbfgs_final.pth.")

    return history

def train_pinn_vanilla(model, data, params, epochs_adam = 1000, epochs_lbfgs=500, lr_adam=1e-3):
    # 2 stage training:
    # stage 1: optimasi dengan ADAM (Global Search)
    # stage 2: optimasi dengan L-BFGS (Fine-tuning Local Search)

    print("\n=== MEMULAI 2-STAGE TRAINING ===")
    history_adam = train_adam(model, data, params, epochs=epochs_adam, lr=lr_adam)
    history_lbfgs = train_lbfgs(model, data, params, max_iter=epochs_lbfgs)

    history_comb = {
        'total_loss': history_adam['total_loss'] + history_lbfgs['total_loss'],
        'L_IC': history_adam['L_IC'] + history_lbfgs['L_IC'],
        'L_BC': history_adam['L_BC'] + history_lbfgs['L_BC'],
        'L_PDE': history_adam['L_PDE'] + history_lbfgs['L_PDE'],
    }

    return history_comb

def train_pinn_conservative(model, data, params, epochs_adam = 1000, epochs_lbfgs=500, lr_adam=1e-3):
    device = data['x_ic'].device
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=lr_adam)
    
    # setup quadrature untuk penalti massa
    L = 1.0
    T = 0.5
    x_quad, w_quad = gaussian_quadrature(n=50, L=L, device=device)
    t_samples = sample_time_points(n_t=25, T=T, device=device)

    # ekstrak parameter
    lambda_ic = params['lambda_ic']
    lambda_bc = params['lambda_bc']
    lambda_pde = params['lambda_pde']
    lambda_mass = params['lambda_mass']
    M0 = params['M0']

    history = {'epoch': [], 'total_loss': [], 'L_IC': [], 'L_BC': [], 'L_PDE': [], 'L_Mass': []}

    print("\n=== MEMULAI TRAINING PINN-CONSERVATIVE ===")
    for epoch in range(epochs_adam):
        optimizer_adam.zero_grad()

        # hitung loss utama
        total_loss, L_IC, L_BC, L_PDE = total_loss_vanilla(model, data, params)

        # hitung penalti massa
        L_Mass = compute_mass_penalty(model, x_quad, w_quad, t_samples, M0)

        # total loss dengan penalti massa
        total_loss_conservative = total_loss + (lambda_mass * L_Mass)

        # backpropagation
        total_loss_conservative.backward()
        optimizer_adam.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Total Loss: {total_loss_conservative.item():.6e}, L_IC: {L_IC.item():.6e}, L_BC: {L_BC.item():.6e}, L_PDE: {L_PDE.item():.6e}, L_Mass: {L_Mass.item():.6e}")
            
        # simpan history
        history['epoch'].append(epoch)
        history['total_loss'].append(total_loss_conservative.item())
        history['L_IC'].append(L_IC.item())
        history['L_BC'].append(L_BC.item())
        history['L_PDE'].append(L_PDE.item())
        history['L_Mass'].append(L_Mass.item())
    
    print("\n Memulai optimasi L-BFGS untuk fine-tuning...")
    optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=epochs_lbfgs, max_eval=epochs_lbfgs*1.25, history_size=100, tolerance_grad=1e-7, tolerance_change=1e-9, line_search_fn='strong_wolfe')

    def closure():
        optimizer_lbfgs.zero_grad()

        # hitung loss utama
        total_loss, L_IC, L_BC, L_PDE = total_loss_vanilla(model, data, params)

        # hitung penalti massa
        L_Mass = compute_mass_penalty(model, x_quad, w_quad, t_samples, M0)

        # total loss dengan penalti massa
        total_loss_conservative = total_loss + (lambda_mass * L_Mass)

        # backpropagation
        total_loss_conservative.backward()

        return total_loss_conservative


    optimizer_lbfgs.step(closure)
    return history