import torch
import os
from tqdm import tqdm
from loss_function import total_loss_vanilla

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
        
    
 
