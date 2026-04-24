import torch
import torch.nn as nn
import os


class PINN_Network(nn.Module):
    def __init__(self):
        super(PINN_Network, self).__init__()
        
        # define the architecture
        self.layers = [2, 50, 50, 50, 50, 50, 1]

        # use nn.linear for each layer
        self.networks = nn.ModuleList()
        for i in range(len(self.layers)-1):
            self.networks.append(nn.Linear(self.layers[i], self.layers[i+1]))

        # use tanh activation function between layers
        self.activation = nn.Tanh()

        # initialize weights : using xavier initialization
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):       #x = tensor berukuran (batch_size, 2) yang berisi kolom (x,t)
        out = x
        for i in range (len(self.networks)-1):
            out = self.activation(self.networks[i](out))
        
        # output layer (no activationn func)
        output = self.networks[-1](out)
        return output
    

model = PINN_Network()
count_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {count_params}\nModel Summary:")
print(model)

'''
# test forward pass
x = torch.rand((50, 2), dtype=torch.float32)
output = model(x)
print(f"Input Value: {x}\nOutput shape: {output.shape}")
'''

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model disimpan ke: {path}")

def load_model(path):
    model = PINN_Network()
    model.load_state_dict(torch.load(path))
    print(f"Model dimuat dari: {path}")
    return model
