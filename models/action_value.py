import torch
import torch.nn as nn
import torch.nn.functional as F


class Action(nn.Module):
    def __init__(self, imagined_state_dim, output_dim, hidden_dim = [300, 300, 300]):
        super().__init__()
        # the action model takes as input the current imagined state and returns a mean of what the
        # action should be as well as a standard devaition. (i.e. the action returns a gaussian)
        self.layer_1 = nn.Linear(imagined_state_dim, hidden_dim[0])
        self.layer_2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.layer_3 = nn.Linear(hidden_dim[1], hidden_dim[2])

        self.mu = nn.Linear(hidden_dim[2], output_dim)
        self.std = nn.Linear(hidden_dim[2], output_dim)

    def forward(self, x):
        x = F.elu(self.layer_1(x))
        x = F.elu(self.layer_2(x))
        x = F.elu(self.layer_3(x))
        mu = self.mu(x) * 5
        std_raw = self.std(x)

        # applies the smooth, differentiable softplus activation function element-wise to an input tensor
        std = F.softplus(std_raw) 
        out = F.tanh(mu + std * torch.randn_like(mu))
        return out  

class Value(nn.Module):
    def __init__(self, imagined_state_dim, hidden_dim = [300, 300, 300]):
        super().__init__()
        self.layer_1 = nn.Linear(imagined_state_dim, hidden_dim[0])
        self.layer_2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.layer_3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.out = nn.Linear(hidden_dim[2], 1)
    
    def forward(self, x):
        x = F.elu(self.layer_1(x))
        x = F.elu(self.layer_2(x))
        x = F.elu(self.layer_3(x))
        out = self.out(x)
        return out
