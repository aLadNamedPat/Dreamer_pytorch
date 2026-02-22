import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Found the official implementation super helpful for finding the differences between what the paper specifies and what
# was actually implemented: https://github.com/google-research/dreamer?tab=readme-ov-file

class Action(nn.Module):
    def __init__(self, imagined_state_dim, output_dim, hidden_dim = [300, 300, 300], min_std = 1e-4, init_std = 5):
        super().__init__()
        # the action model takes as input the current imagined state and returns a mean of what the
        # action should be as well as a standard devaition. (i.e. the action returns a gaussian)
        self.min_std = min_std
        self.init_std = init_std

        self.layer_1 = nn.Linear(imagined_state_dim, hidden_dim[0])
        self.layer_2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.layer_3 = nn.Linear(hidden_dim[1], hidden_dim[2])

        self.mu = nn.Linear(hidden_dim[2], output_dim)
        self.std = nn.Linear(hidden_dim[2], output_dim)

    def forward(self, x):
        x = F.elu(self.layer_1(x))
        x = F.elu(self.layer_2(x))
        x = F.elu(self.layer_3(x))
        mu = F.tanh(self.mu(x) / 5) * 5
        std_raw = self.std(x)

        init_std = torch.log(torch.exp(self.init_std) - 1)  
        std = F.softplus(std_raw + init_std) + self.min_std 
        dist = Normal(mu, std)
        return dist

class Value(nn.Module):
    def __init__(self, imagined_state_dim, hidden_dim = [300, 300, 300], std = 1):
        super().__init__()
        self.std = std
        self.layer_1 = nn.Linear(imagined_state_dim, hidden_dim[0])
        self.layer_2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.layer_3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.out = nn.Linear(hidden_dim[2], 1)
    
    def forward(self, x):
        x = F.elu(self.layer_1(x))
        x = F.elu(self.layer_2(x))
        x = F.elu(self.layer_3(x))
        out = self.out(x)
        
        dist = Normal(out, self.std)
        return dist
