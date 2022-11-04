import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, in_features=128, out_features=60):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.decoder = nn.Linear(in_features=in_features, out_features=out_features)
    
    def forward(self, x):
        '''
        Forward decoder
        shape: (batch_size, num_agents, hidden * 2 = 128) -> (batch_size, num_agents, 2 (x, y) * num_time_steps@10Hz)

        query = x[agent]
        out <- global_net(query, x)
        out <- decoder(out)
        '''

        out = self.decoder(x)
        return out

class MLP(nn.Module):
    def __init__(self, in_features=256, hidden=64, out_features=60):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden, bias=True),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(in_features=hidden, out_features=out_features, bias=False)
        )
        torch.nn.init.kaiming_normal_(self.mlp[0].weight, nonlinearity='relu')
    
    def forward(self, x):
        '''
        Forward mlp decoder
        '''
        out = self.mlp(x)
        return out

class MLP2(nn.Module):
    def __init__(self, in_features=256, hidden=64, out_features=60):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hidden, out_features=out_features, bias=False)
        )
        # torch.nn.init.kaiming_normal_(self.mlp[0].weight, nonlinearity='relu')

    def forward(self, x):
        '''
        Forward mlp decoder
        '''
        out = self.mlp(x)
        return out