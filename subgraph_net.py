import torch
import torch.nn as nn


class SubGraphLayer(nn.Module):
    def __init__(self, in_features=128, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.in_features = in_features
        self.encoder = nn.Linear(
            in_features=in_features, 
            out_features=hidden, 
            bias = True
        )
        torch.nn.init.kaiming_normal_(self.encoder.weight, nonlinearity='relu')
        self.layer_norm = nn.LayerNorm(hidden)
    
    def forward(self, x):
        '''
        Graph forward
        shape: (batch_size, num_polylines, nodes_per_polyline, features_per_node) -> (batch_size, num_polylines, nodes_per_polyline, 2 * hidden)

        x <- encode(x)
        ag <- aggregate(x)
        x <- concatenate(x, ag)
        '''
        x = self.encode(x)
        ag = self.aggregate(x)
        out = torch.concat([x, ag.repeat(1, 1, x.shape[2], 1)], dim=3)
        return out

    def encode(self, x):
        '''
        MLP forward
        shape: (batch_size, num_polylines, nodes_per_polyline, features_per_node) -> (batch_size, num_polylines, nodes_per_polyline, hidden)

        x <- linear(x)
        x <- layer_normalization(x)
        x <- relu(x)
        '''
        x = self.encoder(x)
        x = self.layer_norm(x)
        return torch.relu(x)

    def aggregate(sefl, x):
        '''
        maxpooling aggregation

        shape: (batch_size, num_polylines, nodes_per_polyline, features_per_node) -> (batch_size, num_polylines, 1, features_per_node)
        '''
        out, _ = torch.max(x, dim=2, keepdim=True)
        return out


class SubGraphNet(nn.Module):
    def __init__(self, init_features=14, hidden=64):
        super().__init__()
        self.polyline_net = nn.Sequential(
            SubGraphLayer(init_features, hidden),
            SubGraphLayer(hidden * 2, hidden),
            SubGraphLayer(hidden * 2, hidden),
            # SubGraphLayer(hidden * 2, hidden),
            # SubGraphLayer(hidden * 2, hidden),
        )
    
    def forward(self, x):
        '''
        Forward polyline-level graph network
        shape: (batch_size, num_polylines, nodes_per_polyline, init_features) -> (batch_size, num_polylines, 2 * hidden)

        x <- polyline_net(x)
        x <- maxpooling(x) (across polyline nodes)
        '''

        x = self.polyline_net(x)
        polyline_features = self.polyline_net[0].aggregate(x).squeeze(dim=2)
        return polyline_features, x

