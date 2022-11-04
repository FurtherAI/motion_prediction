import torch
import torch.nn as nn


class GlobalGraphLayer(nn.Module):
    def __init__(self, in_features=128, out_features=128, num_heads=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.proj_Q = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.proj_K = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.proj_V = nn.Linear(in_features=in_features, out_features=out_features, bias=False)

        self.softmax = nn.Softmax(dim=2)

        # self.multi_head_attn = nn.MultiheadAttention(embed_dim=in_features, num_heads=num_heads, bias=True, batch_first=True)
        # self.layer_norm = nn.LayerNorm()

    
    def forward(self, query, x):
        '''
        Self-attention
        query shape: (batch_size, range(1 : num_polylines), in_features) (can vary, calculating queries for all global graph nodes, or just the agent of interest)
        x shape: (batch_size, num_polylines, in_features)
        out shape: (batch_size, range(1 : num_polylines), out_features)

        Q <- proj_Q(query)
        K <- proj_K(x)
        V <- proj_V(x)

        out <- softmax(Q K^T / sqrt(out_features)) V
        '''
        Q = self.proj_Q(query)
        K = self.proj_K(x)
        V = self.proj_V(x)

        attention = self.softmax((torch.bmm(Q, K.transpose(1, 2))) / (self.out_features ** 0.5))
        out = torch.bmm(attention, V)

        # identity = query
        # out = self.multihead_attn_layer(query, x, x, need_weights=False)
        # out = self.layer_norm(identity + out)
        return out


class GlobalGraphNet(nn.Module):
    def __init__(self, in_features=128, out_features=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.global_net = GlobalGraphLayer(in_features=in_features, out_features=out_features)
    
    def forward(self, query, x):
        '''
        Forward global-level graph network
        query shape: (batch_size, range(1 : num_polylines), in_features)
        x shape: (batch_size, num_polylines, in_features)
        out shape: (batch_size, range(1 : num_polylines), out_features)

        query = x[agent]
        out <- global_net(query, x)
        '''

        out = self.global_net(query, x)
        return out



class GlobalGraphNet2(nn.Module):
    def __init__(self, in_features=128, out_features=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.attn1 = GlobalGraphLayer(in_features=in_features, out_features=out_features)
        self.attn2 = GlobalGraphLayer(in_features=in_features, out_features=out_features)
    
    def forward(self, x, query_lastn):
        '''
        Forward global-level graph network
        query shape: (batch_size, range(1 : num_polylines), in_features)
        x shape: (batch_size, num_polylines, in_features)
        out shape: (batch_size, range(1 : num_polylines), out_features)

        query = x[agent]
        out <- global_net(query, x)
        '''

        out = self.attn1(x, x)
        out = self.attn2(x[:, -query_lastn:, :], x)
        return out
