import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads, concat=True, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha

        self.W = nn.Linear(in_features, n_heads * out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x):
        batch_size, num_nodes, _ = x.shape
        
        # Linear Transform
        h = self.W(x)
        h = h.view(batch_size, num_nodes, self.n_heads, self.out_features)
        h = h.transpose(1, 2) # [B, Heads, N, Feat]
        
        # Attention
        h_i = h.unsqueeze(3).expand(-1, -1, -1, num_nodes, -1)
        h_j = h.unsqueeze(2).expand(-1, -1, num_nodes, -1, -1)
        attn_input = torch.cat([h_i, h_j], dim=-1)
        
        e = self.leakyrelu(self.a(attn_input)).squeeze(-1)
        attention = F.softmax(e, dim=-1)
        attention = self.dropout(attention)
        
        h_output = torch.matmul(attention, h) # [B, Heads, N, Feat]
        h_output = h_output.transpose(1, 2)   # [B, N, Heads, Feat]
        
        if self.concat:
            return h_output.reshape(batch_size, num_nodes, -1)
        else:
            return h_output.mean(dim=2)

class GATEncoder(nn.Module):
    """
    Advanced GAT with Residual Connections and Normalization.
    Allows for deeper networks (4+ layers).
    """
    def __init__(self, input_dim, embedding_dim, n_heads=4, n_layers=4):
        super(GATEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() # LayerNorm for stability
        
        # Input Projection to embedding_dim
        self.input_proj = nn.Linear(input_dim, embedding_dim)
        
        hidden_dim = embedding_dim // n_heads

        for _ in range(n_layers):
            # Enforce output to always be embedding_dim (via concatenation)
            self.layers.append(GraphAttentionLayer(embedding_dim, hidden_dim, n_heads, concat=True))
            self.norms.append(nn.LayerNorm(embedding_dim))

    def forward(self, x):
        # Initial Projection
        x = self.input_proj(x)
        
        for layer, norm in zip(self.layers, self.norms):
            h_in = x
            
            # Pass through GAT
            x = layer(x)
            
            # RESIDUAL CONNECTION: x = x_old + x_new
            x = x + h_in
            
            # NORMALIZATION
            x = norm(x)
            
            # Activation
            x = F.relu(x)
            
        return x
