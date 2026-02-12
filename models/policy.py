import torch
import torch.nn as nn
import torch.nn.functional as F
from .gat import GATEncoder

class VRPEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(VRPEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, embedding_dim)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        batch_size, num_entities, _ = x.shape
        x_reshaped = x.view(batch_size * num_entities, -1)
        embedded_x = self.activation(self.fc(x_reshaped))
        return embedded_x.view(batch_size, num_entities, -1)

class VRPAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(VRPAttention, self).__init__()
        self.wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.scale = torch.sqrt(torch.tensor(embedding_dim, dtype=torch.float32))
        
    def forward(self, customer_embeddings, vehicle_embeddings, mask=None):
        q = self.wq(vehicle_embeddings) 
        k = self.wk(customer_embeddings)
        
        # [B, V, E] @ [B, E, N] -> [B, V, N]
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # SAFETY 1: Clamp scores to prevent Inf/NaN in Softmax
        # 10/-10 is a wide enough range for logits, prevents exploding gradients
        scores = torch.clamp(scores, min=-10.0, max=10.0)
        
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        
        return F.softmax(scores, dim=-1)

class VRPPolicy(nn.Module):
    def __init__(self, customer_input_dim, vehicle_input_dim, embedding_dim, n_heads, n_layers):
        super(VRPPolicy, self).__init__()
        self.customer_encoder = GATEncoder(
            input_dim=customer_input_dim, 
            embedding_dim=embedding_dim, 
            n_heads=n_heads,
            n_layers=n_layers
        )
        self.vehicle_encoder = VRPEncoder(vehicle_input_dim, embedding_dim)
        self.attention = VRPAttention(embedding_dim)
        
    def forward(self, customer_features, vehicle_features, demands, vehicle_positions, hidden=None):
        batch_size, num_nodes, _ = customer_features.shape
        num_vehicles = vehicle_features.shape[1]
        
        customer_embeddings = self.customer_encoder(customer_features)
        vehicle_embeddings = self.vehicle_encoder(vehicle_features)
        
        # --- Masks ---
        node_mask = (demands <= 1e-3) 
        node_mask[:, 0] = False
        mask = node_mask.unsqueeze(1).expand(batch_size, num_vehicles, num_nodes).clone()
        
        current_loads = vehicle_features[:, :, 1]
        is_empty = (current_loads <= 1e-2)
        is_empty_expanded = is_empty.unsqueeze(2)
        
        if num_nodes > 1:
            mask[:, :, 1:] = mask[:, :, 1:] | is_empty_expanded
        
        mask[:, :, 0] = mask[:, :, 0] & (~is_empty)

        has_unserved_customers = (~node_mask[:, 1:]).any(dim=1)
        has_plenty_load = (current_loads > 0.25)
        should_mask_depot = has_plenty_load & has_unserved_customers.unsqueeze(1)
        mask[:, :, 0] = mask[:, :, 0] | should_mask_depot

        batch_indices = torch.arange(batch_size, device=demands.device).unsqueeze(1).expand(-1, num_vehicles)
        vehicle_indices = torch.arange(num_vehicles, device=demands.device).unsqueeze(0).expand(batch_size, -1)
        mask[batch_indices, vehicle_indices, vehicle_positions] = True

        all_demands_met = (demands[:, 1:] <= 1e-3).all(dim=1)
        if all_demands_met.any():
            mask[all_demands_met, :, 0] = False

        probs = self.attention(customer_embeddings, vehicle_embeddings, mask)
        
        # SAFETY 2: Epsilon in log to prevent log(0) -> -Inf
        return torch.log(probs + 1e-8), None 
    
    def sample_action(self, log_probs, greedy=False):
        batch_size, num_vehicles, num_nodes = log_probs.shape
        if greedy:
            return torch.argmax(log_probs, dim=-1) 
        else:
            probs = torch.exp(log_probs)
            probs_2d = probs.reshape(-1, num_nodes)
            
            # Renormalize to ensure sum=1 (fix float precision errors)
            sum_probs = probs_2d.sum(dim=1, keepdim=True)
            probs_2d = probs_2d / (sum_probs + 1e-10)
            
            # SAFETY 3: Check for NaNs
            # If NaNs exist, replace with uniform random valid moves to prevent crash
            if torch.isnan(probs_2d).any():
                print("[Warning] NaN detected in probabilities. Replacing with uniform random.")
                probs_2d = torch.where(torch.isnan(probs_2d), torch.ones_like(probs_2d), probs_2d)
                probs_2d = probs_2d / probs_2d.sum(dim=1, keepdim=True)

            actions = torch.multinomial(probs_2d, 1) 
            return actions.reshape(batch_size, num_vehicles)