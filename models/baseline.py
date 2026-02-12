import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    """
    A Deeper Baseline Model (Critic) to better estimate VRP complexity.
    """
    def __init__(self, input_dim, hidden_dim):
        super(BaselineModel, self).__init__()
        
        # 4-Layer MLP for better approximation
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, customer_features, vehicle_features):
        # Simple aggregation: Mean of customers + Vehicle state
        avg_customer = torch.mean(customer_features, dim=1) 
        avg_vehicle = torch.mean(vehicle_features, dim=1)   
        
        x = torch.cat([avg_customer, avg_vehicle], dim=1)
        return self.net(x)