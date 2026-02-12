import torch
import pickle
import os
from env.vrp_env import VRPEnvironment

def generate_validation_dataset(num_instances, num_nodes, num_vehicles, capacity, weather_dim, save_path, device='cpu'):
    print(f"Generating {num_instances} static validation instances...")
    
    env = VRPEnvironment(num_nodes, num_vehicles, capacity, device=device, weather_dim=weather_dim)
    
    dataset = []
    
    for _ in range(num_instances):
        env.reset(batch_size=1, fixed_customers=False, is_deterministic=False)
        
        state_dict = env.get_clonable_state()
        
        cpu_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
        dataset.append(cpu_state)
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
        
    print(f"Dataset saved to {save_path}")
    return dataset

def load_dataset(path, device):
    """Loads a saved dataset and moves tensors to the specified device."""
    if not os.path.exists(path):
        return None
        
    print(f"Loading dataset from {path}...")
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
        
    # Move back to GPU/CPU as needed
    for state in dataset:
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
                
    return dataset