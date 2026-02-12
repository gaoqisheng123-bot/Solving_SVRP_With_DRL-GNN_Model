import torch
import numpy as np

class VRPEnvironment:
    """
    Environment for Stochastic Vehicle Routing Problem (SVRP).
    SIZE-AGNOSTIC: Uses Coordinates instead of Distance Matrix for features.
    """
    
    def __init__(self, num_nodes, num_vehicles, capacity, device='cpu', weather_dim=3):
        self.num_nodes = num_nodes
        self.num_vehicles = num_vehicles
        self.capacity = capacity
        self.device = device
        self.weather_dim = weather_dim 
        
        self.fixed_customer_positions = None
        self.current_customer_positions = None 
        self.current_weather_vector = None 
        self.current_alpha = None
        self.customer_features_dim = 1 + self.weather_dim + 2 + 1 
        self.vehicle_features_dim = 2 

        self.base_demands = None       
        self.real_demands = None       
        self.observable_demands = None 
        self.remaining_real_demands = None 
        
        self.travel_costs = None
        self.vehicle_positions = None
        self.vehicle_loads = None
        self.visited = None
        self.steps = 0
        
    def reset(self, batch_size=1, fixed_customers=True, preset_data=None, is_deterministic=False):
        # --- 1. Positioning (Standard) ---
        if preset_data is not None and 'locations' in preset_data:
            locs = torch.tensor(preset_data['locations'], dtype=torch.float, device=self.device)
            customer_positions = locs.unsqueeze(0).repeat(batch_size, 1, 1)
            self.fixed_customer_positions = customer_positions[0].unsqueeze(0).clone()
        elif fixed_customers and self.fixed_customer_positions is not None:
            customer_positions = self.fixed_customer_positions.repeat(batch_size, 1, 1)
        else:
            customer_positions = torch.zeros(batch_size, self.num_nodes, 2, device=self.device)
            customer_positions[:, 0] = torch.tensor([0.5, 0.5], device=self.device)
            for b in range(batch_size):
                customer_positions[b, 1:] = torch.rand(self.num_nodes - 1, 2, device=self.device)
            if fixed_customers and self.fixed_customer_positions is None:
                self.fixed_customer_positions = customer_positions[0].unsqueeze(0).clone()
        
        self.current_customer_positions = customer_positions
        
        # --- 2. Weather Vector Generation (Standard) ---
        if preset_data is not None and 'weather' in preset_data:
            w_vec = torch.tensor(preset_data['weather'], dtype=torch.float, device=self.device)
            self.current_weather_vector = w_vec.unsqueeze(0).repeat(batch_size, 1)
        else:
            self.current_weather_vector = torch.rand(batch_size, self.weather_dim, device=self.device) * 2 - 1

        # --- 3. SPATIALLY CORRELATED DEMAND ---
        self.real_demands = torch.zeros(batch_size, self.num_nodes, device=self.device)
        self.base_demands = torch.zeros(batch_size, self.num_nodes, device=self.device)
        
        # A. Base Demand
        if preset_data is not None and 'base_demands' in preset_data:
             for i in range(1, self.num_nodes):
                 self.base_demands[:, i] = float(preset_data['base_demands'][i])
        else:
             rand_scalar = torch.rand(batch_size, self.num_nodes, device=self.device) 
             self.base_demands = self.capacity * (0.05 + 0.20 * rand_scalar)
             self.base_demands[:, 0] = 0.0 # Depot is 0

        # B. Alpha (Sensitivity) Logic
        if preset_data is not None and 'alpha' in preset_data:
             alpha = torch.tensor(preset_data['alpha'], device=self.device).repeat(batch_size, 1, 1, 1)
        else:
             # 1. Start with small random noise (Base Personality)
             alpha = torch.randn(batch_size, self.num_nodes, self.weather_dim, self.weather_dim, device=self.device) * 0.1
             
             # 2. Extract Y-Coordinates (North/South position)
             y_coords = self.current_customer_positions[:, :, 1]
             
             # 3. APPLY SPATIAL ZONING RULES
             # Rule: North (>0.7) reacts to Weather[0] (e.g. Heat)
             north_mask = (y_coords > 0.7).float()
             alpha[:, :, 0, 0] += north_mask * 2.0 
             
             # Rule: South (<0.3) reacts to Weather[1] (e.g. Rain)
             south_mask = (y_coords < 0.3).float()
             alpha[:, :, 1, 1] += south_mask * 2.0

        # C. Calculate Effect & Final Demand
        self.current_alpha = alpha
        weather_effect = torch.einsum('bi,bnij,bj->bn', self.current_weather_vector, self.current_alpha, self.current_weather_vector)
        weather_effect = weather_effect * (self.capacity * 0.1)
        
        noise = torch.randn(batch_size, self.num_nodes, device=self.device) * (self.capacity * 0.05)
        
        if is_deterministic:
            self.real_demands = self.base_demands.clone()
        else:
            self.real_demands = self.base_demands + weather_effect + noise
        
        self.real_demands = torch.clamp(self.real_demands, min=1.0, max=self.capacity * 0.5)
        self.real_demands[:, 0] = 0.0

        self.observable_demands = self.base_demands.clone()
        self.remaining_real_demands = self.real_demands.clone()
        
        # --- 4. Travel Costs (Recalculate for new map) ---
        self.travel_costs = torch.zeros(batch_size, self.num_nodes, self.num_nodes, device=self.device)
        for b in range(batch_size):
            p = self.current_customer_positions[b]
            dists = torch.cdist(p, p, p=2)
            self.travel_costs[b] = dists * 2.0 
        
        # --- 5. Vehicle State ---
        self.vehicle_positions = torch.zeros(batch_size, self.num_vehicles, dtype=torch.long, device=self.device) 
        self.vehicle_loads = torch.full((batch_size, self.num_vehicles), self.capacity, dtype=torch.float32, device=self.device)
        self.visited = torch.zeros(batch_size, self.num_nodes, dtype=torch.bool, device=self.device)
        self.visited[:, 0] = True
        self.steps = 0
        
        return self._get_features()
    
    def step(self, actions):
        batch_size = actions.size(0)
        rewards = torch.zeros(batch_size, device=self.device)
        
        for v_idx in range(self.num_vehicles):
            current_vehicle_positions = self.vehicle_positions[:, v_idx]
            next_node_for_vehicle = actions[:, v_idx]
            
            for b_idx in range(batch_size):
                current_loc = current_vehicle_positions[b_idx].item()
                next_loc = next_node_for_vehicle[b_idx].item()
                
                rewards[b_idx] -= self.travel_costs[b_idx, current_loc, next_loc]
                
                if next_loc > 0:  # Visit Customer
                    if not self.visited[b_idx, next_loc]: 
                         self.visited[b_idx, next_loc] = True
                    
                    real_needed = self.remaining_real_demands[b_idx, next_loc]
                    vehicle_has = self.vehicle_loads[b_idx, v_idx]
                    
                    delivered = torch.min(vehicle_has, real_needed)
                    
                    self.remaining_real_demands[b_idx, next_loc] -= delivered
                    self.vehicle_loads[b_idx, v_idx] -= delivered
                    
                    self.observable_demands[b_idx, next_loc] = self.remaining_real_demands[b_idx, next_loc]

                    if self.vehicle_loads[b_idx, v_idx] <= 1e-3 and self.remaining_real_demands[b_idx, next_loc] > 1e-3:
                        rewards[b_idx] -= 50.0 
                    
                else: # Depot
                    self.vehicle_loads[b_idx, v_idx] = self.capacity 
            
            self.vehicle_positions[:, v_idx] = next_node_for_vehicle
        
        # Check completion
        all_demands_met = torch.all(self.remaining_real_demands[:, 1:] <= 1e-3, dim=1) 
        is_at_depot = (self.vehicle_positions[:, 0] == 0)
        done_per_instance = all_demands_met & is_at_depot

        rewards[~done_per_instance] -= 0.1 
        self.steps += 1

        completion_bonus = self.num_nodes * 50.0
        rewards[done_per_instance] += completion_bonus 

        max_episode_steps = self.num_nodes * 3
        if self.steps >= max_episode_steps:
            timeout_penalty_mask = ~done_per_instance
            rewards[timeout_penalty_mask] -= 100.0 
            for b_idx in range(batch_size):
                if timeout_penalty_mask[b_idx]:
                    num_unserved = (self.remaining_real_demands[b_idx, 1:] > 1e-3).sum().item()
                    rewards[b_idx] -= num_unserved * 20.0 
            done_per_instance = torch.full_like(done_per_instance, True, dtype=torch.bool)

        return self._get_features(), rewards, done_per_instance
    
    def _get_features(self):
        batch_size = self.remaining_real_demands.size(0)
        
        customer_features = torch.zeros(batch_size, self.num_nodes, self.customer_features_dim, device=self.device)
        
        # 1. Observable Demands
        customer_features[:, :, 0] = self.observable_demands / self.capacity 
        
        # 2. Weather Vector
        weather_expanded = self.current_weather_vector.unsqueeze(1).repeat(1, self.num_nodes, 1)
        customer_features[:, :, 1:1+self.weather_dim] = weather_expanded

        # 3. Coordinates
        start_idx = 1 + self.weather_dim
        customer_features[:, :, start_idx:start_idx+2] = self.current_customer_positions

        # 4. Mask
        demand_met_mask = (self.remaining_real_demands <= 1e-3).float()
        all_customers_done = torch.all(self.remaining_real_demands[:, 1:] <= 1e-3, dim=1) 
        all_done_expanded = all_customers_done.unsqueeze(1).repeat(1, self.num_nodes)
        
        final_mask = torch.where(all_done_expanded, torch.ones_like(demand_met_mask), demand_met_mask)
        final_mask[:, 0] = 0.0 # Depot always open
        
        customer_features[:, :, -1] = final_mask
        
        vehicle_features = torch.zeros(batch_size, self.num_vehicles, self.vehicle_features_dim, device=self.device)
        vehicle_features[:, :, 0] = self.vehicle_positions.float() / self.num_nodes 
        vehicle_features[:, :, 1] = self.vehicle_loads / self.capacity 
        
        return customer_features, vehicle_features, self.remaining_real_demands

    def get_clonable_state(self):
        if self.real_demands is None: return None
        alpha_to_save = self.current_alpha[0].detach().clone() if self.current_alpha is not None else None
        return {
            'real_demands': self.real_demands[0].detach().clone(),
            'remaining_real_demands': self.remaining_real_demands[0].detach().clone(),
            'observable_demands': self.observable_demands[0].detach().clone(),
            'base_demands': self.base_demands[0].detach().clone(),
            'current_customer_positions': self.current_customer_positions[0].detach().clone(),
            'travel_costs': self.travel_costs[0].detach().clone(),
            'vehicle_positions': self.vehicle_positions[0].detach().clone(),
            'vehicle_loads': self.vehicle_loads[0].detach().clone(),
            'visited': self.visited[0].detach().clone(),
            'steps': self.steps,
            'current_weather_vector': self.current_weather_vector[0].detach().clone(),
            'fixed_customer_positions': self.fixed_customer_positions.clone() if self.fixed_customer_positions is not None else None,
            'alpha': alpha_to_save
        }

    # IMPORTANT: Indentation fixed here
    """
    def set_clonable_state(self, state_dict, batch_size=1, is_deterministic=False):
        self.current_customer_positions = state_dict['current_customer_positions'].clone().unsqueeze(0)
        self.base_demands = state_dict['base_demands'].clone().unsqueeze(0)
        self.current_weather_vector = state_dict['current_weather_vector'].clone().unsqueeze(0)
        
        if is_deterministic:
            self.real_demands = self.base_demands.clone()
        else:
            self.real_demands = state_dict['real_demands'].clone().unsqueeze(0)
            
        self.remaining_real_demands = self.real_demands.clone()
        self.observable_demands = self.base_demands.clone() 
        
        if 'vehicle_positions' in state_dict:
             self.vehicle_positions = state_dict['vehicle_positions'].clone().unsqueeze(0)
             self.vehicle_loads = state_dict['vehicle_loads'].clone().unsqueeze(0)
             self.visited = state_dict['visited'].clone().unsqueeze(0)
             self.steps = state_dict['steps']
        else:
             self.vehicle_positions = torch.zeros(1, self.num_vehicles, dtype=torch.long, device=self.device)
             self.vehicle_loads = torch.full((1, self.num_vehicles), self.capacity, device=self.device)
             self.visited = torch.zeros(1, self.num_nodes, dtype=torch.bool, device=self.device)
             self.visited[:, 0] = True
             self.steps = 0

        p = self.current_customer_positions[0]
        dists = torch.cdist(p, p, p=2)
        self.travel_costs = dists.unsqueeze(0) * 2.0

        if batch_size > 1:
            self.real_demands = self.real_demands.repeat(batch_size, 1)
            self.remaining_real_demands = self.remaining_real_demands.repeat(batch_size, 1)
            self.observable_demands = self.observable_demands.repeat(batch_size, 1)
            self.base_demands = self.base_demands.repeat(batch_size, 1)
            self.current_customer_positions = self.current_customer_positions.repeat(batch_size, 1, 1)
            self.travel_costs = self.travel_costs.repeat(batch_size, 1, 1)
            self.vehicle_positions = self.vehicle_positions.repeat(batch_size, 1)
            self.vehicle_loads = self.vehicle_loads.repeat(batch_size, 1)
            self.visited = self.visited.repeat(batch_size, 1)
            self.current_weather_vector = self.current_weather_vector.repeat(batch_size, 1)
    """

    def set_clonable_state(self, state_dict, batch_size=1, is_deterministic=False):
        # 1. Load Core Data
        self.current_customer_positions = state_dict['current_customer_positions'].clone().unsqueeze(0)
        self.base_demands = state_dict['base_demands'].clone().unsqueeze(0)
        self.current_weather_vector = state_dict['current_weather_vector'].clone().unsqueeze(0)
        
        if 'alpha' in state_dict:
            # Load 3x3 matrices: [Nodes, 3, 3] -> [1, Nodes, 3, 3]
            self.current_alpha = state_dict['alpha'].clone().unsqueeze(0)
        else:
            # Fallback for old datasets without alpha
            self.current_alpha = torch.zeros(1, self.num_nodes, self.weather_dim, self.weather_dim, device=self.device)

        # 2. Intelligent Demand Loading
        if is_deterministic:
            
            # A. What is the Target?
            self.real_demands = self.base_demands.clone()
            
            # B. How much have we delivered so far in the saved state?
            saved_total = state_dict['real_demands'].clone().unsqueeze(0)
            saved_rem = state_dict['remaining_real_demands'].clone().unsqueeze(0)
            amount_delivered = saved_total - saved_rem
            
            # C. Apply progress to the deterministic target
            self.remaining_real_demands = self.real_demands - amount_delivered
            
            # D. Safety Clamp (avoid negative due to float precision)
            self.remaining_real_demands = torch.clamp(self.remaining_real_demands, min=0.0)
            
            # E. Observable is what remains
            self.observable_demands = self.remaining_real_demands.clone()
            
        else:
            self.real_demands = state_dict['real_demands'].clone().unsqueeze(0)
            self.remaining_real_demands = state_dict['remaining_real_demands'].clone().unsqueeze(0)
            self.observable_demands = state_dict['observable_demands'].clone().unsqueeze(0)
        
        # 3. Load Dynamic State
        if 'vehicle_positions' in state_dict:
             self.vehicle_positions = state_dict['vehicle_positions'].clone().unsqueeze(0)
             self.vehicle_loads = state_dict['vehicle_loads'].clone().unsqueeze(0)
             self.visited = state_dict['visited'].clone().unsqueeze(0)
             self.steps = state_dict['steps']
        else:
             self.vehicle_positions = torch.zeros(1, self.num_vehicles, dtype=torch.long, device=self.device)
             self.vehicle_loads = torch.full((1, self.num_vehicles), self.capacity, device=self.device)
             self.visited = torch.zeros(1, self.num_nodes, dtype=torch.bool, device=self.device)
             self.visited[:, 0] = True
             self.steps = 0

        # 4. Rebuild Derived State
        p = self.current_customer_positions[0]
        dists = torch.cdist(p, p, p=2)
        self.travel_costs = dists.unsqueeze(0) * 2.0

        if batch_size > 1:
            self.real_demands = self.real_demands.repeat(batch_size, 1)
            self.remaining_real_demands = self.remaining_real_demands.repeat(batch_size, 1)
            self.observable_demands = self.observable_demands.repeat(batch_size, 1)
            self.base_demands = self.base_demands.repeat(batch_size, 1)
            self.current_customer_positions = self.current_customer_positions.repeat(batch_size, 1, 1)
            self.travel_costs = self.travel_costs.repeat(batch_size, 1, 1)
            self.vehicle_positions = self.vehicle_positions.repeat(batch_size, 1)
            self.vehicle_loads = self.vehicle_loads.repeat(batch_size, 1)
            self.visited = self.visited.repeat(batch_size, 1)
            self.current_weather_vector = self.current_weather_vector.repeat(batch_size, 1)

            if self.current_alpha is not None:
                self.current_alpha = self.current_alpha.repeat(batch_size, 1, 1, 1)