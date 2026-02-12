import torch
import torch.optim as optim
from models.baseline import BaselineModel 

class ReinforceTrainer:
    def __init__(self, 
                 policy_model, 
                 customer_features_dim, 
                 vehicle_features_dim,  
                 embedding_dim,         
                 lr=1e-4,
                 baseline_lr=1e-3,
                 entropy_weight=0.01,
                 device='cpu'):
        self.policy_model = policy_model
        self.device = device
        self.entropy_weight = entropy_weight
        
        self.optimizer = optim.Adam(policy_model.parameters(), lr=lr)
        
        # Reduces LR if loss stops improving for 50 epochs
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=50, verbose=True
        )
        
        baseline_input_dim = customer_features_dim + vehicle_features_dim
        self.baseline_model = BaselineModel(input_dim=baseline_input_dim, hidden_dim=embedding_dim).to(device)
        self.baseline_optimizer = optim.Adam(self.baseline_model.parameters(), lr=baseline_lr)
        
    def train_episode(self, env, batch_size=32, max_steps=100, is_deterministic=False):
        log_probs_list = []       
        entropies_list = []       
        rewards_list = []         
        baseline_values_list = [] 
        mask_list = []            
        
        customer_features, vehicle_features, demands = env.reset(batch_size=batch_size, fixed_customers=False, is_deterministic=is_deterministic)
        hidden_state = None 
        done_mask_for_batch = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for step in range(max_steps):
            if done_mask_for_batch.all(): break
            
            current_vehicle_positions = env.vehicle_positions
            log_probs_policy, hidden_state = self.policy_model(
                customer_features, vehicle_features, demands, current_vehicle_positions, hidden_state
            )
            actions_taken = self.policy_model.sample_action(log_probs_policy, greedy=False)
            
            batch_idx = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, env.num_vehicles)
            vehicle_idx = torch.arange(env.num_vehicles, device=self.device).unsqueeze(0).expand(batch_size, -1)
            action_log_probs = log_probs_policy[batch_idx, vehicle_idx, actions_taken]
            
            probs_policy = torch.exp(log_probs_policy) 
            entropies = -torch.sum(probs_policy * log_probs_policy, dim=-1) 
            mean_entropies_per_instance = entropies.mean(dim=1) 
            
            baseline_value = self.baseline_model(customer_features, vehicle_features).squeeze(-1) 
            
            (next_customer_features, next_vehicle_features, next_demands), \
            step_rewards, step_done_flags = env.step(actions_taken)
            
            log_probs_list.append(action_log_probs) 
            entropies_list.append(mean_entropies_per_instance)
            rewards_list.append(step_rewards)
            baseline_values_list.append(baseline_value)
            mask_list.append(~done_mask_for_batch) 
            
            customer_features, vehicle_features, demands = next_customer_features, next_vehicle_features, next_demands
            done_mask_for_batch = done_mask_for_batch | step_done_flags
        
        if not log_probs_list: return 0.0, 0.0, 0.0

        log_probs_tensor = torch.stack(log_probs_list)      
        entropies_tensor = torch.stack(entropies_list)      
        rewards_tensor = torch.stack(rewards_list)          
        baseline_tensor = torch.stack(baseline_values_list) 
        mask_tensor = torch.stack(mask_list).float()      
        
        returns = self._compute_returns(rewards_tensor, mask_tensor) 
        advantages = returns - baseline_tensor.detach()
        
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        policy_loss = self._compute_policy_loss(log_probs_tensor, entropies_tensor, advantages, mask_tensor)
        baseline_loss = self._compute_baseline_loss(baseline_tensor, returns, mask_tensor)
        
        if torch.isnan(policy_loss) or torch.isinf(policy_loss):
            print("Warning: Policy Loss is NaN/Inf. Skipping step.")
            return 0.0, 0.0, 0.0
            
        if torch.isnan(baseline_loss) or torch.isinf(baseline_loss):
            print("Warning: Baseline Loss is NaN/Inf. Skipping step.")
            return 0.0, 0.0, 0.0
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0) 
        self.optimizer.step()
        
        self.baseline_optimizer.zero_grad()
        baseline_loss.backward()
        self.baseline_optimizer.step()
        
        actual_rewards_sum = (rewards_tensor * mask_tensor).sum(dim=0) 
        mean_batch_reward = actual_rewards_sum.mean().item()
        
        self.scheduler.step(mean_batch_reward)
        
        return mean_batch_reward, policy_loss.item(), baseline_loss.item()
    

    def _compute_returns(self, rewards, mask, gamma=0.99):
        episode_length, batch_size = rewards.size()
        returns = torch.zeros_like(rewards, device=self.device)
        future_return = torch.zeros(batch_size, device=self.device)
        for t in reversed(range(episode_length)):
            future_return = rewards[t] + gamma * future_return * mask[t]
            returns[t] = future_return
        return returns
    
    def _compute_policy_loss(self, log_probs_actions, entropies, advantages, mask_steps):
        summed_log_probs_per_step = log_probs_actions.sum(dim=2)
        num_valid_steps_total = mask_steps.sum()
        if num_valid_steps_total == 0: return torch.tensor(0.0, device=self.device)
        policy_gradient_term = - (summed_log_probs_per_step * advantages.detach() * mask_steps).sum()
        policy_loss = policy_gradient_term / num_valid_steps_total
        entropy_regularization_term = (entropies * mask_steps).sum()
        entropy_loss = -self.entropy_weight * entropy_regularization_term / num_valid_steps_total
        return policy_loss + entropy_loss
    
    def _compute_baseline_loss(self, baseline_values, returns, mask_steps):
        num_valid_steps_total = mask_steps.sum()
        if num_valid_steps_total == 0: return torch.tensor(0.0, device=self.device)
        loss = ((baseline_values - returns.detach())**2 * mask_steps).sum() / num_valid_steps_total
        return loss
    
    def save_models(self, path_prefix):
        torch.save(self.policy_model.state_dict(), f"{path_prefix}_policy.pt")
        torch.save(self.baseline_model.state_dict(), f"{path_prefix}_baseline.pt")
        
    def load_models(self, path_prefix):
        self.policy_model.load_state_dict(torch.load(f"{path_prefix}_policy.pt", map_location=self.device))
        self.baseline_model.load_state_dict(torch.load(f"{path_prefix}_baseline.pt", map_location=self.device))