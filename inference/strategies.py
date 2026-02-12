import torch
import itertools
from env.vrp_env import VRPEnvironment 

class InferenceStrategy:
    def __init__(self, policy_model, device='cpu'):
        self.policy_model = policy_model
        self.device = device
        self.policy_model.eval()

    def solve(self, env: VRPEnvironment, initial_state=None, is_deterministic=False, **kwargs):
        raise NotImplementedError

class GreedyInference(InferenceStrategy):
    def solve(self, env: VRPEnvironment, initial_state=None, is_deterministic=False):
        if initial_state is not None:

            env.set_clonable_state(initial_state)
            
            if is_deterministic:
                env.real_demands = env.base_demands.clone()
                env.remaining_real_demands = env.base_demands.clone()
                env.observable_demands = env.base_demands.clone()
                
            customer_features, vehicle_features, demands = env._get_features()
        else:
            env.reset(batch_size=1, fixed_customers=True, is_deterministic=is_deterministic)
            
        hidden = None
        routes = [[] for _ in range(env.num_vehicles)]
        total_reward_acc = 0.0
        done = False
        step_count = 0
        max_steps = env.num_nodes * 3 

        while not done and step_count < max_steps:
            with torch.no_grad():
                current_vehicle_positions = env.vehicle_positions
                log_probs, hidden = self.policy_model(
                    customer_features, vehicle_features, demands, current_vehicle_positions, hidden
                )

            actions_tensor = self.policy_model.sample_action(log_probs, greedy=True) 
            actions = actions_tensor[0].tolist()

            for v_idx, action in enumerate(actions):
                routes[v_idx].append(action)

            (customer_features, vehicle_features, demands), rewards, done_tensor = env.step(actions_tensor)
            total_reward_acc += rewards.item()
            step_count += 1
            done = done_tensor.item()

        return routes, -total_reward_acc 

class RandomSamplingInference(InferenceStrategy):
    def solve(self, env: VRPEnvironment, num_samples=16, initial_state=None, is_deterministic=False):
        best_cost = float('inf')
        best_routes = None

        if initial_state is not None:
            start_state_dict = initial_state
        else:
            env.reset(batch_size=1, fixed_customers=True, is_deterministic=is_deterministic)
            start_state_dict = env.get_clonable_state()

        for sample_idx in range(num_samples):
            env.set_clonable_state(start_state_dict)
            
            if is_deterministic:
                env.real_demands = env.base_demands.clone()
                env.remaining_real_demands = env.base_demands.clone()
                env.observable_demands = env.base_demands.clone()
            
            customer_features, vehicle_features, demands = env._get_features()

            hidden = None
            routes = [[] for _ in range(env.num_vehicles)]
            current_total_reward_acc = 0.0
            done = False
            step_count = 0
            max_steps = env.num_nodes * 3 

            while not done and step_count < max_steps:
                with torch.no_grad():
                    current_vehicle_positions = env.vehicle_positions
                    log_probs, hidden = self.policy_model(
                        customer_features, vehicle_features, demands, current_vehicle_positions, hidden
                    )

                sampled_actions_tensor = self.policy_model.sample_action(log_probs, greedy=False)
                actions = sampled_actions_tensor[0].tolist()

                for v_idx, action in enumerate(actions):
                    routes[v_idx].append(action)

                (customer_features, vehicle_features, demands), rewards, done_tensor = env.step(sampled_actions_tensor)
                current_total_reward_acc += rewards.item()
                done = done_tensor.item()
                step_count += 1

            current_cost = -current_total_reward_acc
            if current_cost < best_cost:
                best_cost = current_cost
                best_routes = [r.copy() for r in routes]

        return best_routes, best_cost

class BeamSearchInference(InferenceStrategy):
    def solve(self, env: VRPEnvironment, beam_width=3, initial_state=None, is_deterministic=False):
        if initial_state is not None:
            env.set_clonable_state(initial_state)
            if is_deterministic:
                env.real_demands = env.base_demands.clone()
                env.remaining_real_demands = env.base_demands.clone()
                env.observable_demands = env.base_demands.clone()
        else:
            env.reset(batch_size=1, fixed_customers=True, is_deterministic=is_deterministic)
            
        initial_env_state_dict = env.get_clonable_state()
        
        beam = [(0.0, [[] for _ in range(env.num_vehicles)], initial_env_state_dict, None)]
        completed_hypotheses = [] 
        max_steps_beam = env.num_nodes * 3 

        for step_idx in range(max_steps_beam):
            if not beam: break
            next_beam_candidates = []

            for score_sum, routes_hist, current_hypothesis_env_state_dict, policy_hidden in beam:
                env.set_clonable_state(current_hypothesis_env_state_dict)
                
                c_feat, v_feat, d_feat = env._get_features()

                demands_met = torch.all(d_feat[0, 1:] <= 1e-3).item()
                at_depot = (env.vehicle_positions[0, 0].item() == 0)

                if demands_met and at_depot:
                    completed_hypotheses.append((score_sum, [r.copy() for r in routes_hist]))
                    continue
                
                if demands_met and not at_depot:
                    joint_action_nodes = [0] * env.num_vehicles
                    env.set_clonable_state(current_hypothesis_env_state_dict)
                    action_tensor = torch.tensor([joint_action_nodes], dtype=torch.long, device=self.device)
                    _, _, _ = env.step(action_tensor)
                    next_hypothesis_env_state_dict = env.get_clonable_state()
                    new_routes_hist = [r.copy() for r in routes_hist]
                    for v_idx in range(env.num_vehicles): new_routes_hist[v_idx].append(0)
                    next_beam_candidates.append((score_sum, new_routes_hist, next_hypothesis_env_state_dict, policy_hidden))
                    continue 

                with torch.no_grad():
                    current_vehicle_positions = env.vehicle_positions
                    log_probs_policy, new_policy_hidden = self.policy_model(
                        c_feat, v_feat, d_feat, current_vehicle_positions,  policy_hidden
                    )

                per_vehicle_top_k_choices = []
                for v_idx in range(env.num_vehicles):
                    log_probs_v = log_probs_policy[0, v_idx]
                    k_for_vehicle = min(beam_width, log_probs_v.size(0))
                    top_k_logprobs_v, top_k_indices_v = torch.topk(log_probs_v, k_for_vehicle)
                    per_vehicle_top_k_choices.append([(lp.item(), idx.item()) for lp, idx in zip(top_k_logprobs_v, top_k_indices_v)])

                for combined_choice_info in itertools.product(*per_vehicle_top_k_choices):
                    joint_action_nodes = [choice[1] for choice in combined_choice_info]
                    joint_action_log_prob_sum = sum(choice[0] for choice in combined_choice_info)

                    env.set_clonable_state(current_hypothesis_env_state_dict)
                    action_tensor = torch.tensor([joint_action_nodes], dtype=torch.long, device=self.device)
                    _, _, _ = env.step(action_tensor) 
                    
                    next_hypothesis_env_state_dict = env.get_clonable_state() 
                    new_routes_hist = [r.copy() for r in routes_hist]
                    for v_idx, node_action in enumerate(joint_action_nodes): new_routes_hist[v_idx].append(node_action)
                    next_beam_candidates.append((score_sum + joint_action_log_prob_sum, new_routes_hist, next_hypothesis_env_state_dict, new_policy_hidden))

            next_beam_candidates.sort(key=lambda x: x[0], reverse=True)
            beam = next_beam_candidates[:beam_width]

        candidates_to_evaluate = completed_hypotheses if completed_hypotheses else [b[:2] for b in beam]
        best_real_cost = float('inf')
        best_real_routes = [[] for _ in range(env.num_vehicles)]

        if not candidates_to_evaluate:
             return best_real_routes, float('inf')

        for _, candidate_routes in candidates_to_evaluate:
            # RESET TO INITIAL STATIC STATE
            env.set_clonable_state(initial_env_state_dict)
            # Re-apply Deterministic Override if needed
            if is_deterministic:
                env.real_demands = env.base_demands.clone()
                env.remaining_real_demands = env.base_demands.clone()
            
            max_len = 0
            for r in candidate_routes: max_len = max(max_len, len(r))
            
            accumulated_reward = 0.0
            done = False
            for s in range(max_len):
                if done: break
                actions_list = []
                for v in range(env.num_vehicles):
                    if s < len(candidate_routes[v]): actions_list.append(candidate_routes[v][s])
                    else: actions_list.append(0) 
                
                actions_tensor = torch.tensor([actions_list], dtype=torch.long, device=self.device)
                _, r, d = env.step(actions_tensor)
                accumulated_reward += r.item()
                done = d.item()

            cost = -accumulated_reward
            if cost < best_real_cost:
                best_real_cost = cost
                best_real_routes = candidate_routes

        return best_real_routes, best_real_cost