import torch
import numpy as np
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt 

# Local imports
from config import parse_args
from env.vrp_env import VRPEnvironment
from models.policy import VRPPolicy
from training.reinforce import ReinforceTrainer
from inference.strategies import GreedyInference, RandomSamplingInference, BeamSearchInference
from utils.logger_utils import setup_logger
from utils.visualization_utils import visualize_route
from utils.data_utils import generate_validation_dataset, load_dataset 

def train(args, env, trainer, logger, val_dataset):
    logger.info(f"Starting training for {args.epochs} epochs with batch_size {args.batch_size}")
    
    rewards_history = []
    policy_losses_history = []
    baseline_losses_history = []
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        mean_reward, policy_loss, baseline_loss = trainer.train_episode(
            env=env,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            is_deterministic=args.deterministic_env
        )
        
        rewards_history.append(mean_reward)
        policy_losses_history.append(policy_loss)
        baseline_losses_history.append(baseline_loss)
        
        if epoch % args.log_interval == 0:
            logger.info(f"Epoch {epoch}/{args.epochs} | Avg Reward: {mean_reward:.3f} | "
                        f"Policy Loss: {policy_loss:.4f} | Baseline Loss: {baseline_loss:.4f} | "
                        f"Time: {time.time() - epoch_start_time:.2f}s")
        
        if epoch % args.save_interval == 0:
            save_path_prefix = os.path.join(args.save_dir, f"model_epoch_{epoch}")
            trainer.save_models(save_path_prefix)
            logger.info(f"Saved model to {save_path_prefix}_policy.pt")
            
            if args.test_size > 0: 
                 logger.info("Running intermediate evaluation...")
                 eval_subset = val_dataset[:20] if val_dataset else None
                 eval_reward = evaluate(args, env, trainer.policy_model, 
                                        num_instances=len(eval_subset) if eval_subset else 1, 
                                        logger=logger, 
                                        dataset=eval_subset, 
                                        is_interim_eval=True)
                 logger.info(f"Intermediate Eval Avg Reward: {eval_reward:.3f} (Cost: {-eval_reward:.3f})")
    
    final_model_path_prefix = os.path.join(args.save_dir, "model_final")
    trainer.save_models(final_model_path_prefix)
    logger.info(f"Saved final model to {final_model_path_prefix}_policy.pt")
    
    plot_metrics(rewards_history, policy_losses_history, baseline_losses_history, args.save_dir, logger)
    logger.info("Training completed.")

def plot_metrics(rewards, p_losses, b_losses, save_dir, logger):
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(rewards); plt.title('Episode Batch Rewards'); plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.plot(p_losses); plt.title('Policy Losses'); plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.plot(b_losses); plt.title('Baseline Losses'); plt.grid(True)
    plt.tight_layout()
    metrics_plot_path = os.path.join(save_dir, 'training_metrics.png')
    plt.savefig(metrics_plot_path)
    plt.close()
    logger.info(f"Saved training metrics plot to {metrics_plot_path}")


def evaluate(args, env, policy_model, num_instances, logger, dataset=None, is_interim_eval=False):
    eval_type_msg = "Intermediate" if is_interim_eval else "Final"
    logger.info(f"Starting {eval_type_msg} evaluation with {args.inference} strategy.")
    total_cost_acc = 0.0
    
    device = next(policy_model.parameters()).device
    
    if args.inference == 'greedy':
        inference_strategy = GreedyInference(policy_model, device=device)
    elif args.inference == 'random':
        inference_strategy = RandomSamplingInference(policy_model, device=device)
    elif args.inference == 'beam':
        inference_strategy = BeamSearchInference(policy_model, device=device)
    else:
        raise ValueError(f"Unknown inference strategy: {args.inference}")

    best_vis_routes = None
    best_vis_cost = float('inf')
    best_vis_idx = -1
    best_vis_env_state = None

    if dataset is not None:
        loop_range = range(min(num_instances, len(dataset)))
    else:
        loop_range = range(num_instances)

    # --- TIMER START ---
    start_time = time.time()

    
    for i in tqdm(loop_range, desc=f"{eval_type_msg} Evaluating"):
        initial_state = dataset[i] if dataset else None
        
        solve_params = {}
        if args.inference == 'random': solve_params['num_samples'] = args.num_samples
        if args.inference == 'beam': solve_params['beam_width'] = args.beam_width
        
        routes, cost = inference_strategy.solve(env=env, initial_state=initial_state, is_deterministic=args.deterministic_env, **solve_params)
        total_cost_acc += cost
        
        if i == 0:
            best_vis_cost = cost
            best_vis_routes = routes
            best_vis_idx = i
            
            # --- FIX FOR VISUALIZATION ---
            if initial_state:
                # We must clone it so we don't modify the dataset itself
                # Deep copy strategy manually to be safe
                state_copy = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in initial_state.items()}
                
                # If Deterministic Mode is ON, we must overwrite the state data 
                # BEFORE saving it for visualization.
                if args.deterministic_env:
                    state_copy['real_demands'] = state_copy['base_demands'].clone()
                    state_copy['remaining_real_demands'] = state_copy['base_demands'].clone()
                    # Optional: Zero out weather to make it clear
                    # state_copy['current_weather_vector'] = torch.zeros_like(state_copy['current_weather_vector'])
                
                best_vis_env_state = state_copy
            else:
                best_vis_env_state = env.get_clonable_state()
                
        if logger and ((is_interim_eval and i < 1) or (not is_interim_eval and i < 3)):
            logger.info(f"Instance {i}: Cost = {cost:.3f}")
            for v_idx, route in enumerate(routes):
                logger.info(f"  Vehicle {v_idx+1} Raw Route: {route}")
            
            if hasattr(env, 'remaining_real_demands') and env.remaining_real_demands is not None:
                if torch.any(env.remaining_real_demands[0, 1:] > 1e-3):
                    logger.warning(f"  Instance {i}: INCOMPLETE (Real demands not met).")
                else:
                    logger.info(f"  Instance {i}: COMPLETE.")
    
    # --- TIMER END ---
    end_time = time.time()
    total_time = end_time - start_time
    num_solved = len(loop_range)
    avg_time_per_instance = total_time / num_solved if num_solved > 0 else 0

    mean_cost = total_cost_acc / num_solved if num_solved > 0 else 0
    
    # Log Time stats
    logger.info(f"{eval_type_msg} evaluation completed.")
    logger.info(f"  Mean Cost: {mean_cost:.3f}")
    logger.info(f"  Total Time: {total_time:.2f}s")
    logger.info(f"  Avg Time/Instance: {avg_time_per_instance:.4f}s")
    
    if not is_interim_eval and best_vis_routes is not None:
        vis_title = f"Instance {best_vis_idx} - {args.inference} - Cost: {best_vis_cost:.2f}"
        vis_save_path = os.path.join(args.save_dir, f"route_visualization_{args.inference}.png")
        
        if best_vis_env_state:
            env.set_clonable_state(best_vis_env_state)
            
        visualize_route(env, best_vis_routes, logger, title=vis_title, save_path=vis_save_path)
    
    return -mean_cost
"""


def evaluate(args, env, policy_model, num_instances, logger, dataset=None, is_interim_eval=False):
    eval_type_msg = "Intermediate" if is_interim_eval else "Final"
    logger.info(f"Starting {eval_type_msg} evaluation with {args.inference} strategy.")
    total_cost_acc = 0.0
    
    device = next(policy_model.parameters()).device
    
    if args.inference == 'greedy':
        inference_strategy = GreedyInference(policy_model, device=device)
    elif args.inference == 'random':
        inference_strategy = RandomSamplingInference(policy_model, device=device)
    elif args.inference == 'beam':
        inference_strategy = BeamSearchInference(policy_model, device=device)
    else:
        raise ValueError(f"Unknown inference strategy: {args.inference}")

    # Track Global Best (Lowest Cost)
    # Note: solve() returns Positive Cost (Distance + Penalties), so lower is better.
    global_best_cost = float('inf')
    global_best_routes = None
    global_best_idx = -1
    global_best_env_state = None

    if dataset is not None:
        loop_range = range(min(num_instances, len(dataset)))
    else:
        loop_range = range(num_instances)

    start_time = time.time()

    for i in tqdm(loop_range, desc=f"{eval_type_msg} Evaluating"):
        initial_state = dataset[i] if dataset else None
        
        solve_params = {}
        if args.inference == 'random': solve_params['num_samples'] = args.num_samples
        if args.inference == 'beam': solve_params['beam_width'] = args.beam_width
        
        # Solve
        # Note: We pass args.deterministic_env here to ensure solver uses correct mode
        routes, cost = inference_strategy.solve(
            env=env, 
            initial_state=initial_state, 
            is_deterministic=args.deterministic_env, 
            **solve_params
        )
        
        total_cost_acc += cost
        
        # --- TRACK BEST INSTANCE ---
        # We look for the MINIMUM cost (best performance)
        if cost < global_best_cost:
            global_best_cost = cost
            global_best_routes = routes
            global_best_idx = i
            
            # Capture state for visualization
            if initial_state:
                # Deep copy to safe-guard against modification
                state_copy = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in initial_state.items()}
                
                # If in Deterministic Mode, force the visualization data to match
                if args.deterministic_env:
                    state_copy['real_demands'] = state_copy['base_demands'].clone()
                    state_copy['remaining_real_demands'] = state_copy['base_demands'].clone()
                
                global_best_env_state = state_copy
            else:
                global_best_env_state = env.get_clonable_state()

        # Logging (Optional: Log first few instances for debug)
        if logger and ((is_interim_eval and i < 1) or (not is_interim_eval and i < 3)):
            logger.info(f"Instance {i}: Cost = {cost:.3f}")

    # Stats
    end_time = time.time()
    total_time = end_time - start_time
    num_solved = len(loop_range)
    avg_time_per_instance = total_time / num_solved if num_solved > 0 else 0
    mean_cost = total_cost_acc / num_solved if num_solved > 0 else 0
    
    logger.info(f"{eval_type_msg} evaluation completed.")
    logger.info(f"  Mean Cost: {mean_cost:.3f}")
    logger.info(f"  Total Time: {total_time:.2f}s")
    logger.info(f"  Avg Time/Instance: {avg_time_per_instance:.4f}s")
    
    # VISUALIZE THE WINNER
    if not is_interim_eval and global_best_routes is not None:
        vis_title = f"BEST (Inst {global_best_idx}) - {args.inference} - Cost: {global_best_cost:.2f}"
        vis_save_path = os.path.join(args.save_dir, f"route_visualization_{args.inference}.png")
        
        if global_best_env_state:
            env.set_clonable_state(global_best_env_state)
            
        visualize_route(env, global_best_routes, logger, title=vis_title, save_path=vis_save_path)
    
    return -mean_cost
"""
def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    logger = setup_logger(args.save_dir)
    logger.info(f"Running with arguments: {args}")
    logger.info(f"Using device: {device}")
    
    env = VRPEnvironment(
        num_nodes=args.num_nodes,
        num_vehicles=args.num_vehicles,
        capacity=args.capacity,
        device=device,
        weather_dim=3 
    )
    
    policy_model = VRPPolicy(
        customer_input_dim=env.customer_features_dim,
        vehicle_input_dim=env.vehicle_features_dim,
        embedding_dim=args.embedding_dim,
        n_heads=args.gat_heads,
        n_layers=args.gat_layers
    ).to(device)
    
    trainer = ReinforceTrainer(
        policy_model=policy_model,
        customer_features_dim=env.customer_features_dim,
        vehicle_features_dim=env.vehicle_features_dim,
        embedding_dim=args.embedding_dim,
        lr=args.lr,
        baseline_lr=args.baseline_lr,
        entropy_weight=args.entropy_weight,
        device=device
    )
    
    # --- FIXED: DATASET MANAGEMENT ---
    # Create a SHARED data directory independent of save_dir
    common_data_dir = './data'
    os.makedirs(common_data_dir, exist_ok=True)
    
    # File name includes num_nodes so N=5 and N=10 don't conflict
    val_dataset_path = os.path.join(common_data_dir, f'validation_dataset_n{args.num_nodes}.pkl')
    val_dataset = None

    # Try to load existing data
    if os.path.exists(val_dataset_path):
        val_dataset = load_dataset(val_dataset_path, device)
        logger.info(f"Loaded SHARED validation dataset from {val_dataset_path}")
    else:
        # If not found, generate it (This happens only ONCE per problem size)
        logger.info("No shared dataset found. Generating new one...")
        num_val = 1000 if args.test_size > 0 else 50
        val_dataset = generate_validation_dataset(
            num_instances=num_val,
            num_nodes=args.num_nodes,
            num_vehicles=args.num_vehicles,
            capacity=args.capacity,
            weather_dim=3,
            save_path=val_dataset_path,
            device=device
        )
        logger.info(f"Moving generated dataset to {device}...")
        for state in val_dataset:
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        # --- ADD THIS BLOCK END ---

        logger.info(f"Generated and saved SHARED static dataset with {len(val_dataset)} instances.")

    # --- LOAD MODEL WEIGHTS ---
    if args.load_model:
        try:
            trainer.load_models(args.load_model)
            logger.info(f"Loaded model weights from {args.load_model}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return
    
    if not args.test_only:
        train(args, env, trainer, logger, val_dataset)
    
    if args.test_only and not args.load_model:
        logger.warning("Testing with random weights (no model loaded).")

    if args.test_size > 0:
        logger.info(f"Running final evaluation ({args.inference})...")
        evaluate(args, env, policy_model, args.test_size, logger, dataset=val_dataset)
    else:
        logger.info("Skipping final evaluation.")

if __name__ == "__main__":
    main()