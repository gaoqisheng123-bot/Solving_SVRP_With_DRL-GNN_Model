import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='VRP-RL')
    
    # Environment settings
    parser.add_argument('--num_nodes', type=int, default=20, help='Number of nodes (customers + depot)')
    parser.add_argument('--num_vehicles', type=int, default=1, help='Number of vehicles')
    parser.add_argument('--capacity', type=float, default=50.0, help='Vehicle capacity')
    parser.add_argument('--fixed_customers', action='store_true', default=True, help='Use fixed customer positions for training and eval')
    
    # Model settings
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--gat_heads', type=int, default=4, help='Number of heads for GAT layers')
    parser.add_argument('--gat_layers', type=int, default=2, help='Number of GAT layers in the encoder')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (episode batches)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training episodes')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for policy')
    parser.add_argument('--baseline_lr', type=float, default=1e-3, help='Learning rate for baseline')
    parser.add_argument('--entropy_weight', type=float, default=0.01, help='Entropy regularization weight')
    parser.add_argument('--max_steps', type=int, default=None, help='Maximum steps per episode in training (default: num_nodes * 3 from env)')
    
    # Inference settings
    parser.add_argument('--inference', type=str, default='greedy', choices=['greedy', 'random', 'beam'], help='Inference strategy')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples for random sampling inference')
    parser.add_argument('--beam_width', type=int, default=3, help='Beam width for beam search inference')
    parser.add_argument('--test_size', type=int, default=10, help='Number of test instances for evaluation')
    
    # Other settings
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save models and logs')
    parser.add_argument('--load_model', type=str, default=None, help='Path prefix to load model from (e.g., checkpoints/model_epoch_10)')
    parser.add_argument('--test_only', action='store_true', help='Test mode (no training, requires --load_model)')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval for training epochs')
    parser.add_argument('--save_interval', type=int, default=20, help='Save interval for models during training (epochs)')
    parser.add_argument('--deterministic_env', action='store_true', help='If set, environment removes weather noise (Real Demand = Base Demand)')
    


    args = parser.parse_args()
    if args.max_steps is None:
        args.max_steps = args.num_nodes * 3 # Default from env if not specified
    return args