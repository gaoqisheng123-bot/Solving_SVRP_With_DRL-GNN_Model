# Solving Stochastic Vehicle Routing Problem (SVRP) using A Deep Reinforcement Learning and Graph Neural Network Model (DRL-GNN)

This project implements a Deep Reinforcement Learning (DRL) solution for the Stochastic Vehicle Routing Problem using PyTorch. It utilizes a **Graph Attention Network (GAT)** encoder and a **REINFORCE**-based policy gradient method to train an agent to route vehicles under uncertain demand and weather conditions.

## 📂 1. Project Structure

To ensure the Python imports work correctly, your directory structure **must** look like this.
```text
.
├── config.py
├── main.py
├── env/
│   └── vrp_env.py
├── models/
│   ├── baseline.py
│   ├── gat.py
│   └── policy.py
├── training/
│   └── reinforce.py
├── inference/
│   └── strategies.py
├── utils/ 
│   ├── logger_utils.py
│   ├── visualization_utils.py
│   └── data_utils.py
└── checkpoints/
```

## 📦 2. Installation

Ensure you have Python 3.8+ installed. Install the dependencies:

```bash
pip install torch numpy matplotlib tqdm
```

## 🏋️ 3. Training the Model

**Basic Training**

Train a model for a 10-node problem for 100 epochs:

```bash
python main.py --num_nodes 10 --epochs 100 --batch_size 32
```

If you have a CUDA-capable device:
```bash
python main.py --num_nodes 10 --epochs 100 --batch_size 32 --cuda
```

To train on larger instances (e.g., 50 nodes), you may need to adjust the batch size to fit in memory:
```bash
python main.py --num_nodes 10 --epochs 200 --batch_size 16 --cuda
```

Key Training Flags:
- --save_dir: Directory to save checkpoints.
- --lr: Learning rate for the policy network.
- --entropy_weight: Coefficient for entropy regularization (encourages exploration).
- --deterministic_env: If set, turns off weather noise (Real Demand = Base Demand).

Existing model parameter:
```bash
python main.py --num_nodes 10 --epochs 5000 --batch_size 512 --lr 1e-4 --baseline_lr 2e-4 --embedding_dim 256 --gat_layers 4 --gat_heads 8 --entropy_weight 0.05 --save_dir ./checkpoints_final_10_v5 --cuda
```

## 🧠 4. Inference & Evaluation

You can evaluate a trained model using different inference strategies. Use the --test_only flag and specify the model path with --load_model.
Examples are shown with the existing model:

**Strategy A: Greedy Inference**

This is the fastest method. It always selects the node with the highest probability output by the policy.
```bash
python main.py --test_only --load_model ./checkpoints_final_10_v5/model_final --num_nodes 10 --num_vehicles 1 --capacity 50 --embedding_dim 256 --gat_layers 4 --gat_heads 8 --inference greedy --test_size 50 --save_dir results/vrp_final_n10_v5_greedy --cuda
```

**Strategy B: Beam Search**

This maintains the top k most probable partial solutions at every step. It offers a balance between exploration and exploitation but is computationally expensive.
```bash
python main.py --test_only --load_model ./checkpoints_final_10_v5/model_final --num_nodes 10 --num_vehicles 1 --capacity 50 --embedding_dim 256 --gat_layers 4 --gat_heads 8 --inference beam --beam_width 5 --test_size 100 --save_dir results/vrp_final_n10_v5_beam5 --cuda
```

**Strategy C: Random Sampling**

This samples N different solutions from the policy's probability distribution and selects the one with the lowest cost. It is slower and computationally heavier than Greedy and Beam but usually finds better solutions with increasing N.
```bash
python main.py --test_only --load_model ./checkpoints_final_10_v5/model_final --num_nodes 10 --num_vehicles 1 --capacity 50 --embedding_dim 256 --gat_layers 4 --gat_heads 8 --num_samples 1000 --inference random --test_size 50 --save_dir results/vrp_final_n10_v5_random_eval --cuda
```

## 📊 5. Visualization

At the end of training, the script generates visualizations of the routes:
1. Training Metrics: checkpoints/training_metrics.png (Rewards and Losses over time).
2. Route Visualization: checkpoints/route_visualization_greedy.png (A plot of the chosen instance using Greedy Search will be shown).

At the evaluation stage, the script generates visualizations of the routes:
1. Route Visualization: results/vrp_final_n10_v5_{strategy}/route_visualization_{strategy}.png (A plot of the chosen strategy.)

## 🧩 6. Customization

You can adjust the neural network architecture during training, some examples would be:
1. --embedding_dim: Size of the node embeddings (default: 128).
2. --gat_layers: Depth of the Graph Attention Network encoder.
3. --gat_heads: Number of attention heads in the GAT.


## ⚠️ Troubleshooting

RuntimeError: CUDA out of memory
Reduce the --batch_size or --embedding_dim



## Main refences

```
@article{iklassov2023reinforcement,
  title={Reinforcement Learning for Solving Stochastic Vehicle Routing Problem},
  author={Iklassov, Zangir and Sobirov, Ikboljon and Solozabal, Ruben and Tak{\'a}{\v{c}}, Martin},
  journal={arXiv preprint arXiv:2311.07708},
  year={2023}
}

@article{nazari2018reinforcementvrp,
  title={Reinforcement learning for solving the vehicle routing problem},
  author={Nazari, Mohammadreza and Oroojlooy, Afshin and Snyder, Lawrence and Tak{\'a}c, Martin},
  journal={Advances in neural information processing systems},
  Advances in neural information processing systems
  year={2018}
}
```

## License

MIT
