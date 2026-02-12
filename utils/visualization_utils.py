import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_route(env, routes, logger, title="Route Visualization", save_path=None):
    """
    Visualizes VRP routes with a data panel sorted by VISIT SEQUENCE.
    Shows remaining capacity AND Total Distance Travelled.
    """
    # Create figure: Map (Left) and Data (Right)
    fig, (ax_map, ax_data) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1.5, 1]})
    
    # --- 1. GET DATA FROM ENV ---
    depot_pos = env.current_customer_positions[0, 0].cpu().numpy()
    customer_pos = env.current_customer_positions[0, 1:].cpu().numpy()
    
    # Stochastic Data
    weather = env.current_weather_vector[0].cpu().numpy()
    base_demands = env.base_demands[0].cpu().numpy()
    real_demands = env.real_demands[0].cpu().numpy()
    max_capacity = env.capacity
    
    # --- 2. LEFT PANEL: MAP ---
    # Plot Depot
    ax_map.scatter(depot_pos[0], depot_pos[1], c='red', s=200, marker='s', label='Depot', zorder=5)
    # Plot Customers
    ax_map.scatter(customer_pos[:, 0], customer_pos[:, 1], c='skyblue', s=100, label='Customers', zorder=4)
    
    # Annotate IDs
    for i, pos in enumerate(customer_pos):
        # Customer ID is index + 1
        ax_map.text(pos[0], pos[1] + 0.02, str(i + 1), fontsize=9, ha='center', fontweight='bold')

    # Draw Routes & Calculate Distance
    colors = plt.cm.get_cmap('tab10', len(routes))
    total_distance_all_vehicles = 0.0
    
    for v_idx, route in enumerate(routes):
        route_coords = [depot_pos]
        
        for node_idx in route:
            if node_idx == 0:
                coord = depot_pos
            else:
                coord = customer_pos[node_idx - 1]
            route_coords.append(coord)
            
        route_coords = np.array(route_coords)
        
        # Calculate Distance for this vehicle
        # Euclidean distance between consecutive points
        diffs = route_coords[1:] - route_coords[:-1]
        segment_dists = np.linalg.norm(diffs, axis=1)
        vehicle_dist = np.sum(segment_dists)
        total_distance_all_vehicles += vehicle_dist
        
        # Plot
        ax_map.plot(route_coords[:, 0], route_coords[:, 1], 
                    c=colors(v_idx), linewidth=2, label=f'Veh {v_idx+1}')
        
        # Add arrows
        if len(route_coords) > 1:
            mid = len(route_coords) // 2
            start, end = route_coords[mid], route_coords[mid+1]
            ax_map.arrow(start[0], start[1], (end[0]-start[0])*0.5, (end[1]-start[1])*0.5,
                         head_width=0.02, fc=colors(v_idx), ec=colors(v_idx))

    ax_map.set_title(title)
    ax_map.legend(loc='upper right')
    ax_map.grid(True, alpha=0.5)

    # --- 3. RIGHT PANEL: DYNAMIC TABLE ---
    ax_data.axis('off')
    
    # Calculate Load Stats
    total_real_demand = np.sum(real_demands)
    
    # Header Info (Updated with Distance)
    info_text = (
        f"--- STOCHASTIC DATA ---\n"
        f"Weather: {np.round(weather, 2)}\n\n"
        f"Vehicle Stats:\n"
        f"  Max Capacity : {max_capacity}\n"
        f"  Total Demand : {total_real_demand:.1f}\n"
        f"  Total Dist.  : {total_distance_all_vehicles:.4f}\n"
    )
    
    # Dynamic Table Simulation
    # Columns: Sequence | Node ID | Base | Real | Remaining Cap
    table_header = f"{'Seq':<3} | {'ID':<3} | {'Base':<5} | {'Real':<5} | {'Rem.Cap'}\n"
    table_header += "-"*45 + "\n"
    table_rows = ""
    
    seq_counter = 1
    
    # Track visited to find unvisited later
    visited_set = set()

    for v_idx, route in enumerate(routes):
        current_load = max_capacity # Reset load for new vehicle
        
        # Add a separator for vehicle start
        table_rows += f">> Veh {v_idx+1} Start (Load: {current_load})\n"
        
        for node_idx in route:
            if node_idx == 0:
                # Depot Visit -> Refill
                current_load = max_capacity
                table_rows += f"{'--':<3} | {'0':<3} | {'-':<5} | {'-':<5} | {current_load:<7.1f} (Refill)\n"
            else:
                # Customer Visit
                visited_set.add(node_idx)
                
                base = base_demands[node_idx]
                real = real_demands[node_idx]
                
                # Simulate delivery
                current_load -= real
                
                # Check for capacity violation in visualization
                cap_str = f"{current_load:.1f}"
                if current_load < 0:
                    cap_str += " (!)" # Warning marker
                
                row_str = f"{seq_counter:<3} | {node_idx:<3} | {base:<5.1f} | {real:<5.1f} | {cap_str:<7}\n"
                table_rows += row_str
                seq_counter += 1

    # Check for Unvisited Nodes
    unvisited = []
    for i in range(1, env.num_nodes):
        if i not in visited_set:
            unvisited.append(i)
            
    if unvisited:
        table_rows += "\n[!] UNVISITED NODES:\n"
        for i in unvisited:
            base = base_demands[i]
            real = real_demands[i]
            table_rows += f" -- | {i:<3} | {base:<5.1f} | {real:<5.1f} | MISSED\n"

    # Add text to panel
    ax_data.text(0.0, 1.0, info_text, transform=ax_data.transAxes, 
                 fontsize=11, family='monospace', fontweight='bold', va='top')
    
    # Adjusted vertical position for table to make room for header
    ax_data.text(0.0, 0.75, table_header + table_rows, transform=ax_data.transAxes, 
                 fontsize=10, family='monospace', va='top')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()