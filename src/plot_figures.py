import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14

def plot_budget_sweep():
    with open("results/sweeps.json", "r") as f:
        data = json.load(f)["budgets"]
        
    budgets = [float(k) * 100 for k in sorted(data.keys(), key=float)]
    l2_vals = [data[k]["l2"] for k in sorted(data.keys(), key=float)]
    acc_vals = [data[k]["acc"] * 100 for k in sorted(data.keys(), key=float)]
    
    fig, ax1 = plt.subplots(figsize=(6, 4))
    
    color = 'tab:red'
    ax1.set_xlabel('Rewiring Budget $\\tau$ (%)', fontsize=14)
    ax1.set_ylabel('Spectral Gap ($\\lambda_2$)', color=color, fontsize=14)
    ax1.plot(budgets, l2_vals, color=color, marker='o', linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('GCN Test Accuracy (%)', color=color, fontsize=14)
    ax2.plot(budgets, acc_vals, color=color, marker='s', linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('The Budget Trade-Off on CiteSeer', fontsize=16)
    fig.tight_layout()
    plt.savefig('budget_sweep.pdf', format='pdf', bbox_inches='tight')
    plt.close()

def plot_log_complexity():
    with open("results/ogb_timing.json", "r") as f:
        data = json.load(f)
        
    sizes = data["subgraph_sizes"]
    naive_times = data["pseudoinverse_times"]
    core_time = data["k_core_time"]
    
    plt.figure(figsize=(6, 4))
    
    # Plot empirical Naive ER points
    plt.plot(sizes, naive_times, 'ro-', linewidth=2, markersize=8, label='Naive ER $O(|V|^3)$')
    
    # Extrapolate
    x_extrap = np.linspace(sizes[-1], 20000, 100)
    # y = c * x^3 -> c = y[-1] / x[-1]^3
    c = naive_times[-1] / (sizes[-1]**3)
    y_extrap = c * (x_extrap**3)
    
    plt.plot(x_extrap, y_extrap, 'r--', linewidth=2, alpha=0.5, label='Cubic Extrapolation')
    
    # CSER Time (Constant line for full 169k graph to show it's instant)
    plt.axhline(y=core_time, color='b', linestyle='-', linewidth=2, label=f'CSER Filter (169k nodes)')
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel('Graph Size $|V|$', fontsize=14)
    plt.ylabel('Execution Time (seconds)', fontsize=14)
    plt.title('Computational Scaling (ogbn-arxiv)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('complexity_loglog.pdf', format='pdf', bbox_inches='tight')
    plt.close()

def plot_hijack():
    G = nx.Graph()
    # Cluster 1
    G.add_edges_from([(1,2), (2,3), (1,3), (1,4), (2,4), (3,4)])
    # Cluster 2
    G.add_edges_from([(5,6), (6,7), (5,7), (5,8), (6,8), (7,8)])
    # Bottleneck
    G.add_edge(4, 5)
    # Leaf
    G.add_edge(2, 9)
    
    pos = {
        1: (0, 1), 2: (0, -1), 3: (-1, 0), 4: (1, 0),
        5: (3, 0), 6: (4, 1), 7: (4, -1), 8: (5, 0),
        9: (-1, -2)
    }
    
    plt.figure(figsize=(7, 4))
    
    # Compute true ER
    L = nx.laplacian_matrix(G).toarray()
    L_pinv = np.linalg.pinv(L)
    def er(u, v):
        u, v = u-1, v-1
        return L_pinv[u,u] + L_pinv[v,v] - 2*L_pinv[u,v]
    
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', edgecolors='black', node_size=500)
    
    # Draw edges, color bottleneck and leaf
    edge_colors = []
    edge_labels = {}
    for u, v in G.edges():
        r = er(u, v)
        if (u,v) == (4,5) or (v,u) == (4,5):
            edge_colors.append('tab:red')
            edge_labels[(u,v)] = f'R={r:.2f}\nBottleneck'
        elif (u,v) == (2,9) or (v,u) == (2,9):
            edge_colors.append('tab:blue')
            edge_labels[(u,v)] = f'R={r:.2f}\nLeaf Hijack'
        else:
            edge_colors.append('gray')
            
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=3)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, font_color='black')
    
    plt.axis('off')
    plt.title('Topological Leaf-Node Hijacking', fontsize=16)
    plt.tight_layout()
    plt.savefig('hijack_topology.pdf', format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_budget_sweep()
    plot_log_complexity()
    plot_hijack()
    print("Figures generated.")
