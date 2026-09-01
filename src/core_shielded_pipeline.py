import torch
import networkx as nx
from torch_geometric.datasets import WebKB
from torch_geometric.utils import to_networkx
import numpy as np
import argparse

from metrics import effective_resistance
from jacobian import SimpleGCN, compute_jacobian_norms

def run_core_shielded_pipeline(dataset_name, q=25):
    print(f"\n--- Running Core-Shielded Effective Resistance on {dataset_name} ---")
    dataset = WebKB(root='/tmp/WebKB', name=dataset_name)
    
    # 1. Load Raw Graph
    G_raw = to_networkx(dataset[0], to_undirected=True)
    largest_cc = max(nx.connected_components(G_raw), key=len)
    G_raw = G_raw.subgraph(largest_cc).copy()
    
    # Prune self-loops which crash k-core
    G_raw.remove_edges_from(nx.selfloop_edges(G_raw))
    print(f"Raw LCC Graph: {G_raw.number_of_nodes()} nodes, {G_raw.number_of_edges()} edges")
    
    # Compute ground truth (Jacobian norms on raw graph to find true structural bottlenecks)
    torch.manual_seed(42)
    jaco_norms_avg = {e: 0.0 for e in G_raw.edges()}
    runs = 10
    for _ in range(runs):
        model = SimpleGCN(in_channels=dataset.num_features, hidden_channels=64, num_layers=2)
        model.eval()
        jaco = compute_jacobian_norms(G_raw, model, feature_dim=dataset.num_features)
        for e, val in jaco.items():
            if e in jaco_norms_avg:
                jaco_norms_avg[e] += val / runs
                
    j_global = list(jaco_norms_avg.values())
    j_threshold = np.percentile(j_global, q)
    
    # Identify True High-Betweenness Bottlenecks
    ebc = nx.edge_betweenness_centrality(G_raw)
    p67 = np.percentile(list(ebc.values()), 67)
    
    true_bottlenecks = []
    for u, v in G_raw.edges():
        if ebc[(u, v)] > p67:
            if jaco_norms_avg[(u, v)] <= j_threshold:
                true_bottlenecks.append((u, v))
                
    print(f"Total True High-Betweenness Bottlenecks (q={q}): {len(true_bottlenecks)}")
    
    # 2. ALGORITHMIC SOLUTION: The k=2 Core Filter (O(V+E))
    G_core = nx.k_core(G_raw, k=2)
    print(f"2-Core Filtered Graph: {G_core.number_of_nodes()} nodes, {G_core.number_of_edges()} edges")
    print(f"Nodes Pruned: {G_raw.number_of_nodes() - G_core.number_of_nodes()}")
    print(f"Edges Pruned (Trivial Leaves): {G_raw.number_of_edges() - G_core.number_of_edges()}")
    
    # 3. Compute Spectral Metric on the Shielded Core (O(V^3))
    print("Computing Core-Shielded Effective Resistance (CSER)...")
    cser_vals = {}
    for u, v in G_core.edges():
        cser_vals[(u, v)] = effective_resistance(G_core, u, v)
        
    c_threshold = np.percentile([-v for v in cser_vals.values()], q) # negative for bottleneck
    
    # Evaluate MOSR specifically on the True Bottlenecks that survived the core
    missed = 0
    evaluated = 0
    for u, v in true_bottlenecks:
        if G_core.has_edge(u, v):
            evaluated += 1
            val = cser_vals[(u, v)]
            if -val > c_threshold: # Not flagged
                missed += 1
                
    mosr = missed / evaluated if evaluated > 0 else 0
    print(f"\n[RESULTS] Core-Shielded MOSR: {missed}/{evaluated} missed ({mosr:.4f})")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Texas', choices=['Texas', 'Cornell'])
    args = parser.parse_args()
    run_core_shielded_pipeline(args.dataset)
