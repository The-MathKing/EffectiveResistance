import networkx as nx
import torch
import numpy as np
from graphs import build_dense_Gc, build_connected_dense_Gc
from jacobian import compute_jacobian_norms, SimpleGCN
import random

def test_synthetic(graph_func, name):
    print(f"\n--- {name} ---")
    for n, m in [(2, 2), (5, 5), (10, 10)]:
        G = graph_func(n, m)
        
        # 3 seeds
        j_norms_list = []
        for seed in range(3):
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            model = SimpleGCN(num_layers=2, normalize=True) # Normalized proxy
            j_norms = compute_jacobian_norms(G, model)
            j_norms_list.append(j_norms)
            
        edge_flows = {}
        for (u, v) in j_norms_list[0].keys():
            edge_flows[(u, v)] = np.mean([j[(u, v)] for j in j_norms_list])
            
        st_flow = edge_flows.get(('s', 't')) or edge_flows.get(('t', 's'))
        
        flows = []
        for u, v in G.edges():
            flows.append(edge_flows[(u, v)])
        
        flows.sort() # ascending: smallest flow is rank 1 (most squashed)
        
        rank = flows.index(st_flow) + 1
        percentile = rank / len(flows) * 100
        
        print(f"n={n}, m={m} | (s,t) Flow: {st_flow:.5e} | Rank: {rank}/{len(flows)} ({percentile:.1f}th percentile)")

if __name__ == '__main__':
    test_synthetic(build_dense_Gc, "Experiment A (Dense G_c)")
    test_synthetic(build_connected_dense_Gc, "Experiment B (Connected Dense G_c)")
