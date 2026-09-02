import time
import networkx as nx
import numpy as np
import scipy.sparse as sp
import csv
import os
import torch
_old_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _old_load(*args, **kwargs)
torch.load = safe_load
from ogb.nodeproppred import NodePropPredDataset

def create_subgraph(G, target_nodes):
    nodes = set([0])
    queue = [0]
    while queue and len(nodes) < target_nodes:
        curr = queue.pop(0)
        for neighbor in G.neighbors(curr):
            if neighbor not in nodes:
                nodes.add(neighbor)
                queue.append(neighbor)
            if len(nodes) >= target_nodes:
                break
    return G.subgraph(nodes).copy()

def main():
    print("Loading ogbn-arxiv...")
    dataset = NodePropPredDataset(name='ogbn-arxiv', root='/tmp/ogb')
    graph = dataset[0] 
    
    edge_index = graph[0]['edge_index']
    
    G_full = nx.Graph()
    edges = [(edge_index[0][i], edge_index[1][i]) for i in range(edge_index.shape[1])]
    G_full.add_edges_from(edges)
    
    sizes = [1000, 2000, 4000, 8000, 16000]
    results = []
    
    for N in sizes:
        print(f"--- Benchmarking N={N} ---")
        G_sub = create_subgraph(G_full, N)
        actual_N = G_sub.number_of_nodes()
        
        # 1. CSER Pipeline time
        start_cser = time.time()
        G_core = nx.k_core(G_sub, k=2)
        kcore_time = time.time() - start_cser
        
        cser_pinv_time = 0
        if G_core.number_of_nodes() > 0:
            L_core = nx.laplacian_matrix(G_core).toarray()
            start_pinv = time.time()
            _ = np.linalg.pinv(L_core)
            cser_pinv_time = time.time() - start_pinv
            
        cser_total_time = kcore_time + cser_pinv_time
        v_core_frac = G_core.number_of_nodes() / actual_N if actual_N > 0 else 0
        
        # 2. Naive ER (dense pinv of Laplacian) time
        L = nx.laplacian_matrix(G_sub).toarray()
        
        start_naive = time.time()
        L_pinv = np.linalg.pinv(L)
        naive_time = time.time() - start_naive
        
        print(f"N={actual_N} | Naive ER: {naive_time:.4f}s | CSER Total: {cser_total_time:.4f}s (k-core: {kcore_time:.4f}, pinv: {cser_pinv_time:.4f}) | V_core/V: {v_core_frac:.4f}")
        results.append({
            'N_nodes': actual_N,
            'Naive_ER_time_sec': naive_time,
            'CSER_time_sec': cser_total_time,
            'V_core_fraction': v_core_frac
        })
        
    with open('benchmark_scaling.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['N_nodes', 'Naive_ER_time_sec', 'CSER_time_sec', 'V_core_fraction'])
        writer.writeheader()
        writer.writerows(results)
        
    print("Done. Saved to benchmark_scaling.csv")

if __name__ == '__main__':
    main()
