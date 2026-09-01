import tracemalloc
import time
import numpy as np
import networkx as nx
import csv
import random
import torch
_old_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _old_load(*args, **kwargs)
torch.load = safe_load
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.utils import to_networkx
import scipy.sparse as sp

def measure_peak_memory(func, *args, **kwargs):
    tracemalloc.start()
    try:
        func(*args, **kwargs)
    except MemoryError:
        pass
    except Exception as e:
        print(f"Error during memory measure: {e}")
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return peak / (1024 * 1024 * 1024) # Return in GB

def naive_er(G):
    # O(V^3) operation
    # Converting to dense matrix for pseudoinverse requires O(V^2) memory and O(V^3) time
    L = nx.laplacian_matrix(G).todense()
    _ = np.linalg.pinv(L)
    
def cser_filter(G):
    # O(V+E) k-core
    _ = nx.k_core(G, k=2)

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
    dataset = NodePropPredDataset(name='ogbn-arxiv', root='/tmp/ogbn-arxiv')
    graph, _ = dataset[0]
    edge_index = graph['edge_index']
    
    G_full = nx.Graph()
    edges = [(edge_index[0][i], edge_index[1][i]) for i in range(edge_index.shape[1])]
    G_full.add_edges_from(edges)
    
    sample_sizes = [1000, 2000, 4000, 8000, 16000]
    results = []
    
    for N in sample_sizes:
        print(f"--- Profiling N={N} ---")
        G_sub = create_subgraph(G_full, N)
        
        # Profile CSER k-core
        print("  Profiling CSER...")
        cser_mem = measure_peak_memory(cser_filter, G_sub)
        
        # Profile Naive ER
        print("  Profiling Naive ER...")
        naive_mem = measure_peak_memory(naive_er, G_sub)
        
        print(f"  Naive ER: {naive_mem:.4f} GB | CSER: {cser_mem:.6f} GB")
        results.append({
            'N_nodes': N,
            'naive_er_gb': naive_mem,
            'cser_gb': cser_mem
        })
        
    with open('memory_scaling.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['N_nodes', 'naive_er_gb', 'cser_gb'])
        writer.writeheader()
        writer.writerows(results)
        
    print("Saved to memory_scaling.csv")

if __name__ == '__main__':
    main()
