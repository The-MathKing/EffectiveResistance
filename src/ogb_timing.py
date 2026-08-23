import time
import numpy as np
import networkx as nx
from ogb.nodeproppred import NodePropPredDataset
import json
import os

def run_ogb_timing():
    print("Loading ogbn-arxiv...")
    dataset = NodePropPredDataset(name='ogbn-arxiv', root='/tmp/ogb')
    graph, _ = dataset[0]
    
    edges = graph['edge_index']
    G_raw = nx.Graph()
    G_raw.add_edges_from(edges.T)
    
    print(f"Full ogbn-arxiv graph: {G_raw.number_of_nodes()} nodes, {G_raw.number_of_edges()} edges")
    
    print("\n--- Benchmarking k-Core Pruning (O(|V| + |E|)) ---")
    start = time.time()
    G_core = nx.k_core(G_raw, k=2)
    end = time.time()
    core_time = end - start
    print(f"2-Core execution time on full 169k graph: {core_time:.4f} seconds")
    print(f"Core graph size: {G_core.number_of_nodes()} nodes")
    
    print("\n--- Benchmarking Naive ER (Laplacian Pseudoinverse O(|V|^3)) ---")
    sizes = [1000, 2000, 4000, 8000]
    inversion_times = []
    
    nodes_list = list(G_raw.nodes())
    for n in sizes:
        print(f"  Sampling {n} nodes...")
        sampled = np.random.choice(nodes_list, size=n, replace=False)
        sub_g = G_raw.subgraph(sampled)
        
        L = nx.laplacian_matrix(sub_g).toarray()
        t0 = time.time()
        _ = np.linalg.pinv(L)
        t1 = time.time()
        elapsed = t1 - t0
        inversion_times.append(elapsed)
        print(f"  Pseudoinverse for N={n}: {elapsed:.4f} seconds")
        
    os.makedirs("results", exist_ok=True)
    results = {
        "full_nodes": G_raw.number_of_nodes(),
        "k_core_time": core_time,
        "subgraph_sizes": sizes,
        "pseudoinverse_times": inversion_times
    }
    with open("results/ogb_timing.json", "w") as f:
        json.dump(results, f, indent=2)
        
if __name__ == "__main__":
    run_ogb_timing()
