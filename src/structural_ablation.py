import networkx as nx
import numpy as np
import scipy.sparse.linalg as sla
from torch_geometric.datasets import WebKB
from torch_geometric.utils import to_networkx

def get_er_stats(G):
    if G.number_of_nodes() == 0:
        return 0.0, 0
    L = nx.laplacian_matrix(G).toarray()
    L_pinv = np.linalg.pinv(L)
    
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    max_r = 0.0
    r_eq_1 = 0
    
    # Evaluate ER on existing edges (as per the bottleneck detection protocol)
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        r = L_pinv[i, i] + L_pinv[j, j] - 2 * L_pinv[i, j]
        if r > max_r:
            max_r = r
        # Tolerance for float precision
        if r >= 0.9999:
            r_eq_1 += 1
            
    return max_r, r_eq_1

def remove_leaves(G):
    G_new = G.copy()
    leaves = [n for n, d in G_new.degree() if d == 1]
    while leaves:
        G_new.remove_nodes_from(leaves)
        leaves = [n for n, d in G_new.degree() if d == 1]
    return G_new

def main():
    dataset = WebKB(root='/tmp/Texas', name='Texas')
    data = dataset[0]
    G_raw = to_networkx(data, to_undirected=True)
    largest_cc = max(nx.connected_components(G_raw), key=len)
    G_raw = G_raw.subgraph(largest_cc).copy()
    G_raw.remove_edges_from(nx.selfloop_edges(G_raw))
    
    print("--- Structural Core Ablation on Texas ---")
    
    print(f"\n1. Raw Graph: {G_raw.number_of_nodes()} nodes, {G_raw.number_of_edges()} edges")
    max_r, r_1 = get_er_stats(G_raw)
    print(f"   Max Edge ER: {max_r:.4f}, Cut-Edges (R=1): {r_1}")
    
    G_leaf = remove_leaves(G_raw)
    print(f"\n2. Leaf-Removed (1-core iterative): {G_leaf.number_of_nodes()} nodes, {G_leaf.number_of_edges()} edges")
    max_r, r_1 = get_er_stats(G_leaf)
    print(f"   Max Edge ER: {max_r:.4f}, Cut-Edges (R=1): {r_1}")
    
    G_2core = nx.k_core(G_raw, k=2)
    print(f"\n3. 2-Core: {G_2core.number_of_nodes()} nodes, {G_2core.number_of_edges()} edges")
    max_r, r_1 = get_er_stats(G_2core)
    print(f"   Max Edge ER: {max_r:.4f}, Cut-Edges (R=1): {r_1}")
    
    G_3core = nx.k_core(G_raw, k=3)
    print(f"\n4. 3-Core: {G_3core.number_of_nodes()} nodes, {G_3core.number_of_edges()} edges")
    max_r, r_1 = get_er_stats(G_3core)
    print(f"   Max Edge ER: {max_r:.4f}, Cut-Edges (R=1): {r_1}")

if __name__ == '__main__':
    main()
