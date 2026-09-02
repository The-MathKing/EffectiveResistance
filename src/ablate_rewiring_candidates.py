import torch
import numpy as np
import networkx as nx
from torch_geometric.datasets import WebKB
from torch_geometric.utils import to_networkx

def get_top_k_nonedges(G, k):
    # compute effective resistance for all non-edges
    L = nx.laplacian_matrix(G).toarray()
    L_pinv = np.linalg.pinv(L)
    
    non_edges = list(nx.non_edges(G))
    resistances = []
    for u, v in non_edges:
        r = L_pinv[u, u] + L_pinv[v, v] - 2 * L_pinv[u, v]
        resistances.append((r, u, v))
        
    resistances.sort(reverse=True, key=lambda x: x[0])
    return resistances[:k]

def analyze_candidates(dataset_name):
    dataset = WebKB(root='/tmp/' + dataset_name, name=dataset_name)
    data = dataset[0]
    G = to_networkx(data, to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    
    # get LCC
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0]).copy()
    
    # 2-core
    G_core = nx.k_core(G, k=2)
    core_nodes = set(G_core.nodes())
    periphery_nodes = set(G.nodes()) - core_nodes
    
    tau = 0.05
    k = int(tau * G.number_of_edges())
    
    top_k = get_top_k_nonedges(G, k)
    
    touches_periphery = 0
    for r, u, v in top_k:
        if u in periphery_nodes or v in periphery_nodes:
            touches_periphery += 1
            
    print(f"{dataset_name}: Top {k} candidate non-edges (raw ER).")
    print(f"Number touching peripheral nodes: {touches_periphery} ({touches_periphery/k*100:.1f}%)")

if __name__ == '__main__':
    analyze_candidates('Texas')
    analyze_candidates('Cornell')
