import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import WebKB, Planetoid, WikipediaNetwork, Amazon
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
import scipy.sparse as sp
import scipy.linalg as spl

def get_dataset(name):
    transform = T.NormalizeFeatures()
    if name in ['Texas', 'Cornell', 'Wisconsin']:
        return WebKB(root='/tmp/' + name, name=name, transform=transform)[0]
    elif name in ['Cora', 'CiteSeer']:
        return Planetoid(root='/tmp/' + name, name=name, transform=transform)[0]
    elif name in ['Chameleon', 'Squirrel']:
        return WikipediaNetwork(root='/tmp/' + name, name=name, transform=transform)[0]
    elif name == 'Photo':
        return Amazon(root='/tmp/' + name, name=name, transform=transform)[0]
    return None

def compute_all_er(G):
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G).todense()
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    L_pinv = spl.pinvh(L)
    
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    
    er_scores = []
    for u, v in G.edges():
        i, j = node_idx[u], node_idx[v]
        r = L_pinv[i, i] + L_pinv[j, j] - 2 * L_pinv[i, j]
        er_scores.append((u, v, float(r)))
    return er_scores

def main():
    datasets = ['Texas', 'Cornell', 'Wisconsin', 'Cora', 'CiteSeer', 'Chameleon', 'Squirrel', 'Photo']
    q_vals = [5, 10, 15, 20, 25, 30, 40]
    
    plt.figure(figsize=(8, 6))
    plt.style.use('ggplot')
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    
    for idx, dname in enumerate(datasets):
        print(f"Processing {dname}...")
        data = get_dataset(dname)
        G_raw = to_networkx(data, to_undirected=True)
        lcc = max(nx.connected_components(G_raw), key=len)
        G = G_raw.subgraph(lcc).copy()
        G.remove_edges_from(nx.selfloop_edges(G))
        
        bridges = list(nx.bridges(G))
        bridge_set = set([(u,v) for u,v in bridges] + [(v,u) for u,v in bridges])
        pendant_fraction = len(bridges) / G.number_of_edges()
        
        er_scores = compute_all_er(G)
        er_scores.sort(key=lambda x: x[2], reverse=True)
        
        mosr_curve = []
        for q in q_vals:
            budget = max(1, int(len(er_scores) * (q / 100.0)))
            top_edges = er_scores[:budget]
            
            cut_edge_count = sum(1 for u, v, _ in top_edges if (u,v) in bridge_set or (v,u) in bridge_set)
            fraction_cut = cut_edge_count / budget
            mosr_curve.append(fraction_cut)
            
        plt.plot(q_vals, mosr_curve, label=dname, color=colors[idx], marker='o', linewidth=2)
        plt.axvline(x=pendant_fraction * 100, color=colors[idx], linestyle='--', alpha=0.5)

    plt.xlabel('Global Rank Threshold $q$ (Percentile)')
    plt.ylabel('Fraction of Budget Wasted on Cut-Edges')
    plt.title('Empirical $q$-Sweep: Cut-Edge Saturation Across 8 Datasets')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('empirical_q_sweep.pdf', dpi=300)
    print("Saved empirical_q_sweep.pdf")

if __name__ == '__main__':
    main()
