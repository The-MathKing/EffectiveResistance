import networkx as nx
import numpy as np
from torch_geometric.datasets import WebKB
from torch_geometric.utils import to_networkx
from jacobian import SimpleGCN, compute_jacobian_norms
import torch

def evaluate_er(dataset_name):
    print(f"\n========== {dataset_name} ==========")
    dataset = WebKB(root='/tmp/WebKB', name=dataset_name)
    data = dataset[0]
    G = to_networkx(data, to_undirected=True)
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

    # 1. Edge betweenness
    ebc = nx.edge_betweenness_centrality(G)
    ebc_undirected = {}
    for (u, v), val in ebc.items():
        ebc_undirected[(u, v)] = val
        ebc_undirected[(v, u)] = val

    ebc_vals = list(ebc.values())
    p33, p67 = np.percentile(ebc_vals, 33), np.percentile(ebc_vals, 67)
    bridges = [(u, v) for u, v in G.edges() if ebc_undirected[(u, v)] > p67]

    # 2. Compute ER
    er_dict = {}
    for u, v in G.edges():
        r = nx.resistance_distance(G, u, v)
        er_dict[(u, v)] = r
        er_dict[(v, u)] = r

    er_values = [er_dict[(u, v)] for u, v in G.edges()]
    
    # 3. Analyze R=1.0 and degree-1 nodes
    eps = 1e-5
    r_eq_1 = [(u, v) for u, v in G.edges() if abs(er_dict[(u, v)] - 1.0) < eps]
    print(f"Total edges: {G.number_of_edges()}")
    print(f"Edges with ER approx 1.0 (cut-edges): {len(r_eq_1)} ({len(r_eq_1)/G.number_of_edges()*100:.1f}%)")

    degree_1_nodes = [n for n, d in G.degree() if d == 1]
    edges_touching_d1 = [(u, v) for u, v in G.edges() if G.degree(u) == 1 or G.degree(v) == 1]
    print(f"Number of degree-1 nodes: {len(degree_1_nodes)}")
    print(f"Number of edges touching a degree-1 node: {len(edges_touching_d1)}")
    
    # 4. Tie-breaking and thresholding at q=25
    q = 25
    neg_er = [-r for r in er_values]
    threshold = np.percentile(neg_er, q)
    print(f"q=25 Threshold for -ER: {threshold:.4f} (ER >= {-threshold:.4f})")
    
    flagged = [(u, v) for u, v in G.edges() if -er_dict[(u, v)] <= threshold]
    print(f"Total flagged by `c <= threshold` rule: {len(flagged)} edges ({(len(flagged)/G.number_of_edges())*100:.1f}% of total)")

    # 5. True structural bridges (next highest resistance)
    er_unique = sorted(list(set([round(r, 6) for r in er_values])), reverse=True)
    next_highest_er = er_unique[1] if len(er_unique) > 1 else None
    print(f"Highest ER: {er_unique[0]}, Next highest ER: {next_highest_er}")
    
    if next_highest_er is not None:
        next_highest_edges = [(u, v) for u, v in G.edges() if abs(er_dict[(u, v)] - next_highest_er) < eps]
        print(f"Sample of next-highest ER edges (ER={next_highest_er}):")
        for e in next_highest_edges[:3]:
            print(f"  Edge {e}: Betweenness={ebc_undirected[e]:.4f}")

    # 6. Sanity check: re-evaluate MOSR after removing leaf nodes (degree-1)
    # We prune iteratively until no degree-1 nodes exist (2-core)
    G_no_loops = G.copy()
    G_no_loops.remove_edges_from(nx.selfloop_edges(G_no_loops))
    G_core = nx.k_core(G_no_loops, k=2)
    print(f"\n--- After pruning leaf nodes (2-core) ---")
    print(f"Core edges: {G_core.number_of_edges()}")
    
    # Recompute ER on core
    er_core = {}
    for u, v in G_core.edges():
        r = nx.resistance_distance(G_core, u, v)
        er_core[(u, v)] = r
        er_core[(v, u)] = r
        
    core_er_vals = [er_core[(u, v)] for u, v in G_core.edges()]
    if len(core_er_vals) > 0:
        print(f"Core ER: Min {min(core_er_vals):.4f}, Max {max(core_er_vals):.4f}, Mean {np.mean(core_er_vals):.4f}")
    
    # Recompute Jacobian on core
    model = SimpleGCN(in_channels=dataset.num_features, hidden_channels=64, num_layers=2)
    model.eval()
    jaco = compute_jacobian_norms(G_core, model, feature_dim=dataset.num_features)
    jaco_core_avg = {e: 0.0 for e in G_core.edges()}
    jaco_core_avg.update({(v, u): 0.0 for u, v in G_core.edges()})
    
    runs = 3
    for _ in range(runs):
        j_iter = compute_jacobian_norms(G_core, model, feature_dim=dataset.num_features)
        for e, val in j_iter.items():
            if e in jaco_core_avg:
                jaco_core_avg[e] += val / runs

    # Core betweenness
    ebc_core = nx.edge_betweenness_centrality(G_core)
    ebc_core_undirected = {}
    for (u, v), val in ebc_core.items():
        ebc_core_undirected[(u, v)] = val
        ebc_core_undirected[(v, u)] = val
        
    p67_core = np.percentile(list(ebc_core.values()), 67)
    core_bridges = [(u, v) for u, v in G_core.edges() if ebc_core_undirected[(u, v)] > p67_core]
    
    # MOSR on Core Bridges (High Betweenness)
    from mosr import calculate_mosr
    G_core_sub = nx.Graph()
    G_core_sub.add_edges_from(core_bridges)
    
    if len(core_bridges) > 0:
        # We pass -ER because mosr looks for low values
        neg_er_core = {e: -val for e, val in er_core.items()}
        mosr_core = calculate_mosr(G_core, G_core_sub, neg_er_core, jaco_core_avg, q=25)
        print(f"MOSR of ER on Core Bridges (q=25): {mosr_core:.4f}")
    else:
        print("No core bridges found.")

if __name__ == "__main__":
    evaluate_er('Texas')
    evaluate_er('Cornell')
