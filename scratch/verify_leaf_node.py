import networkx as nx
import torch
import json
from torch_geometric.datasets import WebKB
from torch_geometric.utils import to_networkx
import numpy as np
from metrics import effective_resistance
from jacobian import SimpleGCN, compute_jacobian_norms
from mosr import calculate_mosr

dataset = WebKB(root='/tmp/WebKB', name='Texas')
data = dataset[0]
G_full = to_networkx(data, to_undirected=True)
largest_cc = max(nx.connected_components(G_full), key=len)
G = G_full.subgraph(largest_cc).copy()
G.remove_edges_from(nx.selfloop_edges(G))

num_edges = G.number_of_edges()
degrees = dict(G.degree())
deg_1_nodes = [n for n, d in degrees.items() if d == 1]
cut_edges = []
for u, v in G.edges():
    if degrees[u] == 1 or degrees[v] == 1:
        cut_edges.append((u, v))

print(f"Texas LCC Edges: {num_edges}")
print(f"Degree-1 Nodes: {len(deg_1_nodes)}")
print(f"Cut-Edges (connected to deg-1): {len(cut_edges)}")
print(f"Percentage of total edges: {len(cut_edges) / num_edges * 100:.2f}%")

# Compute ER for these cut edges
r_vals = []
for u, v in cut_edges:
    r = effective_resistance(G, u, v)
    r_vals.append(r)
print(f"Min R on cut-edges: {min(r_vals):.6f}")
print(f"Max R on cut-edges: {max(r_vals):.6f}")

# Check threshold logic
# We need to compute ER for all edges to see where the threshold lands
all_res = {e: effective_resistance(G, *e) for e in G.edges()}
# Add reverse edges because calculate_mosr iterates over G_sub.edges() which might be ordered differently
all_res.update({(v, u): r for (u, v), r in all_res.items()})

print(f"Total ER == 1.0 edges (bidirectional): {sum(1 for v in all_res.values() if abs(v - 1.0) < 1e-9)}")
# The thresholding in calculate_mosr looks at the metric dictionary.
# Wait, for ER, we used negative ER in test_cora:
res_curvs = {e: -r for e, r in all_res.items()}

# Now we need the jaco norms to check MOSR
torch.manual_seed(42)
jaco_norms_avg = {e: 0.0 for e in G.edges()}
jaco_norms_avg.update({(v, u): 0.0 for u, v in G.edges()})
runs = 3
for run in range(runs):
    model = SimpleGCN(in_channels=dataset.num_features, hidden_channels=64, num_layers=2)
    model.eval()
    jaco = compute_jacobian_norms(G, model, feature_dim=dataset.num_features)
    for e, val in jaco.items():
        if e in jaco_norms_avg:
            jaco_norms_avg[e] += val / runs

# Run 2-core pruning experiment
# 2-core removes all nodes of degree < 2 iteratively
G_core = nx.k_core(G, 2)
print(f"\n2-Core Texas Nodes: {G_core.number_of_nodes()}, Edges: {G_core.number_of_edges()}")

ebc_core = nx.edge_betweenness_centrality(G_core)
p67_core = np.percentile(list(ebc_core.values()), 67)
ebc_undirected_core = {}
for (u, v), val in ebc_core.items():
    ebc_undirected_core[(u, v)] = val
    ebc_undirected_core[(v, u)] = val

high_betw_core_edges = [(u, v) for u, v in G_core.edges() if ebc_undirected_core[(u, v)] > p67_core]
G_sub_core = nx.Graph()
G_sub_core.add_edges_from(high_betw_core_edges)

# Compute ER on 2-core
res_curvs_core = {}
for u, v in G_core.edges():
    r = effective_resistance(G_core, u, v)
    res_curvs_core[(u, v)] = -r
    res_curvs_core[(v, u)] = -r

score, num, denom = calculate_mosr(G_core, G_sub_core, res_curvs_core, jaco_norms_avg, q=25)
print(f"2-Core ER High-Betweenness q=25 MOSR: {score:.4f} ({num}/{denom})")
