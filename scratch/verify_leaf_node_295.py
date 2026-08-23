import networkx as nx
import torch
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

# Note: We do NOT remove self-loops here, so this perfectly matches dump_json.py
num_edges = G.number_of_edges()

# Count cut-edges connected to degree-1 nodes
degrees = dict(G.degree())
deg_1_nodes = [n for n, d in degrees.items() if d == 1]
cut_edges = []
for u, v in G.edges():
    if u != v and (degrees[u] == 1 or degrees[v] == 1):
        cut_edges.append((u, v))

# Compute ER for all edges to see where the threshold lands
all_res = {e: effective_resistance(G, *e) for e in G.edges() if e[0] != e[1]}
er_1_edges = [e for e, r in all_res.items() if abs(r - 1.0) < 1e-9]

print("=== EXACT MATCH WITH DUMP_JSON.PY (No Self-Loop Pruning) ===")
print(f"Texas LCC Total Edges: {num_edges}")
print(f"Degree-1 Nodes: {len(deg_1_nodes)}")
print(f"Cut-Edges (connected to deg-1): {len(cut_edges)}")
print(f"Percentage of total edges (budget saturation): {len(cut_edges) / num_edges * 100:.2f}%")
print(f"Total directional edges with ER == 1.0: {len(er_1_edges) * 2}") 
print("==============================================================\n")

# To run 2-core, we MUST remove self-loops because nx.k_core forbids them
G_no_loops = G.copy()
G_no_loops.remove_edges_from(nx.selfloop_edges(G_no_loops))

# 2-core removes all nodes of degree < 2 iteratively
G_core = nx.k_core(G_no_loops, 2)
print("=== 2-CORE PRUNING EXPERIMENT ===")
print(f"2-Core Texas Nodes: {G_core.number_of_nodes()}, Edges: {G_core.number_of_edges()}")

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

# We need the jaco norms to check MOSR
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

score, num, denom = calculate_mosr(G_core, G_sub_core, res_curvs_core, jaco_norms_avg, q=25)
print(f"2-Core ER High-Betweenness q=25 MOSR: {score:.4f} ({num}/{denom})")
print("=================================")
