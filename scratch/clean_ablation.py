import networkx as nx
import torch
import numpy as np
from torch_geometric.datasets import WebKB
from torch_geometric.utils import to_networkx
from metrics import effective_resistance
from jacobian import SimpleGCN, compute_jacobian_norms
from mosr import calculate_mosr

dataset = WebKB(root='/tmp/WebKB', name='Texas')
data = dataset[0]
G_full = to_networkx(data, to_undirected=True)
largest_cc = max(nx.connected_components(G_full), key=len)
G = G_full.subgraph(largest_cc).copy()

num_edges = G.number_of_edges()
degrees = dict(G.degree())

# 1. Identify leaf-adjacent edges vs. nx.bridges
leaf_adj_edges = set()
for u, v in G.edges():
    if u != v and (degrees[u] == 1 or degrees[v] == 1):
        leaf_adj_edges.add(frozenset((u, v)))

bridges = list(nx.bridges(G))
bridge_sets = {frozenset((u, v)) for u, v in bridges}

print(f"Total nx.bridges: {len(bridge_sets)}")
print(f"Total leaf-adjacent edges: {len(leaf_adj_edges)}")

non_leaf_bridges = bridge_sets - leaf_adj_edges
print(f"Non-leaf bridges: {len(non_leaf_bridges)}")

# 2. Check betweenness of non-leaf bridges
ebc = nx.edge_betweenness_centrality(G)
# ebc returns (u, v) tuples. We need a way to look up frozenset
ebc_frozen = {frozenset((u, v)): val for (u, v), val in ebc.items()}

# Get the original 47 high-betweenness edges (Top 33%)
p67 = np.percentile(list(ebc.values()), 67)
original_high_betw = {fset for fset, val in ebc_frozen.items() if val > p67}

print(f"Original High-Betweenness edge count (denominator): {len(original_high_betw)}")

non_leaf_bridge_high_betw = non_leaf_bridges.intersection(original_high_betw)
print(f"Number of non-leaf bridges that are in the High-Betweenness stratum: {len(non_leaf_bridge_high_betw)}")
for fset in non_leaf_bridge_high_betw:
    print(f"  - Edge {list(fset)}, Betweenness: {ebc_frozen[fset]:.4f}")

# 3. Clean Ablation: Exclude leaf-adjacent edges from the top-25% budget on the original graph
# Get ER for all edges
all_res = {frozenset((u, v)): effective_resistance(G, u, v) for u, v in G.edges() if u != v}

# Let's see what happens if we just score the original 47 edges using the full graph ER, but removing leaf_adj from the budget.
# original q=25 logic flags the top 25% of all edges. 
# total edges = 295. 25% = 73.75 -> 74 edges flagged.
# If we exclude leaf_adj_edges (63) from being flagged, we flag the top 74 edges out of the remaining 232 edges.
eligible_edges = {fset: val for fset, val in all_res.items() if fset not in leaf_adj_edges}
# Sort eligible edges by highest resistance (lowest negative resistance)
sorted_eligible = sorted(eligible_edges.items(), key=lambda x: x[1], reverse=True)
budget = int(np.ceil(num_edges * 0.25))
flagged_edges = {fset for fset, val in sorted_eligible[:budget]}

# Now, out of the original 47 high-betweenness edges, which ones are true bottlenecks?
# We need Jacobian norms
torch.manual_seed(42)
jaco_norms_avg = {frozenset((u, v)): 0.0 for u, v in G.edges() if u != v}
runs = 3
for run in range(runs):
    model = SimpleGCN(in_channels=dataset.num_features, hidden_channels=64, num_layers=2)
    model.eval()
    jaco = compute_jacobian_norms(G, model, feature_dim=dataset.num_features)
    for (u, v), val in jaco.items():
        if u != v:
            fset = frozenset((u, v))
            if fset in jaco_norms_avg:
                jaco_norms_avg[fset] += val / runs

# True bottlenecks are the lowest 25% of Jacobian norms in the whole graph
sorted_jaco = sorted(jaco_norms_avg.items(), key=lambda x: x[1])
true_bottlenecks_full = {fset for fset, val in sorted_jaco[:budget]}

# The 47 high-betw true bottlenecks
true_bottlenecks_high_betw = true_bottlenecks_full.intersection(original_high_betw)
print(f"Number of true bottlenecks in original High-Betw stratum: {len(true_bottlenecks_high_betw)}")

# How many did the cleaned ER miss?
missed = true_bottlenecks_high_betw - flagged_edges
print(f"Cleaned Ablation MOSR (excluding leaves from budget): {len(missed) / len(true_bottlenecks_high_betw):.4f} ({len(missed)}/{len(true_bottlenecks_high_betw)})")

# Let's also check what 2-core pruning actually did to the original 47
G_no_loops = G.copy()
G_no_loops.remove_edges_from(nx.selfloop_edges(G_no_loops))
G_core = nx.k_core(G_no_loops, 2)

surviving_high_betw = set()
for fset in original_high_betw:
    u, v = list(fset)
    if G_core.has_edge(u, v):
        surviving_high_betw.add(fset)
print(f"Original High-Betw edges surviving 2-core pruning: {len(surviving_high_betw)} / {len(original_high_betw)}")
