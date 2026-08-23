import networkx as nx
import numpy as np
from torch_geometric.datasets import WebKB
from torch_geometric.utils import to_networkx

dataset = WebKB(root='/tmp/WebKB', name='Cornell')
G = to_networkx(dataset[0], to_undirected=True)
largest_cc = max(nx.connected_components(G), key=len)
G = G.subgraph(largest_cc).copy()

ebc = nx.edge_betweenness_centrality(G)
p33, p67 = np.percentile(list(ebc.values()), [33, 67])

bridges = [(u, v) for u, v in G.edges() if ebc.get((u, v), ebc.get((v, u))) > p67]
intra = [(u, v) for u, v in G.edges() if ebc.get((u, v), ebc.get((v, u))) <= p33]

print("Texas ER distribution")
er_all = [nx.resistance_distance(G, u, v) for u, v in G.edges()]
er_bridges = [nx.resistance_distance(G, u, v) for u, v in bridges]
er_intra = [nx.resistance_distance(G, u, v) for u, v in intra]

print(f"All ER: Min {min(er_all):.4f}, Max {max(er_all):.4f}, Mean {np.mean(er_all):.4f}")
print(f"Bridges ER: Min {min(er_bridges):.4f}, Max {max(er_bridges):.4f}, Mean {np.mean(er_bridges):.4f}")
print(f"Intra ER: Min {min(er_intra):.4f}, Max {max(er_intra):.4f}, Mean {np.mean(er_intra):.4f}")

q = 25
threshold = np.percentile([-r for r in er_all], q)
print(f"Global threshold for -ER (q={q}): {threshold:.4f} (meaning ER >= {-threshold:.4f} is flagged)")
flagged_all = sum(1 for r in er_all if -r <= threshold)
flagged_bridges = sum(1 for r in er_bridges if -r <= threshold)
print(f"Flagged all: {flagged_all} / {len(er_all)}")
print(f"Flagged bridges: {flagged_bridges} / {len(er_bridges)}")

from jacobian import SimpleGCN, compute_jacobian_norms
import torch
model = SimpleGCN(in_channels=dataset.num_features, hidden_channels=64, num_layers=2)
model.eval()
jaco = compute_jacobian_norms(G, model, feature_dim=dataset.num_features)
jaco_all = [jaco.get((u, v), jaco.get((v, u))) for u, v in G.edges()]
j_threshold = np.percentile(jaco_all, q)
print(f"Global threshold for Jacobian (q={q}): {j_threshold:.4f} (meaning Jaco <= {j_threshold:.4f} is true bottleneck)")

jaco_bridges = [jaco.get((u, v), jaco.get((v, u))) for u, v in bridges]
true_bridges = sum(1 for j in jaco_bridges if j <= j_threshold)
print(f"True bottlenecks in bridges: {true_bridges} / {len(jaco_bridges)}")
