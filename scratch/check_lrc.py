import networkx as nx
import numpy as np
from torch_geometric.datasets import WebKB
from torch_geometric.utils import to_networkx
from metrics import link_resistance_curvature

dataset = WebKB(root='/tmp/WebKB', name='Cornell')
data = dataset[0]
G = to_networkx(data, to_undirected=True)
largest_cc = max(nx.connected_components(G), key=len)
G = G.subgraph(largest_cc).copy()

ebc = nx.edge_betweenness_centrality(G)
ebc_vals = list(ebc.values())
p33 = np.percentile(ebc_vals, 33)
p67 = np.percentile(ebc_vals, 67)

ebc_undirected = {}
for (u, v), val in ebc.items():
    ebc_undirected[(u, v)] = val
    ebc_undirected[(v, u)] = val

intra = [(u, v) for u, v in G.edges() if ebc_undirected[(u, v)] <= p33]
bridges = [(u, v) for u, v in G.edges() if ebc_undirected[(u, v)] > p67]

lrc_intra = [link_resistance_curvature(G, u, v) for u, v in intra]
lrc_bridges = [link_resistance_curvature(G, u, v) for u, v in bridges]

print("Intra-cluster LRC:")
print(f"  Count: {len(lrc_intra)}")
print(f"  Negative count: {sum(1 for c in lrc_intra if c < 0)}")
print(f"  Min: {min(lrc_intra):.4f}, Max: {max(lrc_intra):.4f}, Mean: {np.mean(lrc_intra):.4f}")

print("Bridges LRC:")
print(f"  Count: {len(lrc_bridges)}")
print(f"  Negative count: {sum(1 for c in lrc_bridges if c < 0)}")
print(f"  Min: {min(lrc_bridges):.4f}, Max: {max(lrc_bridges):.4f}, Mean: {np.mean(lrc_bridges):.4f}")
