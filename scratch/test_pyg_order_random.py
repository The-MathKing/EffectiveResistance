import networkx as nx
from torch_geometric.utils import from_networkx
import random

G = nx.erdos_renyi_graph(100, 0.1)
mapping = {n: f"node_{n}_{random.randint(0,1000)}" for n in G.nodes()}
G = nx.relabel_nodes(G, mapping)

data = from_networkx(G)
nodes = list(G.nodes())
node_idx = {u: i for i, u in enumerate(nodes)}

# check if every edge in data.edge_index maps correctly
edges_pyg = set()
for i in range(data.edge_index.shape[1]):
    u_idx = data.edge_index[0, i].item()
    v_idx = data.edge_index[1, i].item()
    edges_pyg.add((nodes[u_idx], nodes[v_idx]))

edges_nx = set(G.edges())
# PyG adds both directions for undirected graphs
edges_nx_directed = set()
for u, v in edges_nx:
    edges_nx_directed.add((u, v))
    edges_nx_directed.add((v, u))

if edges_pyg == edges_nx_directed:
    print("MATCH!")
else:
    print("MISMATCH!")
