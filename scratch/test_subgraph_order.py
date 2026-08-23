import networkx as nx
from torch_geometric.utils import from_networkx

G = nx.Graph()
G.add_nodes_from([5, 17, 42, 100])
G.add_edges_from([(5, 17), (17, 42), (42, 100)])
# drop node 17 from a "largest_cc"-style subgraph to force non-contiguous labels
G_sub = G.subgraph([5, 42, 100]).copy()  # non-contiguous, and not starting at 0

data = from_networkx(G_sub)
nodes = list(G_sub.nodes())
print(list(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist())))
print(nodes)
