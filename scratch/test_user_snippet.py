import networkx as nx
from torch_geometric.utils import from_networkx

G_test = nx.path_graph(5)  # nodes 0-1-2-3-4, unambiguous structure
data = from_networkx(G_test)
nodes = list(G_test.nodes())
# edge_index should reflect the path structure in the SAME node order as `nodes`
print(list(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist())))
print(nodes)
