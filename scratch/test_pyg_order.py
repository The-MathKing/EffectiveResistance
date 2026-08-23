import networkx as nx
from torch_geometric.utils import from_networkx

G_test = nx.path_graph(5)
# make it unsorted just to test
G_test = nx.Graph()
G_test.add_node(3)
G_test.add_node(1)
G_test.add_node(4)
G_test.add_node(2)
G_test.add_node(0)
G_test.add_edge(3, 1)
G_test.add_edge(1, 4)
G_test.add_edge(4, 2)
G_test.add_edge(2, 0)

data = from_networkx(G_test)
nodes = list(G_test.nodes())

print("Edge index pairs:")
print(list(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist())))
print("Nodes list:")
print(nodes)
