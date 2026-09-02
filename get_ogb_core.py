import networkx as nx
from ogb.nodeproppred import NodePropPredDataset

dataset = NodePropPredDataset(name='ogbn-arxiv', root='/tmp/ogbn-arxiv')
data = dataset[0]
edges = data[0]['edge_index']
G_raw = nx.Graph()
G_raw.add_edges_from(zip(edges[0], edges[1]))
largest_cc = max(nx.connected_components(G_raw), key=len)
G_raw = G_raw.subgraph(largest_cc).copy()
G_raw.remove_edges_from(nx.selfloop_edges(G_raw))
core_2 = nx.k_core(G_raw, k=2)
pruned = G_raw.number_of_nodes() - core_2.number_of_nodes()
print(f"ogbn-arxiv: {pruned} / {G_raw.number_of_nodes()} ({pruned/G_raw.number_of_nodes()*100:.1f}%)")
