import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx

# Build a small non-sequential PyG Data object
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                           [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
x = torch.randn(5, 16)
data = Data(x=x, edge_index=edge_index)

# 1. to_networkx
G = to_networkx(data, to_undirected=True)

# 2. Extract a subgraph that drops nodes to force non-contiguous labels (drop node 2)
G_sub = G.subgraph([0, 1, 3, 4]).copy()

# 3. from_networkx
data_sub = from_networkx(G_sub)
nodes = list(G_sub.nodes())

# 4. Check mapping
print("PyG edge_index pairs:")
print(list(zip(data_sub.edge_index[0].tolist(), data_sub.edge_index[1].tolist())))
print("G_sub nodes list:")
print(nodes)
