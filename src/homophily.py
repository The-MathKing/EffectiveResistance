from torch_geometric.datasets import Planetoid

def get_homophily(name):
    dataset = Planetoid(root='/tmp/' + name, name=name)
    data = dataset[0]
    edge_index = data.edge_index
    y = data.y
    src, dst = edge_index
    matches = (y[src] == y[dst]).sum().item()
    total = edge_index.size(1)
    return matches / total

print("Cora Homophily:", get_homophily("Cora"))
print("CiteSeer Homophily:", get_homophily("CiteSeer"))
