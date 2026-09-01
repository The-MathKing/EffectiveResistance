import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import csv
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_networkx

from train_rewired_gnn import compute_cser_rewired_edges

class GCN_GraphLevel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

def rewire_graph(data, budget=0.05):
    # Rewires a single graph
    G = to_networkx(data, to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    if G.number_of_nodes() < 5 or G.number_of_edges() < 5:
        return data.edge_index # Too small
        
    try:
        new_edges = compute_cser_rewired_edges(G, budget)
        edge_index_rewired = torch.cat([data.edge_index, new_edges], dim=1)
        return edge_index_rewired
    except Exception as e:
        # If disconnected or issues with ER calculation
        return data.edge_index

def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

def main():
    datasets = ['MUTAG', 'PROTEINS', 'ENZYMES', 'NCI1']
    results = []
    
    for ds_name in datasets:
        print(f"Loading {ds_name}...")
        dataset = TUDataset(root=f'/tmp/TUDataset/{ds_name}', name=ds_name)
        
        # Split 80/20
        torch.manual_seed(42)
        dataset = dataset.shuffle()
        train_dataset = dataset[:int(0.8 * len(dataset))]
        test_dataset = dataset[int(0.8 * len(dataset)):]
        
        # We need to evaluate both RAW and CSER
        raw_train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        raw_test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        print(f"  Training RAW GCN on {ds_name}...")
        raw_accs = []
        for seed in range(5):
            torch.manual_seed(seed)
            model = GCN_GraphLevel(dataset.num_node_features, 64, dataset.num_classes)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            for epoch in range(100):
                train(model, raw_train_loader, optimizer)
            acc = test(model, raw_test_loader)
            raw_accs.append(acc)
        
        mean_raw_acc = np.mean(raw_accs)
        print(f"  RAW Acc: {mean_raw_acc*100:.2f}%")
        
        # Rewire training and test sets
        print(f"  Rewiring {ds_name} with CSER...")
        import copy
        rewired_train = copy.deepcopy(train_dataset)
        rewired_test = copy.deepcopy(test_dataset)
        
        for i in range(len(rewired_train)):
            if 'x' not in rewired_train[i] or rewired_train[i].x is None:
                # Some datasets like MUTAG have node labels, not features. Add dummy features if needed.
                if hasattr(dataset, 'num_node_features') and dataset.num_node_features > 0:
                    pass
                else:
                    rewired_train[i].x = torch.ones((rewired_train[i].num_nodes, 1))
            rewired_train[i].edge_index = rewire_graph(rewired_train[i])
            
        for i in range(len(rewired_test)):
            if 'x' not in rewired_test[i] or rewired_test[i].x is None:
                if hasattr(dataset, 'num_node_features') and dataset.num_node_features > 0:
                    pass
                else:
                    rewired_test[i].x = torch.ones((rewired_test[i].num_nodes, 1))
            rewired_test[i].edge_index = rewire_graph(rewired_test[i])
            
        cser_train_loader = DataLoader(rewired_train, batch_size=64, shuffle=True)
        cser_test_loader = DataLoader(rewired_test, batch_size=64, shuffle=False)
        
        print(f"  Training CSER GCN on {ds_name}...")
        cser_accs = []
        for seed in range(5):
            torch.manual_seed(seed)
            model = GCN_GraphLevel(dataset.num_node_features if dataset.num_node_features > 0 else 1, 64, dataset.num_classes)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            for epoch in range(100):
                train(model, cser_train_loader, optimizer)
            acc = test(model, cser_test_loader)
            cser_accs.append(acc)
            
        mean_cser_acc = np.mean(cser_accs)
        print(f"  CSER Acc: {mean_cser_acc*100:.2f}%")
        
        results.append({
            'dataset': ds_name,
            'raw_acc': mean_raw_acc * 100,
            'cser_acc': mean_cser_acc * 100
        })
        
    with open('graph_classification.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'raw_acc', 'cser_acc'])
        writer.writeheader()
        writer.writerows(results)
        
    print("Saved to graph_classification.csv")

if __name__ == '__main__':
    main()
