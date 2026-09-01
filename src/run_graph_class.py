import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_networkx, degree

from train_rewired_gnn import compute_cser_rewired_edges
import copy

class GCN_GraphLevel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

def rewire_graph(data, budget=0.05):
    G = to_networkx(data, to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    if G.number_of_nodes() < 5 or G.number_of_edges() < 5:
        return data.edge_index
        
    try:
        new_edges = compute_cser_rewired_edges(G, budget)
        if new_edges.numel() > 0:
            return torch.cat([data.edge_index, new_edges], dim=1)
        return data.edge_index
    except Exception:
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

def add_features(dataset):
    dataset_list = []
    for data in dataset:
        if data.x is None or data.x.size(1) == 0:
            # Use node degree as feature
            row, col = data.edge_index
            deg = degree(row, data.num_nodes, dtype=torch.float).view(-1, 1)
            data.x = deg
        dataset_list.append(data)
    return dataset_list

def main():
    datasets = ['MUTAG', 'PROTEINS', 'ENZYMES', 'NCI1']
    
    for ds_name in datasets:
        print(f"Loading {ds_name}...")
        dataset = TUDataset(root=f'/tmp/TUDataset/{ds_name}', name=ds_name)
        
        # Add features if missing
        dataset_list = add_features(dataset)
        num_features = dataset_list[0].x.size(1)
        num_classes = dataset.num_classes
        
        raw_accs = []
        cser_accs = []
        
        for seed in range(5):
            print(f"--- Seed {seed} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Shuffle and split per seed
            indices = torch.randperm(len(dataset_list)).tolist()
            train_idx = indices[:int(0.8 * len(dataset_list))]
            test_idx = indices[int(0.8 * len(dataset_list)):]
            
            train_data_raw = [dataset_list[i] for i in train_idx]
            test_data_raw = [dataset_list[i] for i in test_idx]
            
            # Create rewired copies
            train_data_cser = copy.deepcopy(train_data_raw)
            test_data_cser = copy.deepcopy(test_data_raw)
            
            for d in train_data_cser:
                d.edge_index = rewire_graph(d)
            for d in test_data_cser:
                d.edge_index = rewire_graph(d)
                
            # Loaders
            raw_train_loader = DataLoader(train_data_raw, batch_size=32, shuffle=True)
            raw_test_loader = DataLoader(test_data_raw, batch_size=32, shuffle=False)
            
            cser_train_loader = DataLoader(train_data_cser, batch_size=32, shuffle=True)
            cser_test_loader = DataLoader(test_data_cser, batch_size=32, shuffle=False)
            
            # Train RAW
            model_raw = GCN_GraphLevel(num_features, 64, num_classes)
            opt_raw = torch.optim.Adam(model_raw.parameters(), lr=0.005)
            for epoch in range(100):
                train(model_raw, raw_train_loader, opt_raw)
            raw_acc = test(model_raw, raw_test_loader)
            raw_accs.append(raw_acc)
            
            # Train CSER
            torch.manual_seed(seed) # reset seed to ensure identical init
            model_cser = GCN_GraphLevel(num_features, 64, num_classes)
            opt_cser = torch.optim.Adam(model_cser.parameters(), lr=0.005)
            for epoch in range(100):
                train(model_cser, cser_train_loader, opt_cser)
            cser_acc = test(model_cser, cser_test_loader)
            cser_accs.append(cser_acc)
            
            print(f"Seed {seed} | RAW Acc: {raw_acc*100:.2f}% | CSER Acc: {cser_acc*100:.2f}%")
            
        print(f"\nFinal {ds_name} Results:")
        print(f"RAW:  {np.mean(raw_accs)*100:.2f}% +/- {np.std(raw_accs)*100:.2f}%")
        print(f"CSER: {np.mean(cser_accs)*100:.2f}% +/- {np.std(cser_accs)*100:.2f}%")

if __name__ == '__main__':
    main()
