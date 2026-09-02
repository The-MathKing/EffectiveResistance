import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import csv
from sklearn.model_selection import KFold
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
        
        # Rewire all graphs first
        print(f"  Rewiring {ds_name} with CSER...")
        
        raw_data_list = []
        cser_data_list = []
        
        for i in range(len(dataset)):
            # Extract data object
            d_raw = dataset[i]
            d_cser = d_raw.clone()
            
            if 'x' not in d_cser or d_cser.x is None:
                if hasattr(dataset, 'num_node_features') and dataset.num_node_features > 0:
                    pass
                else:
                    d_cser.x = torch.ones((d_cser.num_nodes, 1))
                    d_raw.x = torch.ones((d_raw.num_nodes, 1))
                    
            cser_edge_index = rewire_graph(d_cser)
            d_cser.edge_index = cser_edge_index
            
            raw_data_list.append(d_raw)
            cser_data_list.append(d_cser)
            
        n_add_total = sum(d.edge_index.size(1) - r.edge_index.size(1) for d, r in zip(cser_data_list, raw_data_list)) // 2
        print(f"  Total N_add across all {len(dataset)} graphs: {n_add_total}")
        
        if n_add_total == 0:
            print(f"  No edges added for {ds_name}. Skipping to next dataset.")
            continue
            
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        
        raw_accs = []
        cser_accs = []
        
        fold_idx = 0
        for train_idx, test_idx in kf.split(dataset):
            print(f"  Fold {fold_idx + 1}/10")
            
            # RAW
            train_dataset_raw = [raw_data_list[i] for i in train_idx]
            test_dataset_raw = [raw_data_list[i] for i in test_idx]
            
            raw_train_loader = DataLoader(train_dataset_raw, batch_size=64, shuffle=True)
            raw_test_loader = DataLoader(test_dataset_raw, batch_size=64, shuffle=False)
            
            torch.manual_seed(fold_idx) # Use fold_idx as seed for simplicity
            model_raw = GCN_GraphLevel(dataset.num_node_features if dataset.num_node_features > 0 else 1, 64, dataset.num_classes)
            optimizer_raw = torch.optim.Adam(model_raw.parameters(), lr=0.01)
            
            for epoch in range(100):
                train(model_raw, raw_train_loader, optimizer_raw)
            acc_raw = test(model_raw, raw_test_loader)
            raw_accs.append(acc_raw)
            
            # CSER
            train_dataset_cser = [cser_data_list[i] for i in train_idx]
            test_dataset_cser = [cser_data_list[i] for i in test_idx]
            
            cser_train_loader = DataLoader(train_dataset_cser, batch_size=64, shuffle=True)
            cser_test_loader = DataLoader(test_dataset_cser, batch_size=64, shuffle=False)
            
            torch.manual_seed(fold_idx)
            model_cser = GCN_GraphLevel(dataset.num_node_features if dataset.num_node_features > 0 else 1, 64, dataset.num_classes)
            optimizer_cser = torch.optim.Adam(model_cser.parameters(), lr=0.01)
            
            for epoch in range(100):
                train(model_cser, cser_train_loader, optimizer_cser)
            acc_cser = test(model_cser, cser_test_loader)
            cser_accs.append(acc_cser)
            
            fold_idx += 1
            
        mean_raw_acc = np.mean(raw_accs)
        mean_cser_acc = np.mean(cser_accs)
        print(f"  RAW 10-fold CV Acc: {mean_raw_acc*100:.2f}%")
        print(f"  CSER 10-fold CV Acc: {mean_cser_acc*100:.2f}%")
        
        results.append({
            'dataset': ds_name,
            'raw_acc': mean_raw_acc * 100,
            'raw_std': np.std(raw_accs) * 100,
            'cser_acc': mean_cser_acc * 100,
            'cser_std': np.std(cser_accs) * 100
        })
        
    with open('graph_classification.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'raw_acc', 'raw_std', 'cser_acc', 'cser_std'])
        writer.writeheader()
        writer.writerows(results)
        
    print("Saved to graph_classification.csv")

if __name__ == '__main__':
    main()
