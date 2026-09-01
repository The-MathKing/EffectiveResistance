import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch_geometric.datasets import Planetoid, WebKB
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import csv
import time
import os
import copy
from metrics import effective_resistance

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels // 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=0.6)
    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def create_masks(data):
    if not hasattr(data, 'train_mask') or data.train_mask.dim() == 2:
        num_nodes = data.num_nodes
        indices = torch.randperm(num_nodes)
        train_idx = indices[:int(0.6 * num_nodes)]
        val_idx = indices[int(0.6 * num_nodes):int(0.8 * num_nodes)]
        test_idx = indices[int(0.8 * num_nodes):]
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
    return data

def train_eval(model, data, edge_index):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val_acc = 0
    final_test_acc = 0
    
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, edge_index)
        
        if data.train_mask.dim() == 2:
            train_mask = data.train_mask[:, 0]
            val_mask = data.val_mask[:, 0]
            test_mask = data.test_mask[:, 0]
        else:
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask
            
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        
        model.eval()
        pred = out.argmax(dim=1)
        val_acc = (pred[val_mask] == data.y[val_mask]).sum().item() / val_mask.sum().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            final_test_acc = (pred[test_mask] == data.y[test_mask]).sum().item() / test_mask.sum().item()
            
    return final_test_acc

def compute_cser_rewired_edges(G, budget):
    G_core = nx.k_core(G, k=2)
    # Calculate ER for all possible non-edges in the core?
    # Or just edges? "Select the top 5% of node pairs with the lowest R values and inject them"
    # Actually, calculating ER for all non-edges O(|V|^2) is slow. We can just use the spectral gap / FOSR approach, 
    # but the instruction says "compute Core-Shielded Effective Resistance. Select top 5% of node pairs with lowest R values".
    # Wait, 5% of node pairs is huge. 5% of *existing edges* budget.
    n_edges = max(1, int(G.number_of_edges() * budget))
    
    L_core = nx.laplacian_matrix(G_core).toarray()
    L_pinv = np.linalg.pinv(L_core)
    core_nodes = list(G_core.nodes())
    node_to_idx = {n: i for i, n in enumerate(core_nodes)}
    
    # We only want to evaluate non-edges.
    # To save time, we can randomly sample non-edges and pick the best, or since Cora is 2485 nodes, 2485^2 = 6 million,
    # computing ER from L_pinv is fast.
    R_vals = []
    # Vectorized computation of effective resistance for all pairs in core
    # R = diag(L_pinv) 1^T + 1 diag(L_pinv)^T - 2 L_pinv
    diag = np.diag(L_pinv).reshape(-1, 1)
    R_matrix = diag + diag.T - 2 * L_pinv
    
    # Extract non-edges
    import itertools
    # To avoid O(V^2) memory bottleneck, we can just grab upper triangle
    R_triu = np.triu(R_matrix, k=1)
    
    # zero out existing edges so we don't pick them
    for u, v in G_core.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        R_triu[min(i,j), max(i,j)] = np.inf
        
    # Get top n_edges with LOWEST R
    # Flatten and argsort
    flat_indices = np.argsort(R_triu, axis=None)
    
    # Filter out zeros from lower triangle/diag (which are 0) by selecting where R_triu > 0 and R_triu < inf
    # Wait, argsort will put 0 first. We should set lower triangle and diag to inf
    R_triu[np.tril_indices(len(core_nodes))] = np.inf
    flat_indices = np.argsort(R_triu, axis=None)
    
    best_indices = flat_indices[:n_edges]
    
    new_edges = []
    for idx in best_indices:
        i, j = np.unravel_index(idx, R_triu.shape)
        u, v = core_nodes[i], core_nodes[j]
        new_edges.append([u, v])
        new_edges.append([v, u])
        
    return torch.tensor(new_edges, dtype=torch.long).t()

def main():
    datasets = {
        'Planetoid': ['Cora', 'CiteSeer'],
        'WebKB': ['Texas', 'Cornell']
    }
    
    results = []
    
    for cat, names in datasets.items():
        for name in names:
            print(f"Processing {name}...")
            if cat == 'Planetoid':
                dataset = Planetoid(root='/tmp/' + name, name=name)
            else:
                dataset = WebKB(root='/tmp/' + name, name=name)
                
            data = dataset[0]
            data = create_masks(data)
            
            G_raw = to_networkx(data, to_undirected=True)
            lcc = max(nx.connected_components(G_raw), key=len)
            
            # Subgraph data to LCC
            node_idx = torch.tensor(sorted(list(lcc)))
            from torch_geometric.utils import subgraph
            edge_index_raw, _ = subgraph(node_idx, data.edge_index, relabel_nodes=True)
            data.x = data.x[node_idx]
            data.y = data.y[node_idx]
            if data.train_mask.dim() == 2:
                data.train_mask = data.train_mask[node_idx, :]
                data.val_mask = data.val_mask[node_idx, :]
                data.test_mask = data.test_mask[node_idx, :]
            else:
                data.train_mask = data.train_mask[node_idx]
                data.val_mask = data.val_mask[node_idx]
                data.test_mask = data.test_mask[node_idx]
                
            G_raw = nx.Graph()
            G_raw.add_nodes_from(range(data.x.size(0)))
            edges = [(edge_index_raw[0][i].item(), edge_index_raw[1][i].item()) for i in range(edge_index_raw.shape[1])]
            G_raw.add_edges_from(edges)
            G_raw.remove_edges_from(nx.selfloop_edges(G_raw))
                
            print("Computing CSER rewired edges...")
            new_edges = compute_cser_rewired_edges(G_raw, 0.05)
            edge_index_rewired = torch.cat([edge_index_raw, new_edges], dim=1)
            
            models = {
                'GCN': GCN,
                'GAT': GAT,
                'GraphSAGE': SAGE
            }
            
            for m_name, m_cls in models.items():
                print(f"Training {m_name} on {name}...")
                raw_accs = []
                cser_accs = []
                for seed in range(10):
                    torch.manual_seed(seed)
                    model_raw = m_cls(dataset.num_features, 64, dataset.num_classes)
                    raw_accs.append(train_eval(model_raw, data, edge_index_raw))
                    
                    torch.manual_seed(seed)
                    model_cser = m_cls(dataset.num_features, 64, dataset.num_classes)
                    cser_accs.append(train_eval(model_cser, data, edge_index_rewired))
                
                mean_raw = np.mean(raw_accs) * 100
                mean_cser = np.mean(cser_accs) * 100
                print(f"{name} {m_name} - Raw: {mean_raw:.2f}% | CSER: {mean_cser:.2f}%")
                
                results.append({
                    'Dataset': name,
                    'Model': m_name,
                    'Raw_Acc': mean_raw,
                    'CSER_Acc': mean_cser
                })
                
    with open('gnn_benchmarks.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Dataset', 'Model', 'Raw_Acc', 'CSER_Acc'])
        writer.writeheader()
        writer.writerows(results)
    print("Saved to gnn_benchmarks.csv")

if __name__ == '__main__':
    main()
