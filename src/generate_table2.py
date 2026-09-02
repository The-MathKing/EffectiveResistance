import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch_geometric.datasets import Planetoid, WebKB
from torch_geometric.utils import to_networkx, subgraph
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import torch_geometric.transforms as T
import csv
import time
import os

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

def create_random_split(data):
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
    patience = 50
    best_epoch = 0
    
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, edge_index)
        
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            out = model(data.x, edge_index)
            pred = out.argmax(dim=1)
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
                best_epoch = epoch
                
        if epoch - best_epoch > patience:
            break
            
    return best_val_acc, final_test_acc

def compute_cser_rewired_edges(G, budget, use_cser=True):
    G_core = nx.k_core(G, k=2) if use_cser else G
    if G_core.number_of_nodes() == 0:
        G_core = G
        
    n_edges = max(1, int(G.number_of_edges() * budget))
    
    L_core = nx.laplacian_matrix(G_core).toarray()
    L_pinv = np.linalg.pinv(L_core)
    core_nodes = list(G_core.nodes())
    node_to_idx = {n: i for i, n in enumerate(core_nodes)}
    
    diag = np.diag(L_pinv).reshape(-1, 1)
    R_matrix = diag + diag.T - 2 * L_pinv
    R_triu = np.triu(R_matrix, k=1)
    
    for u, v in G_core.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        R_triu[min(i,j), max(i,j)] = -np.inf
        
    R_triu[np.tril_indices(len(core_nodes))] = -np.inf
    flat_indices = np.argsort(R_triu, axis=None)[::-1]
    
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
            transform = T.NormalizeFeatures()
            if cat == 'Planetoid':
                dataset = Planetoid(root='/tmp/' + name, name=name, transform=transform)
            else:
                dataset = WebKB(root='/tmp/' + name, name=name, transform=transform)
                
            data = dataset[0]
            
            G_raw = to_networkx(data, to_undirected=True)
            lcc = max(nx.connected_components(G_raw), key=len)
            
            # Subgraph data to LCC
            node_idx = torch.tensor(sorted(list(lcc)))
            edge_index_raw, _ = subgraph(node_idx, data.edge_index, relabel_nodes=True)
            data.x = data.x[node_idx]
            data.y = data.y[node_idx]
            
            G_raw = nx.Graph()
            G_raw.add_nodes_from(range(data.x.size(0)))
            edges = [(edge_index_raw[0][i].item(), edge_index_raw[1][i].item()) for i in range(edge_index_raw.shape[1])]
            G_raw.add_edges_from(edges)
            G_raw.remove_edges_from(nx.selfloop_edges(G_raw))
                
            print(f"Computing ER and CSER rewired edges...")
            cser_new_edges = compute_cser_rewired_edges(G_raw, 0.05, use_cser=True)
            er_new_edges = compute_cser_rewired_edges(G_raw, 0.05, use_cser=False)
            
            print(f"  N_add CSER: {cser_new_edges.shape[1]//2}, N_add Naive ER: {er_new_edges.shape[1]//2}")
            # Ensure CSER and ER don't just output identical edges
            if cser_new_edges.shape[1] > 0 and er_new_edges.shape[1] > 0:
                cser_set = set([tuple(sorted(x)) for x in cser_new_edges.t().tolist()])
                er_set = set([tuple(sorted(x)) for x in er_new_edges.t().tolist()])
                overlap = len(cser_set.intersection(er_set))
                print(f"  Overlap between CSER and Naive ER non-edges: {overlap} / {len(cser_set)}")
            
            edge_index_cser = torch.cat([edge_index_raw, cser_new_edges], dim=1)
            edge_index_er = torch.cat([edge_index_raw, er_new_edges], dim=1)
            
            models = {
                'GCN': GCN,
                'GAT': GAT,
                'GraphSAGE': SAGE
            }
            
            for m_name, m_cls in models.items():
                print(f"Training {m_name} on {name}...")
                raw_val_accs, raw_test_accs = [], []
                er_val_accs, er_test_accs = [], []
                cser_val_accs, cser_test_accs = [], []
                
                # 5 seeds * 10 random splits = 50 runs
                for seed in range(5):
                    for split in range(10):
                        torch.manual_seed(seed * 10 + split)
                        data_split = create_random_split(data.clone())
                        
                        torch.manual_seed(seed * 10 + split)
                        model_raw = m_cls(dataset.num_features, 64, dataset.num_classes)
                        v_acc, t_acc = train_eval(model_raw, data_split, edge_index_raw)
                        raw_val_accs.append(v_acc)
                        raw_test_accs.append(t_acc)
                        
                        torch.manual_seed(seed * 10 + split)
                        model_er = m_cls(dataset.num_features, 64, dataset.num_classes)
                        v_acc, t_acc = train_eval(model_er, data_split, edge_index_er)
                        er_val_accs.append(v_acc)
                        er_test_accs.append(t_acc)
                        
                        torch.manual_seed(seed * 10 + split)
                        model_cser = m_cls(dataset.num_features, 64, dataset.num_classes)
                        v_acc, t_acc = train_eval(model_cser, data_split, edge_index_cser)
                        cser_val_accs.append(v_acc)
                        cser_test_accs.append(t_acc)
                
                mean_raw_val = np.mean(raw_val_accs) * 100
                mean_raw_test = np.mean(raw_test_accs) * 100
                std_raw_test = np.std(raw_test_accs) * 100
                
                mean_er_val = np.mean(er_val_accs) * 100
                mean_er_test = np.mean(er_test_accs) * 100
                std_er_test = np.std(er_test_accs) * 100
                
                mean_cser_val = np.mean(cser_val_accs) * 100
                mean_cser_test = np.mean(cser_test_accs) * 100
                std_cser_test = np.std(cser_test_accs) * 100
                
                print(f"{name} {m_name} - Raw: {mean_raw_test:.2f}±{std_raw_test:.2f}% | ER: {mean_er_test:.2f}±{std_er_test:.2f}% | CSER: {mean_cser_test:.2f}±{std_cser_test:.2f}%")
                
                results.append({
                    'Dataset': name,
                    'Model': m_name,
                    'Raw_Val_Acc': mean_raw_val,
                    'Raw_Test_Acc': mean_raw_test,
                    'Raw_Test_Std': std_raw_test,
                    'ER_Val_Acc': mean_er_val,
                    'ER_Test_Acc': mean_er_test,
                    'ER_Test_Std': std_er_test,
                    'CSER_Val_Acc': mean_cser_val,
                    'CSER_Test_Acc': mean_cser_test,
                    'CSER_Test_Std': std_cser_test
                })
                
    with open('gnn_benchmarks.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Dataset', 'Model', 'Raw_Val_Acc', 'Raw_Test_Acc', 'Raw_Test_Std', 'ER_Val_Acc', 'ER_Test_Acc', 'ER_Test_Std', 'CSER_Val_Acc', 'CSER_Test_Acc', 'CSER_Test_Std'])
        writer.writeheader()
        writer.writerows(results)
    print("Saved to gnn_benchmarks.csv")

if __name__ == '__main__':
    main()
