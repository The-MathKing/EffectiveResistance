import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork, Amazon
from torch_geometric.utils import to_networkx, subgraph
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
import csv
import time

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
            
    return final_test_acc

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
        R_triu[min(i,j), max(i,j)] = np.inf
        
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

def get_homophily(data):
    from torch_geometric.utils import homophily
    return homophily(data.edge_index, data.y, method='edge')

def main():
    datasets = [
        ('Planetoid', 'Cora'), ('Planetoid', 'CiteSeer'),
        ('WebKB', 'Texas'), ('WebKB', 'Cornell'), ('WebKB', 'Wisconsin'),
        ('Wikipedia', 'Chameleon'), ('Wikipedia', 'Squirrel'),
        ('Amazon', 'Photo')
    ]
    
    results = []
    
    for cat, name in datasets:
        print(f"Processing {name}...")
        transform = T.NormalizeFeatures()
        if cat == 'Planetoid':
            dataset = Planetoid(root='/tmp/' + name, name=name, transform=transform)
        elif cat == 'WebKB':
            dataset = WebKB(root='/tmp/' + name, name=name, transform=transform)
        elif cat == 'Wikipedia':
            dataset = WikipediaNetwork(root='/tmp/' + name, name=name, transform=transform)
        elif cat == 'Amazon':
            dataset = Amazon(root='/tmp/' + name, name=name, transform=transform)
            
        data = dataset[0]
        
        G_raw = to_networkx(data, to_undirected=True)
        lcc = max(nx.connected_components(G_raw), key=len)
        
        node_idx = torch.tensor(sorted(list(lcc)))
        edge_index_raw, _ = subgraph(node_idx, data.edge_index, relabel_nodes=True)
        data.x = data.x[node_idx]
        data.y = data.y[node_idx]
        data.edge_index = edge_index_raw
        h_ratio = get_homophily(data)
        
        G_lcc = nx.Graph()
        G_lcc.add_nodes_from(range(data.x.size(0)))
        edges = [(edge_index_raw[0][i].item(), edge_index_raw[1][i].item()) for i in range(edge_index_raw.shape[1])]
        G_lcc.add_edges_from(edges)
        G_lcc.remove_edges_from(nx.selfloop_edges(G_lcc))
            
        cser_new_edges = compute_cser_rewired_edges(G_lcc, 0.05, use_cser=True)
        edge_index_cser = torch.cat([edge_index_raw, cser_new_edges], dim=1)
        
        deltas = []
        for seed in range(5):
            for split in range(5): # 5 splits * 5 seeds = 25 runs
                torch.manual_seed(seed * 10 + split)
                data_split = create_random_split(data.clone())
                
                torch.manual_seed(seed * 10 + split)
                model_raw = GCN(dataset.num_features, 64, dataset.num_classes)
                t_acc_raw = train_eval(model_raw, data_split, edge_index_raw)
                
                torch.manual_seed(seed * 10 + split)
                model_cser = GCN(dataset.num_features, 64, dataset.num_classes)
                t_acc_cser = train_eval(model_cser, data_split, edge_index_cser)
                
                deltas.append((t_acc_cser - t_acc_raw) * 100)
                
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        
        print(f"{name} Homophily: {h_ratio:.4f} | Delta: {mean_delta:.2f} +- {std_delta:.2f}")
        results.append({
            'Dataset': name,
            'Homophily': h_ratio,
            'Delta_Accuracy': mean_delta,
            'Delta_Std': std_delta
        })
        
        # Save incrementally
        with open('diagnostic_sweep_new.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Dataset', 'Homophily', 'Delta_Accuracy', 'Delta_Std'])
            writer.writeheader()
            writer.writerows(results)

if __name__ == '__main__':
    main()
