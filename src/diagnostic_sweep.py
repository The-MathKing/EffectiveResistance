import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid, WebKB, Amazon
from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import os

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

def train_model(data, num_features, num_classes, epochs=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_features, 64, num_classes).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    best_val_acc = 0
    best_test_acc = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=1)
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
    return best_test_acc

def cser_rewiring(G, num_edges_to_add, k=2):
    G_new = G.copy()
    G_eval = nx.k_core(G_new, k=k)
    if G_eval.number_of_nodes() == 0:
        G_eval = G_new
        
    nodes = list(G_eval.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    
    L = nx.laplacian_matrix(G_eval).toarray()
    L_pinv = np.linalg.pinv(L)
    
    def get_er(u, v):
        i, j = node_to_idx[u], node_to_idx[v]
        return L_pinv[i, i] + L_pinv[j, j] - 2 * L_pinv[i, j]
    
    non_edges = []
    for _ in range(num_edges_to_add * 10):
        u, v = random.sample(nodes, 2)
        if not G_eval.has_edge(u, v):
            non_edges.append((get_er(u, v), u, v))
            
    non_edges.sort(reverse=True, key=lambda x: x[0])
    added = 0
    for r, u, v in non_edges:
        if added >= num_edges_to_add: break
        if not G_new.has_edge(u, v):
            G_new.add_edge(u, v)
            added += 1
    return G_new

def get_homophily(data):
    src, dst = data.edge_index
    matches = (data.y[src] == data.y[dst]).sum().item()
    total = data.edge_index.size(1)
    return matches / total if total > 0 else 0

def create_masks(data):
    if not hasattr(data, 'train_mask') or data.train_mask is None or data.train_mask.dim() == 2:
        num_nodes = data.num_nodes
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        indices = torch.randperm(num_nodes)
        train_end = int(0.6 * num_nodes)
        val_end = int(0.8 * num_nodes)
        
        train_mask[indices[:train_end]] = True
        val_mask[indices[train_end:val_end]] = True
        test_mask[indices[val_end:]] = True
        
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
    return data

datasets = {
    'Planetoid': ['Cora', 'CiteSeer', 'PubMed'],
    'WebKB': ['Texas', 'Cornell', 'Wisconsin'],
    'Amazon': ['Computers', 'Photo']
}

def run_diagnostic():
    results = []
    
    for category, names in datasets.items():
        for name in names:
            print(f"Processing {name}...")
            if category == 'Planetoid':
                dataset = Planetoid(root='/tmp/' + name, name=name)
            elif category == 'WebKB':
                dataset = WebKB(root='/tmp/' + name, name=name)
            elif category == 'Amazon':
                dataset = Amazon(root='/tmp/' + name, name=name)
                
            data = dataset[0]
            data = create_masks(data)
            
            homophily = get_homophily(data)
            
            G_raw = to_networkx(data, to_undirected=True)
            largest_cc = max(nx.connected_components(G_raw), key=len)
            G_raw = G_raw.subgraph(largest_cc).copy()
            G_raw.remove_edges_from(nx.selfloop_edges(G_raw))
            
            node_mapping = {n: i for i, n in enumerate(G_raw.nodes())}
            node_list = list(G_raw.nodes())
            
            data_lcc = data.clone()
            data_lcc.x = data.x[node_list]
            data_lcc.y = data.y[node_list]
            data_lcc.train_mask = data.train_mask[node_list]
            data_lcc.val_mask = data.val_mask[node_list]
            data_lcc.test_mask = data.test_mask[node_list]
            ei = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G_raw.edges()]).t().contiguous()
            if ei.numel() == 0: ei = torch.empty((2, 0), dtype=torch.long)
            data_lcc.edge_index = ei
            
            acc_raw = np.mean([train_model(data_lcc.clone(), dataset.num_features, dataset.num_classes) for _ in range(3)])
            
            # CSER 5% budget
            n_edges = int(G_raw.number_of_edges() * 0.05)
            # Skip if graph is too huge for dense matrix inversion on CPU (e.g. PubMed/Computers)
            if G_raw.number_of_nodes() > 5000:
                print(f"Skipping {name} due to dense O(N^3) memory constraints.")
                continue
                
            G_cser = cser_rewiring(G_raw, n_edges, k=2)
            ei_cser = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G_cser.edges()]).t().contiguous()
            data_cser = data_lcc.clone()
            data_cser.edge_index = ei_cser
            
            acc_cser = np.mean([train_model(data_cser, dataset.num_features, dataset.num_classes) for _ in range(3)])
            
            delta = (acc_cser - acc_raw) * 100
            print(f"{name}: Homophily={homophily:.3f}, Delta={delta:.3f}%")
            results.append((name, homophily, delta))
            
    # Plot Scatter
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 14
    
    fig, ax = plt.subplots(figsize=(6, 4))
    homophilies = [r[1] for r in results]
    deltas = [r[2] for r in results]
    names = [r[0] for r in results]
    
    ax.scatter(homophilies, deltas, color='tab:blue', s=100, edgecolor='black', zorder=3)
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    ax.axvline(0.75, color='tab:red', linewidth=2, linestyle='--', label='Diagnostic Boundary (h=0.75)')
    ax.fill_betweenx([min(deltas)-1, max(deltas)+1], 0, 0.75, color='tab:green', alpha=0.1, label='Rewiring Recommended')
    ax.fill_betweenx([min(deltas)-1, max(deltas)+1], 0.75, 1.0, color='tab:red', alpha=0.1, label='Rewiring Contraindicated')
    
    for i, name in enumerate(names):
        ax.annotate(name, (homophilies[i], deltas[i]), xytext=(5, 5), textcoords='offset points', fontsize=10)
        
    ax.set_xlabel('Initial Edge Homophily ($h$)', fontsize=14)
    ax.set_ylabel('$\Delta$ Test Accuracy (%)', fontsize=14)
    ax.set_title('Diagnostic Validation: Homophily vs CSER Boost', fontsize=16)
    ax.legend(fontsize=10, loc='lower left')
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('diagnostic_scatter.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    
    print("diagnostic_scatter.pdf generated.")

if __name__ == "__main__":
    run_diagnostic()
