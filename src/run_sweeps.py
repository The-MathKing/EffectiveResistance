import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np
import scipy.sparse.linalg as sla
import argparse
import random
import json
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

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * 8, out_channels, heads=1, concat=False, dropout=0.6)
    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        return self.conv2(x, edge_index)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

def train_model(model_cls, data, num_features, num_classes, epochs=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_cls(num_features, 16 if model_cls == GAT else 64, num_classes).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01 if model_cls != GAT else 0.005, weight_decay=5e-4)
    
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

def compute_lambda_2(G):
    L = nx.normalized_laplacian_matrix(G).astype(float)
    try:
        eigenvalues, _ = sla.eigsh(L, k=2, which='SM')
        return eigenvalues[1]
    except:
        return 0.0

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

if __name__ == "__main__":
    dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')
    data = dataset[0]
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
    
    # Need edge_index mapping
    ei = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G_raw.edges()]).t().contiguous()
    data_lcc.edge_index = ei
    
    results = {"budgets": {}, "k_core": {}, "architectures": {}}
    
    print("--- 1. Budget Sweep (CSER k=2) ---")
    budgets = [0.01, 0.05, 0.10, 0.15, 0.20]
    for b in budgets:
        n_edges = int(G_raw.number_of_edges() * b)
        G_cser = cser_rewiring(G_raw, n_edges, k=2)
        l2 = compute_lambda_2(G_cser)
        ei = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G_cser.edges()]).t().contiguous()
        data_cser = data_lcc.clone()
        data_cser.edge_index = ei
        acc = np.mean([train_model(GCN, data_cser, dataset.num_features, dataset.num_classes) for _ in range(3)])
        print(f"Budget {b*100}% -> L2: {l2:.4f}, Acc: {acc:.4f}")
        results["budgets"][str(b)] = {"l2": l2, "acc": acc}
        
    print("\n--- 2. k-Core Ablation (Budget 5%) ---")
    n_edges = int(G_raw.number_of_edges() * 0.05)
    for k in [2, 3, 4]:
        G_cser = cser_rewiring(G_raw, n_edges, k=k)
        l2 = compute_lambda_2(G_cser)
        ei = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G_cser.edges()]).t().contiguous()
        data_cser = data_lcc.clone()
        data_cser.edge_index = ei
        acc = np.mean([train_model(GCN, data_cser, dataset.num_features, dataset.num_classes) for _ in range(3)])
        print(f"k={k} -> L2: {l2:.4f}, Acc: {acc:.4f}")
        results["k_core"][str(k)] = {"l2": l2, "acc": acc}
        
    print("\n--- 3. Architectural Ablation (Budget 5%, k=2) ---")
    G_cser = cser_rewiring(G_raw, n_edges, k=2)
    ei = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G_cser.edges()]).t().contiguous()
    data_cser = data_lcc.clone()
    data_cser.edge_index = ei
    
    for name, cls in [("GCN", GCN), ("GAT", GAT), ("GraphSAGE", GraphSAGE)]:
        acc_raw = np.mean([train_model(cls, data_lcc.clone(), dataset.num_features, dataset.num_classes) for _ in range(3)])
        acc_cser = np.mean([train_model(cls, data_cser, dataset.num_features, dataset.num_classes) for _ in range(3)])
        print(f"{name} -> Raw Acc: {acc_raw:.4f}, CSER Acc: {acc_cser:.4f}")
        results["architectures"][name] = {"raw_acc": acc_raw, "cser_acc": acc_cser}
        
    os.makedirs("results", exist_ok=True)
    with open("results/sweeps.json", "w") as f:
        json.dump(results, f, indent=2)
