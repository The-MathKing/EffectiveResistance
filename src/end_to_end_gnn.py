import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import numpy as np
import scipy.sparse.linalg as sla
import argparse
from tqdm import tqdm
import random

from metrics import effective_resistance, waf3_exact

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def train_gcn(data, num_features, num_classes, epochs=200):
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

def compute_lambda_2(G):
    L = nx.normalized_laplacian_matrix(G).astype(float)
    try:
        eigenvalues, _ = sla.eigsh(L, k=2, which='SM')
        return eigenvalues[1]
    except:
        return 0.0

def fosr_rewiring(G, num_edges_to_add):
    G_new = G.copy()
    L = nx.laplacian_matrix(G_new).astype(float)
    try:
        vals, vecs = sla.eigsh(L, k=2, which='SM')
        fiedler = vecs[:, 1]
    except:
        return G_new
        
    nodes = list(G_new.nodes())
    non_edges = []
    
    # Sample some non-edges to find max diff
    for _ in range(num_edges_to_add * 20):
        u, v = random.sample(nodes, 2)
        if not G_new.has_edge(u, v):
            idx_u = nodes.index(u)
            idx_v = nodes.index(v)
            diff = (fiedler[idx_u] - fiedler[idx_v])**2
            non_edges.append((diff, u, v))
            
    non_edges.sort(reverse=True, key=lambda x: x[0])
    
    added = 0
    for diff, u, v in non_edges:
        if added >= num_edges_to_add:
            break
        if not G_new.has_edge(u, v):
            G_new.add_edge(u, v)
            added += 1
            
    return G_new

def er_rewiring(G, num_edges_to_add, use_cser=False):
    G_new = G.copy()
    
    if use_cser:
        G_eval = nx.k_core(G_new, k=2)
        if G_eval.number_of_nodes() == 0:
            G_eval = G_new
    else:
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
            r = get_er(u, v)
            non_edges.append((r, u, v))
            
    non_edges.sort(reverse=True, key=lambda x: x[0])
    
    added = 0
    for r, u, v in non_edges:
        if added >= num_edges_to_add:
            break
        if not G_new.has_edge(u, v):
            G_new.add_edge(u, v)
            added += 1
            
    return G_new

def waf3_rewiring(G, num_edges_to_add):
    G_new = G.copy()
    nodes = list(G_new.nodes())
    non_edges = []
    
    for _ in range(num_edges_to_add * 10):
        u, v = random.sample(nodes, 2)
        if not G_new.has_edge(u, v):
            w = waf3_exact(G_new, u, v)
            non_edges.append((w, u, v))
            
    # For WAF3, bottlenecks have low (negative) curvature, so we want to add edges with MINIMUM curvature
    non_edges.sort(key=lambda x: x[0])
    
    added = 0
    for w, u, v in non_edges:
        if added >= num_edges_to_add:
            break
        if not G_new.has_edge(u, v):
            G_new.add_edge(u, v)
            added += 1
            
    return G_new

def run_end_to_end(dataset_name):
    print(f"\n--- End-to-End Pipeline on {dataset_name} ---")
    dataset = Planetoid(root='/tmp/' + dataset_name, name=dataset_name)
    data = dataset[0]
    
    G_raw = to_networkx(data, to_undirected=True)
    largest_cc = max(nx.connected_components(G_raw), key=len)
    G_raw = G_raw.subgraph(largest_cc).copy()
    G_raw.remove_edges_from(nx.selfloop_edges(G_raw))
    
    # Map LCC back to PyG data
    node_mapping = {n: i for i, n in enumerate(G_raw.nodes())}
    edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G_raw.edges()]).t().contiguous()
    
    # We create a clean subgraph data object
    node_list = list(G_raw.nodes())
    data_lcc = data.clone()
    data_lcc.x = data.x[node_list]
    data_lcc.y = data.y[node_list]
    data_lcc.train_mask = data.train_mask[node_list]
    data_lcc.val_mask = data.val_mask[node_list]
    data_lcc.test_mask = data.test_mask[node_list]
    data_lcc.edge_index = edge_index
    
    num_edges_to_add = int(G_raw.number_of_edges() * 0.05) # Add 5% edges
    print(f"Graph nodes: {G_raw.number_of_nodes()}, edges: {G_raw.number_of_edges()}")
    print(f"Edges to add (5%): {num_edges_to_add}")
    
    results = {}
    
    # Baseline: Raw
    l2_raw = compute_lambda_2(G_raw)
    acc_raw = np.mean([train_gcn(data_lcc, dataset.num_features, dataset.num_classes) for _ in range(5)])
    results['Raw'] = {'lambda_2': l2_raw, 'test_acc': acc_raw}
    print(f"Raw -> Lambda_2: {l2_raw:.4f}, Test Acc: {acc_raw:.4f}")
    
    # Rewire FOSR
    G_fosr = fosr_rewiring(G_raw, num_edges_to_add)
    edge_index_fosr = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G_fosr.edges()]).t().contiguous()
    data_fosr = data_lcc.clone()
    data_fosr.edge_index = edge_index_fosr
    l2_fosr = compute_lambda_2(G_fosr)
    acc_fosr = np.mean([train_gcn(data_fosr, dataset.num_features, dataset.num_classes) for _ in range(5)])
    results['FOSR'] = {'lambda_2': l2_fosr, 'test_acc': acc_fosr}
    print(f"FOSR -> Lambda_2: {l2_fosr:.4f}, Test Acc: {acc_fosr:.4f}")
    
    # Rewire Naive ER
    G_er = er_rewiring(G_raw, num_edges_to_add, use_cser=False)
    edge_index_er = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G_er.edges()]).t().contiguous()
    data_er = data_lcc.clone()
    data_er.edge_index = edge_index_er
    l2_er = compute_lambda_2(G_er)
    acc_er = np.mean([train_gcn(data_er, dataset.num_features, dataset.num_classes) for _ in range(5)])
    results['Naive ER'] = {'lambda_2': l2_er, 'test_acc': acc_er}
    print(f"Naive ER -> Lambda_2: {l2_er:.4f}, Test Acc: {acc_er:.4f}")
    
    # Rewire WAF3
    G_waf3 = waf3_rewiring(G_raw, num_edges_to_add)
    edge_index_waf3 = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G_waf3.edges()]).t().contiguous()
    data_waf3 = data_lcc.clone()
    data_waf3.edge_index = edge_index_waf3
    l2_waf3 = compute_lambda_2(G_waf3)
    acc_waf3 = np.mean([train_gcn(data_waf3, dataset.num_features, dataset.num_classes) for _ in range(5)])
    results['WAF3'] = {'lambda_2': l2_waf3, 'test_acc': acc_waf3}
    print(f"WAF3 -> Lambda_2: {l2_waf3:.4f}, Test Acc: {acc_waf3:.4f}")
    
    # Rewire CSER
    G_cser = er_rewiring(G_raw, num_edges_to_add, use_cser=True)
    edge_index_cser = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G_cser.edges()]).t().contiguous()
    data_cser = data_lcc.clone()
    data_cser.edge_index = edge_index_cser
    l2_cser = compute_lambda_2(G_cser)
    acc_cser = np.mean([train_gcn(data_cser, dataset.num_features, dataset.num_classes) for _ in range(5)])
    results['CSER'] = {'lambda_2': l2_cser, 'test_acc': acc_cser}
    print(f"CSER -> Lambda_2: {l2_cser:.4f}, Test Acc: {acc_cser:.4f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'CiteSeer'])
    args = parser.parse_args()
    
    run_end_to_end(args.dataset)
