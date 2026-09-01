import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx, subgraph
import csv

from train_rewired_gnn import GCN, create_masks, train_eval, compute_cser_rewired_edges

def compute_spectral_gap(G):
    if G.number_of_nodes() == 0:
        return 0.0
    L = nx.normalized_laplacian_matrix(G)
    # We want the second smallest eigenvalue (Fiedler value)
    try:
        # eigsh finds the largest magnitude by default, use which='SM' for smallest
        # L is positive semi-definite, eigenvalues are >= 0
        eigenvalues, _ = eigsh(L, k=2, which='SM', tol=1e-2)
        return eigenvalues[1]
    except:
        return 0.0

def main():
    print("--- Phase 3: Deep Parameter Sweeps ---")
    dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')
    data = dataset[0]
    data = create_masks(data)
    
    G_raw = to_networkx(data, to_undirected=True)
    lcc = max(nx.connected_components(G_raw), key=len)
    
    node_idx = torch.tensor(sorted(list(lcc)))
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
        
    G_lcc = nx.Graph()
    G_lcc.add_nodes_from(range(data.x.size(0)))
    edges = [(edge_index_raw[0][i].item(), edge_index_raw[1][i].item()) for i in range(edge_index_raw.shape[1])]
    G_lcc.add_edges_from(edges)
    G_lcc.remove_edges_from(nx.selfloop_edges(G_lcc))
    
    budget_taus = [0.01, 0.05, 0.10, 0.15, 0.20]
    tau_results = []
    
    for tau in budget_taus:
        print(f"Sweeping tau = {tau}...")
        new_edges = compute_cser_rewired_edges(G_lcc, tau)
        edge_index_rewired = torch.cat([edge_index_raw, new_edges], dim=1)
        
        G_rewired = G_lcc.copy()
        new_edges_list = [(new_edges[0][i].item(), new_edges[1][i].item()) for i in range(new_edges.shape[1])]
        G_rewired.add_edges_from(new_edges_list)
        
        gap = compute_spectral_gap(G_rewired)
        
        accs = []
        for seed in range(5):
            torch.manual_seed(seed)
            model = GCN(dataset.num_features, 64, dataset.num_classes)
            acc = train_eval(model, data, edge_index_rewired)
            accs.append(acc)
            
        mean_acc = np.mean(accs)
        tau_results.append({
            'budget_tau': tau,
            'spectral_gap': gap,
            'accuracy': mean_acc * 100
        })
        print(f"tau={tau} | Gap: {gap:.4f} | Acc: {mean_acc*100:.2f}%")
        
    # K-core sweep with fixed tau=0.05
    k_cores = [2, 3, 4]
    k_results = []
    
    for k in k_cores:
        print(f"Sweeping k = {k}...")
        # Modified compute_cser_rewired_edges to support custom k
        G_core = nx.k_core(G_lcc, k=k)
        if G_core.number_of_nodes() == 0:
            print(f"k={k} core is empty.")
            continue
            
        core_nodes = list(G_core.nodes())
        L_core = nx.laplacian_matrix(G_core).todense()
        L_pinv = np.linalg.pinv(L_core)
        
        R_vals = []
        for i in range(len(core_nodes)):
            for j in range(i+1, len(core_nodes)):
                if not G_core.has_edge(core_nodes[i], core_nodes[j]):
                    R_eff = L_pinv[i, i] + L_pinv[j, j] - 2 * L_pinv[i, j]
                    R_vals.append((R_eff, core_nodes[i], core_nodes[j]))
                    
        R_vals.sort(key=lambda x: x[0])
        n_add = int(0.05 * G_lcc.number_of_edges())
        top_k = R_vals[:n_add]
        
        new_edges_list = [(u, v) for _, u, v in top_k]
        # undirected edges
        new_edges_list += [(v, u) for _, u, v in top_k]
        
        new_edges_tensor = torch.tensor(new_edges_list, dtype=torch.long).t().contiguous()
        if new_edges_tensor.numel() == 0:
            new_edges_tensor = torch.empty((2, 0), dtype=torch.long)
            
        edge_index_rewired = torch.cat([edge_index_raw, new_edges_tensor], dim=1)
        
        G_rewired = G_lcc.copy()
        G_rewired.add_edges_from(new_edges_list)
        
        gap = compute_spectral_gap(G_rewired)
        
        accs = []
        for seed in range(5):
            torch.manual_seed(seed)
            model = GCN(dataset.num_features, 64, dataset.num_classes)
            acc = train_eval(model, data, edge_index_rewired)
            accs.append(acc)
            
        mean_acc = np.mean(accs)
        k_results.append({
            'k_core': k,
            'spectral_gap': gap,
            'accuracy': mean_acc * 100
        })
        print(f"k={k} | Gap: {gap:.4f} | Acc: {mean_acc*100:.2f}%")
        
    with open('sweeps_tau.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['budget_tau', 'spectral_gap', 'accuracy'])
        writer.writeheader()
        writer.writerows(tau_results)
        
    with open('sweeps_k.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['k_core', 'spectral_gap', 'accuracy'])
        writer.writeheader()
        writer.writerows(k_results)
        
    print("Saved to sweeps_tau.csv and sweeps_k.csv")

if __name__ == '__main__':
    main()
