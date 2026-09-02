import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import csv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx, subgraph
from torch_geometric.nn import GATConv

from train_rewired_gnn import create_masks, compute_cser_rewired_edges

class ProfilerGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels // 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=0.6)
        
    def forward(self, x, edge_index, return_attention_weights=False):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        
        if return_attention_weights:
            out, (edge_index_out, alpha) = self.conv2(x, edge_index, return_attention_weights=True)
            return out, edge_index_out, alpha
        else:
            x = self.conv2(x, edge_index)
            return x

def train_gat(model, data, edge_index, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def main():
    print("Loading CiteSeer...")
    dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')
    data = dataset[0]
    data = create_masks(data)
    
    # Isolate Largest Connected Component
    G_raw = to_networkx(data, to_undirected=True)
    lcc = max(nx.connected_components(G_raw), key=len)
    node_idx = torch.tensor(sorted(list(lcc)))
    edge_index_raw, _ = subgraph(node_idx, data.edge_index, relabel_nodes=True)
    data.x = data.x[node_idx]
    data.y = data.y[node_idx]
    if data.train_mask.dim() == 2:
        data.train_mask = data.train_mask[node_idx, :]
    else:
        data.train_mask = data.train_mask[node_idx]
        
    G_lcc = nx.Graph()
    G_lcc.add_nodes_from(range(data.x.size(0)))
    edges = [(edge_index_raw[0][i].item(), edge_index_raw[1][i].item()) for i in range(edge_index_raw.shape[1])]
    G_lcc.add_edges_from(edges)
    G_lcc.remove_edges_from(nx.selfloop_edges(G_lcc))
    
    print("Computing CSER bridges...")
    # top 5% budget
    new_edges = compute_cser_rewired_edges(G_lcc, 0.05)
    edge_index_rewired = torch.cat([edge_index_raw, new_edges], dim=1)
    
    # Create set of injected edges for fast lookup
    injected_set = set()
    for i in range(new_edges.shape[1]):
        u, v = new_edges[0][i].item(), new_edges[1][i].item()
        injected_set.add((u, v))
        injected_set.add((v, u))
        
    print("Training GAT over 5 seeds...")
    results = []
    
    for seed in range(5):
        torch.manual_seed(seed * 10)
        model = ProfilerGAT(dataset.num_features, 64, dataset.num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        
        for epoch in range(200):
            train_gat(model, data, edge_index_rewired, optimizer)
            
        model.eval()
        with torch.no_grad():
            _, edge_index_out, alpha = model(data.x, edge_index_rewired, return_attention_weights=True)
            
        alpha_np = alpha.squeeze().cpu().numpy()
        
        for i in range(edge_index_out.shape[1]):
            u = edge_index_out[0][i].item()
            v = edge_index_out[1][i].item()
            w = alpha_np[i]
            
            is_injected = (u, v) in injected_set
            edge_type = "CSER_Injected" if is_injected else "Native"
            results.append({'seed': seed, 'u': u, 'v': v, 'edge_type': edge_type, 'attention_weight': w})
        
    with open('attention_weights.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['seed', 'u', 'v', 'edge_type', 'attention_weight'])
        writer.writeheader()
        writer.writerows(results)
        
    print("Saved to attention_weights.csv")

if __name__ == '__main__':
    main()
