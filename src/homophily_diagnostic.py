import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork, Amazon
from torch_geometric.utils import to_networkx, homophily
import csv
import time

from train_rewired_gnn import GCN, create_masks, train_eval, compute_cser_rewired_edges

def main():
    datasets_dict = {
        'Planetoid': ['Cora', 'CiteSeer'],
        'WebKB': ['Texas', 'Cornell', 'Wisconsin'],
        'WikipediaNetwork': ['Chameleon', 'Squirrel'],
        'Amazon': ['Computers', 'Photo']
    }
    
    results = []
    
    for cat, names in datasets_dict.items():
        for name in names:
            print(f"--- Diagnostic for {name} ---")
            if cat == 'Planetoid':
                dataset = Planetoid(root='/tmp/' + name, name=name)
            elif cat == 'WebKB':
                dataset = WebKB(root='/tmp/' + name, name=name)
            elif cat == 'WikipediaNetwork':
                dataset = WikipediaNetwork(root='/tmp/' + name, name=name)
            elif cat == 'Amazon':
                dataset = Amazon(root='/tmp/' + name, name=name)
                
            data = dataset[0]
            data = create_masks(data)
            
            # Subgraph LCC
            G_raw = to_networkx(data, to_undirected=True)
            lcc = max(nx.connected_components(G_raw), key=len)
            G_raw = G_raw.subgraph(lcc).copy()
            
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
                
            h = homophily(edge_index_raw, data.y)
            
            print(f"Homophily: {h:.4f}")
            
            # CSER rewiring
            # Skip Amazon datasets if they are too big (memory limit on dense pinv)
            if G_raw.number_of_nodes() > 8000:
                print(f"Skipping {name} due to N={G_raw.number_of_nodes()} > 8000")
                continue
                
            new_edges = compute_cser_rewired_edges(G_raw, 0.05)
            edge_index_rewired = torch.cat([edge_index_raw, new_edges], dim=1)
            
            raw_accs = []
            cser_accs = []
            for seed in range(5): # 5 seeds to save time
                torch.manual_seed(seed)
                model_raw = GCN(dataset.num_features, 64, dataset.num_classes)
                raw_accs.append(train_eval(model_raw, data, edge_index_raw))
                
                torch.manual_seed(seed)
                model_cser = GCN(dataset.num_features, 64, dataset.num_classes)
                cser_accs.append(train_eval(model_cser, data, edge_index_rewired))
            
            delta_acc = (np.mean(cser_accs) - np.mean(raw_accs)) * 100
            print(f"Delta Acc: {delta_acc:.2f}%")
            
            results.append({
                'Dataset': name,
                'Homophily': h,
                'Delta_Accuracy': delta_acc
            })
            
    with open('diagnostic_sweep.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Dataset', 'Homophily', 'Delta_Accuracy'])
        writer.writeheader()
        writer.writerows(results)
    print("Saved to diagnostic_sweep.csv")

if __name__ == '__main__':
    main()
