import torch
import networkx as nx
import numpy as np
import csv
from torch_geometric.datasets import WebKB
from torch_geometric.utils import to_networkx, subgraph
import torch_geometric.transforms as T

from jacobian import SimpleGCN, compute_jacobian_norms
from mosr import calculate_mosr
from metrics import effective_resistance

def main():
    datasets = [('WebKB', 'Texas'), ('WebKB', 'Cornell')]
    configs = [
        {'K': 2, 'normalize': True},
        {'K': 2, 'normalize': False},
        {'K': 3, 'normalize': True},
        {'K': 3, 'normalize': False},
        {'K': 4, 'normalize': True},
        {'K': 4, 'normalize': False}
    ]
    
    results = []
    
    for cat, name in datasets:
        print(f"Processing {name}...")
        transform = T.NormalizeFeatures()
        dataset = WebKB(root='/tmp/' + name, name=name, transform=transform)
        data = dataset[0]
        
        G_raw = to_networkx(data, to_undirected=True)
        lcc = max(nx.connected_components(G_raw), key=len)
        node_idx = torch.tensor(sorted(list(lcc)))
        data.x = data.x[node_idx]
        data.y = data.y[node_idx]
        edge_index_raw, _ = subgraph(node_idx, data.edge_index, relabel_nodes=True)
        data.edge_index = edge_index_raw
        
        G_lcc = nx.Graph()
        G_lcc.add_nodes_from(range(data.x.size(0)))
        edges = [(edge_index_raw[0][i].item(), edge_index_raw[1][i].item()) for i in range(edge_index_raw.shape[1])]
        G_lcc.add_edges_from(edges)
        G_lcc.remove_edges_from(nx.selfloop_edges(G_lcc))
        
        er = {}
        for u, v in G_lcc.edges():
            r = effective_resistance(G_lcc, u, v)
            er[(u, v)] = r
        
        # We need a subgraph for MOSR calculation to define the stratum.
        # Let's use the 2-core.
        G_core = nx.k_core(G_lcc, k=2)
        if G_core.number_of_nodes() == 0:
            print(f"No 2-core for {name}, using LCC.")
            G_core = G_lcc
            
        for config in configs:
            K = config['K']
            normalize = config['normalize']
            print(f"  Evaluating K={K}, normalize={normalize}...")
            
            torch.manual_seed(42)
            model = SimpleGCN(in_channels=dataset.num_features, hidden_channels=64, num_layers=K, normalize=normalize)
            model.eval()
            
            jacobian_norms = compute_jacobian_norms(G_lcc, model, feature_dim=dataset.num_features)
            
            mosr, num, den, auroc, ap = calculate_mosr(G_lcc, G_core, er, jacobian_norms, q=25, lower_is_bottleneck=False)
            
            print(f"    MOSR: {mosr:.4f} | AUROC: {auroc:.4f} | AP: {ap:.4f}")
            results.append({
                'Dataset': name,
                'K': K,
                'Normalize': normalize,
                'MOSR': mosr,
                'AUROC': auroc,
                'AP': ap
            })
            
    with open('mosr_sensitivity.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Dataset', 'K', 'Normalize', 'MOSR', 'AUROC', 'AP'])
        writer.writeheader()
        writer.writerows(results)
        
    print("Saved to mosr_sensitivity.csv")

if __name__ == '__main__':
    main()
