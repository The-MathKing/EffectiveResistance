import torch
import networkx as nx
from torch_geometric.datasets import Planetoid, WikiCS, WebKB
from torch_geometric.utils import to_networkx
import numpy as np
import json
import os
import csv

from metrics import waf3_exact, compute_ollivier_ricci, effective_resistance, commute_time, link_resistance_curvature
from jacobian import SimpleGCN, compute_jacobian_norms

def get_dataset(name):
    if name in ['Cora', 'CiteSeer']:
        dataset = Planetoid(root='/tmp/' + name, name=name)
    elif name == 'WikiCS':
        dataset = WikiCS(root='/tmp/WikiCS')
    elif name in ['Cornell', 'Texas']:
        dataset = WebKB(root='/tmp/WebKB', name=name)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return dataset

def run_dump(dataset_name, csv_writer, counts_dict):
    print(f"Processing {dataset_name}...")
    dataset = get_dataset(dataset_name)
    data = dataset[0]
    
    G = to_networkx(data, to_undirected=True)
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    
    ebc = nx.edge_betweenness_centrality(G)
    ebc_undirected = {}
    for (u, v), val in ebc.items():
        ebc_undirected[(u, v)] = val
        ebc_undirected[(v, u)] = val

    ebc_vals = list(ebc.values())
    p33 = np.percentile(ebc_vals, 33)
    p67 = np.percentile(ebc_vals, 67)
    
    def get_bucket(u, v):
        val = ebc_undirected[(u, v)]
        if val <= p33: return "Low Betweenness (Intra-cluster)"
        if val <= p67: return "Medium Betweenness"
        return "High Betweenness (Inter-cluster)"

    # Metrics
    orc_curvs = compute_ollivier_ricci(G)
    waf3_curvs = {}
    res_curvs = {}
    ct_curvs = {}
    lrc_curvs = {}
    
    for u, v in G.edges():
        c_waf3 = waf3_exact(G, u, v)
        r = effective_resistance(G, u, v)
        c_time = commute_time(G, u, v)
        lrc = link_resistance_curvature(G, u, v)
        
        waf3_curvs[(u, v)] = c_waf3; waf3_curvs[(v, u)] = c_waf3
        res_curvs[(u, v)] = -r; res_curvs[(v, u)] = -r
        ct_curvs[(u, v)] = -c_time; ct_curvs[(v, u)] = -c_time
        lrc_curvs[(u, v)] = lrc; lrc_curvs[(v, u)] = lrc
        
    hybrid_curvs = {}
    if len(G.edges()) > 0:
        w_min, w_max = min(waf3_curvs.values()), max(waf3_curvs.values())
        r_min, r_max = min(res_curvs.values()), max(res_curvs.values())
        for u, v in G.edges():
            w_norm = (waf3_curvs[(u, v)] - w_min) / (w_max - w_min + 1e-8)
            r_norm = (res_curvs[(u, v)] - r_min) / (r_max - r_min + 1e-8)
            h = w_norm + r_norm
            hybrid_curvs[(u, v)] = h; hybrid_curvs[(v, u)] = h

    jaco_norms_avg = {e: 0.0 for e in G.edges()}
    jaco_norms_avg.update({(v, u): 0.0 for u, v in G.edges()})
    
    # Use deterministic seed for reproducibility of these exact numbers
    torch.manual_seed(42)
    runs = 3
    for run in range(runs):
        model = SimpleGCN(in_channels=dataset.num_features, hidden_channels=64, num_layers=2)
        model.eval()
        jaco = compute_jacobian_norms(G, model, feature_dim=dataset.num_features)
        for e, val in jaco.items():
            if e in jaco_norms_avg:
                jaco_norms_avg[e] += val / runs

    # Write CSV
    for u, v in G.edges():
        csv_writer.writerow({
            'dataset': dataset_name,
            'u': u,
            'v': v,
            'betweenness_bucket': get_bucket(u, v),
            'jaco_norm': jaco_norms_avg[(u, v)],
            'orc': orc_curvs.get((u, v), 0),
            'waf3': waf3_curvs[(u, v)],
            'eff_resistance': -res_curvs[(u, v)], # positive R
            'commute_time': -ct_curvs[(u, v)],
            'lrc': lrc_curvs[(u, v)],
            'hybrid': hybrid_curvs[(u, v)]
        })

    metrics = {
        "ORC": orc_curvs,
        "WAF3": waf3_curvs,
        "Eff. Resistance (neg)": res_curvs,
        "Commute Time (neg)": ct_curvs,
        "Link Res. Curv": lrc_curvs,
        "Hybrid (WAF3+EffRes)": hybrid_curvs
    }

    edge_subsets = {
        "All": G.edges(),
        "Low Betweenness (Intra-cluster)": [(u, v) for u, v in G.edges() if get_bucket(u, v) == "Low Betweenness (Intra-cluster)"],
        "Medium Betweenness": [(u, v) for u, v in G.edges() if get_bucket(u, v) == "Medium Betweenness"],
        "High Betweenness (Inter-cluster)": [(u, v) for u, v in G.edges() if get_bucket(u, v) == "High Betweenness (Inter-cluster)"]
    }

    counts_dict[dataset_name] = {}
    for subset_name, edges in edge_subsets.items():
        counts_dict[dataset_name][subset_name] = {}
        G_sub = nx.Graph()
        G_sub.add_edges_from(edges)
        
        for q in [10, 25]:
            counts_dict[dataset_name][subset_name][f"q={q}"] = {}
            j_global = list(jaco_norms_avg.values())
            j_threshold = np.percentile(j_global, q)
            
            for m_name, m_dict in metrics.items():
                if not m_dict: continue
                m_global = list(m_dict.values())
                c_threshold = np.percentile(m_global, q)
                
                num, den = 0, 0
                for u, v in G_sub.edges():
                    c = m_dict.get((u, v))
                    j = jaco_norms_avg.get((u, v))
                    if c is not None and j is not None:
                        is_tb = (j <= j_threshold)
                        is_flag = (c <= c_threshold)
                        if is_tb:
                            den += 1
                            if not is_flag: num += 1
                
                counts_dict[dataset_name][subset_name][f"q={q}"][m_name] = {
                    "numerator": num,
                    "denominator": den,
                    "mosr": num/den if den > 0 else 0.0
                }

if __name__ == "__main__":
    with open('dumped_metrics.csv', 'w', newline='') as csvfile:
        fieldnames = ['dataset', 'u', 'v', 'betweenness_bucket', 'jaco_norm', 'orc', 'waf3', 'eff_resistance', 'commute_time', 'lrc', 'hybrid']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        counts = {}
        for ds in ['Cornell', 'Texas']:
            run_dump(ds, writer, counts)
            
        with open('dumped_counts.json', 'w') as f:
            json.dump(counts, f, indent=2)
