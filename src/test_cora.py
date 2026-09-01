print("Starting benchmarks...")
import torch
import networkx as nx
from torch_geometric.datasets import Planetoid, WikiCS, WebKB
from torch_geometric.utils import to_networkx
import numpy as np
import json
import os

from metrics import waf3_exact, compute_ollivier_ricci, effective_resistance, commute_time, link_resistance_curvature
from jacobian import SimpleGCN, compute_jacobian_norms
from mosr import calculate_mosr

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

def run_benchmark(dataset_name, q_list=[10, 25]):
    print(f"\n================ Processing {dataset_name} ================")
    dataset = get_dataset(dataset_name)
    data = dataset[0]
    
    G = to_networkx(data, to_undirected=True)
    # LCC
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    print(f"LCC: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Compute edge betweenness
    print("Computing Edge Betweenness Centrality...")
    ebc = nx.edge_betweenness_centrality(G)
    # Map to undirected
    ebc_undirected = {}
    for (u, v), val in ebc.items():
        ebc_undirected[(u, v)] = val
        ebc_undirected[(v, u)] = val
        
    print("Computing Ollivier-Ricci...")
    orc_curvs = compute_ollivier_ricci(G)
    
    print("Computing WAF3...")
    waf3_curvs = {}
    for u, v in G.edges():
        c = waf3_exact(G, u, v)
        waf3_curvs[(u, v)] = c
        waf3_curvs[(v, u)] = c
        
    print("Computing Effective Resistance & variants...")
    res_curvs = {}
    ct_curvs = {}
    lrc_curvs = {}
    for u, v in G.edges():
        r = effective_resistance(G, u, v)
        c_time = commute_time(G, u, v)
        lrc = link_resistance_curvature(G, u, v)
        # Note: Effective resistance measures distance, so we negative it to act like curvature
        # where low = bottleneck
        res_curvs[(u, v)] = -r
        res_curvs[(v, u)] = -r
        ct_curvs[(u, v)] = -c_time
        ct_curvs[(v, u)] = -c_time
        lrc_curvs[(u, v)] = lrc
        lrc_curvs[(u, v)] = lrc
        lrc_curvs[(v, u)] = lrc
        
    print("Computing Hybrid Metric (WAF3 + EffRes)...")
    hybrid_curvs = {}
    if len(G.edges()) > 0:
        waf3_vals = list(waf3_curvs.values())
        res_vals = list(res_curvs.values())
        w_min, w_max = min(waf3_vals), max(waf3_vals)
        r_min, r_max = min(res_vals), max(res_vals)
        for u, v in G.edges():
            w = waf3_curvs[(u, v)]
            r = res_curvs[(u, v)]
            w_norm = (w - w_min) / (w_max - w_min + 1e-8)
            r_norm = (r - r_min) / (r_max - r_min + 1e-8)
            h = w_norm + r_norm
            hybrid_curvs[(u, v)] = h
            hybrid_curvs[(v, u)] = h
        
    print("Computing Jacobian norms...")
    jaco_norms_avg = {e: 0.0 for e in G.edges()}
    jaco_norms_avg.update({(v, u): 0.0 for u, v in G.edges()})
    
    torch.manual_seed(42)
    runs = 10
    for run in range(runs):
        model = SimpleGCN(in_channels=dataset.num_features, hidden_channels=64, num_layers=2)
        model.eval()
        jaco = compute_jacobian_norms(G, model, feature_dim=dataset.num_features)
        for e, val in jaco.items():
            if e in jaco_norms_avg:
                jaco_norms_avg[e] += val / runs
                
    # Evaluate MOSR stratified by betweenness
    ebc_vals = list(ebc.values())
    p33 = np.percentile(ebc_vals, 33)
    p67 = np.percentile(ebc_vals, 67)
    
    # We create subgraphs / edge subsets
    edge_subsets = {
        "All": G.edges(),
        "Low Betweenness (Intra-cluster)": [(u, v) for u, v in G.edges() if ebc_undirected[(u, v)] <= p33],
        "Medium Betweenness": [(u, v) for u, v in G.edges() if p33 < ebc_undirected[(u, v)] <= p67],
        "High Betweenness (Inter-cluster)": [(u, v) for u, v in G.edges() if ebc_undirected[(u, v)] > p67]
    }
    
    metrics = {
        "ORC": orc_curvs,
        "WAF3": waf3_curvs,
        "Eff. Resistance (neg)": res_curvs,
        "Commute Time (neg)": ct_curvs,
        "Link Res. Curv": lrc_curvs,
        "Hybrid (WAF3+EffRes)": hybrid_curvs
    }
    
    results = {}
    for subset_name, edges in edge_subsets.items():
        print(f"\n--- {subset_name} ---")
        results[subset_name] = {}
        # We need a sub-graph representation for mosr to iterate over edges
        G_sub = nx.Graph()
        G_sub.add_edges_from(edges)
        
        for q in q_list:
            print(f"  q={q}")
            results[subset_name][f"q={q}"] = {}
            for m_name, m_dict in metrics.items():
                if not m_dict:
                    continue
                
                score, num, denom = calculate_mosr(G, G_sub, m_dict, jaco_norms_avg, q=q)
                print(f"    {m_name:<24} : {score:.4f} (num={num}, den={denom})")
                results[subset_name][f"q={q}"][m_name] = {
                    "mosr": float(score),
                    "num": int(num),
                    "denom": int(denom)
                }
                
    return results

if __name__ == "__main__":
    datasets = ['Cornell', 'Texas'] # Subset for time
    # 'CiteSeer' and 'WikiCS' can be added if time permits
    all_results = {}
    for ds in datasets:
        all_results[ds] = run_benchmark(ds)
        
    os.makedirs('results', exist_ok=True)
    with open('results/benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("Done. Saved to results/benchmark_results.json")
