import torch
import networkx as nx
from torch_geometric.datasets import WebKB
from torch_geometric.utils import to_networkx
import numpy as np
import json
import traceback

from metrics import waf3_exact, compute_ollivier_ricci, effective_resistance, commute_time, link_resistance_curvature
from jacobian import SimpleGCN, compute_jacobian_norms

def run():
    counts_dict = {}
    for dataset_name in ['Cornell', 'Texas']:
        print(f"Processing {dataset_name}...")
        dataset = WebKB(root='/tmp/WebKB', name=dataset_name)
        G = to_networkx(dataset[0], to_undirected=True)
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

        orc_curvs = {} # compute_ollivier_ricci(G)
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
        w_min, w_max = min(waf3_curvs.values()), max(waf3_curvs.values())
        r_min, r_max = min(res_curvs.values()), max(res_curvs.values())
        for u, v in G.edges():
            w_norm = (waf3_curvs[(u, v)] - w_min) / (w_max - w_min + 1e-8)
            r_norm = (res_curvs[(u, v)] - r_min) / (r_max - r_min + 1e-8)
            h = w_norm + r_norm
            hybrid_curvs[(u, v)] = h; hybrid_curvs[(v, u)] = h

        jaco_norms_avg = {e: 0.0 for e in G.edges()}
        jaco_norms_avg.update({(v, u): 0.0 for u, v in G.edges()})
        
        torch.manual_seed(42)
        runs = 3
        for _ in range(runs):
            model = SimpleGCN(in_channels=dataset.num_features, hidden_channels=64, num_layers=2)
            model.eval()
            jaco = compute_jacobian_norms(G, model, feature_dim=dataset.num_features)
            for e, val in jaco.items():
                if e in jaco_norms_avg:
                    jaco_norms_avg[e] += val / runs

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
            counts_dict[dataset_name][subset_name]["subset_size"] = len(edges)
            
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

    with open('dumped_counts2.json', 'w') as f:
        json.dump(counts_dict, f, indent=2)
    print("Done")

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        traceback.print_exc()
