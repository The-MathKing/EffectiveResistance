import networkx as nx
import torch
import numpy as np
from torch_geometric.datasets import WebKB
from torch_geometric.utils import to_networkx
from metrics import effective_resistance, waf3_exact, link_resistance_curvature
from jacobian import SimpleGCN, compute_jacobian_norms
from mosr import calculate_mosr

def run_bootstrap_mosr(dataset_name, B=200):
    dataset = WebKB(root='/tmp/WebKB', name=dataset_name)
    data = dataset[0]
    G_full = to_networkx(data, to_undirected=True)
    largest_cc = max(nx.connected_components(G_full), key=len)
    G = G_full.subgraph(largest_cc).copy()

    # Precompute metrics
    res_curvs = {e: -effective_resistance(G, *e) for e in G.edges()}
    res_curvs.update({(v, u): val for (u, v), val in res_curvs.items()})
    
    waf_curvs = {e: waf3_exact(G, *e) for e in G.edges()}
    waf_curvs.update({(v, u): val for (u, v), val in waf_curvs.items()})

    # Precompute Jaco norms
    torch.manual_seed(42)
    jaco_norms_avg = {e: 0.0 for e in G.edges()}
    jaco_norms_avg.update({(v, u): 0.0 for u, v in G.edges()})
    runs = 3
    for _ in range(runs):
        model = SimpleGCN(in_channels=dataset.num_features, hidden_channels=64, num_layers=2)
        model.eval()
        jaco = compute_jacobian_norms(G, model, feature_dim=dataset.num_features)
        for (u, v), val in jaco.items():
            if u != v:
                if (u, v) in jaco_norms_avg:
                    jaco_norms_avg[(u, v)] += val / runs
                if (v, u) in jaco_norms_avg:
                    jaco_norms_avg[(v, u)] += val / runs

    ebc = nx.edge_betweenness_centrality(G)
    ebc_undirected = {}
    for (u, v), val in ebc.items():
        ebc_undirected[(u, v)] = val
        ebc_undirected[(v, u)] = val

    p33 = np.percentile(list(ebc.values()), 33)
    p67 = np.percentile(list(ebc.values()), 67)

    low_edges = [(u, v) for u, v in G.edges() if ebc_undirected[(u, v)] <= p33]
    high_edges = [(u, v) for u, v in G.edges() if ebc_undirected[(u, v)] > p67]
    
    G_low = nx.Graph()
    G_low.add_edges_from(low_edges)
    G_high = nx.Graph()
    G_high.add_edges_from(high_edges)

    def bootstrap(G_sub, metric_dict, q=25):
        # We need the base true bottlenecks from the whole graph
        # Wait, calculate_mosr recalculates it. Let's just use calculate_mosr to get the true base numbers
        # actually, the MOSR is |E_q_sub - E_c_sub| / |E_q_sub|.
        # We can just extract E_q_sub and E_c_sub
        
        c_vals = list(metric_dict.values())
        c_threshold = np.percentile(c_vals, q)
        flagged_edges = set()
        for e in G.edges(): # check all directional edges
            if metric_dict[e] <= c_threshold:
                flagged_edges.add(e)
                flagged_edges.add((e[1], e[0]))

        j_vals = list(jaco_norms_avg.values())
        j_threshold = np.percentile(j_vals, q)
        true_bottlenecks = set()
        for e in G.edges():
            if jaco_norms_avg[e] <= j_threshold:
                true_bottlenecks.add(e)
                true_bottlenecks.add((e[1], e[0]))

        # Subgraph edges (directional)
        sub_edges = list(G_sub.edges())
        sub_edges_dir = sub_edges + [(v, u) for u, v in sub_edges]
        
        # Filter true bottlenecks to subgraph
        E_q_sub = [e for e in sub_edges_dir if e in true_bottlenecks]
        
        if len(E_q_sub) == 0:
            return 0, 0, 0, [0.0, 0.0]

        # Which ones were missed?
        missed = [e for e in E_q_sub if e not in flagged_edges]
        
        # Now bootstrap the denominator (E_q_sub)
        n = len(E_q_sub)
        boot_mosrs = []
        for _ in range(B):
            # sample with replacement from E_q_sub
            sample = np.random.choice(len(E_q_sub), size=n, replace=True)
            # count how many of the sampled true bottlenecks were missed
            missed_count = sum(1 for idx in sample if E_q_sub[idx] not in flagged_edges)
            boot_mosrs.append(missed_count / n)
            
        ci_lower = np.percentile(boot_mosrs, 2.5)
        ci_upper = np.percentile(boot_mosrs, 97.5)
        mean_mosr = len(missed) / n
        return mean_mosr, len(missed), n, [ci_lower, ci_upper]

    print(f"--- {dataset_name} (q=25) ---")
    w_low_mean, w_low_num, w_low_den, w_low_ci = bootstrap(G_low, waf_curvs)
    r_low_mean, r_low_num, r_low_den, r_low_ci = bootstrap(G_low, res_curvs)
    print(f"Low Betw (n={w_low_den}): WAF3 MOSR: {w_low_mean:.4f} 95% CI [{w_low_ci[0]:.4f}, {w_low_ci[1]:.4f}]")
    print(f"Low Betw (n={r_low_den}): ER MOSR:   {r_low_mean:.4f} 95% CI [{r_low_ci[0]:.4f}, {r_low_ci[1]:.4f}]")

    w_hi_mean, w_hi_num, w_hi_den, w_hi_ci = bootstrap(G_high, waf_curvs)
    r_hi_mean, r_hi_num, r_hi_den, r_hi_ci = bootstrap(G_high, res_curvs)
    print(f"High Betw (n={w_hi_den}): WAF3 MOSR: {w_hi_mean:.4f} 95% CI [{w_hi_ci[0]:.4f}, {w_hi_ci[1]:.4f}]")
    print(f"High Betw (n={r_hi_den}): ER MOSR:   {r_hi_mean:.4f} 95% CI [{r_hi_ci[0]:.4f}, {r_hi_ci[1]:.4f}]")

np.random.seed(42)
run_bootstrap_mosr('Texas')
run_bootstrap_mosr('Cornell')
