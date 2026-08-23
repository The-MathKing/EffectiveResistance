import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os

from graphs import build_Gc
from metrics import effective_resistance, waf3_exact, commute_time, link_resistance_curvature

def build_density_sweep_Gc(n, m, p):
    """
    Builds Gc with n, m, but adds edges between the m nodes of each u_i 
    with probability p.
    p=0 is original Gc (dangling leaves).
    p=1 is fully dense intra-cluster clique (build_dense_Gc).
    """
    G = nx.Graph()
    s, t = "s", "t"
    G.add_edge(s, t)
    
    for i in range(n):
        u_i = f"u{i}"
        G.add_edge(s, u_i)
        G.add_edge(t, u_i)
        
        v_nodes = [f"v{i}_{j}" for j in range(m)]
        for v in v_nodes:
            G.add_edge(u_i, v)
        
        # Add random internal edges with prob p
        for j1 in range(m):
            for j2 in range(j1+1, m):
                if np.random.rand() < p:
                    G.add_edge(v_nodes[j1], v_nodes[j2])
                    
    return G

def run_sweep():
    n_fixed = 4
    
    # --- Experiment A: Varying m (bridge-node degree) ---
    ms = [2, 4, 8, 16, 32, 64]
    res_m = []
    waf3_m = []
    ct_m = []
    lrc_m = []
    
    for m in ms:
        G = build_density_sweep_Gc(n_fixed, m, p=0.0) # Original dangling leaves
        r = effective_resistance(G, "s", "t")
        w = waf3_exact(G, "s", "t")
        c = commute_time(G, "s", "t")
        l = link_resistance_curvature(G, "s", "t")
        res_m.append(r)
        waf3_m.append(w)
        ct_m.append(c / 100) # scale down for plot
        lrc_m.append(l)
        
    # --- Experiment B: Varying p (internal density) ---
    m_fixed = 10
    ps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    res_p = []
    waf3_p = []
    ct_p = []
    lrc_p = []
    
    # We use multiple seeds to smooth out the randomness of p
    n_seeds = 5
    for p in ps:
        r_avg, w_avg, c_avg, l_avg = 0.0, 0.0, 0.0, 0.0
        for seed in range(n_seeds):
            np.random.seed(seed + int(p*100))
            G = build_density_sweep_Gc(n_fixed, m_fixed, p)
            r_avg += effective_resistance(G, "s", "t")
            w_avg += waf3_exact(G, "s", "t")
            c_avg += commute_time(G, "s", "t")
            l_avg += link_resistance_curvature(G, "s", "t")
        res_p.append(r_avg / n_seeds)
        waf3_p.append(w_avg / n_seeds)
        ct_p.append((c_avg / n_seeds) / 100) # scale down
        lrc_p.append(l_avg / n_seeds)
        
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Exp A
    ax1 = axes[0]
    ax1.plot(ms, waf3_m, marker='s', label='WAF3 Curvature', color='blue')
    ax1.plot(ms, res_m, marker='o', label='Effective Resistance', color='red')
    ax1.plot(ms, lrc_m, marker='^', label='Link Resistance Curvature', color='orange')
    ax1.plot(ms, ct_m, marker='v', label='Commute Time (/100)', color='purple', linestyle=':')
    ax1.axhline(y=2/(n_fixed+2), color='red', linestyle='--', alpha=0.5, label='ER Theoretical')
    ax1.set_xlabel('Bridge-node inflation (m)')
    ax1.set_ylabel('Metric Value')
    ax1.set_title(f'Exp A: Varying Cluster Size (n={n_fixed}, p=0)\nWAF3 detects inflation, ER/LRC/CT are blind')
    ax1.legend()
    ax1.grid(True)
    
    # Subplot 2: Exp B
    ax2 = axes[1]
    ax2.plot(ps, waf3_p, marker='s', label='WAF3 Curvature', color='blue')
    ax2.plot(ps, res_p, marker='o', label='Effective Resistance', color='red')
    ax2.plot(ps, lrc_p, marker='^', label='Link Resistance Curvature', color='orange')
    ax2.plot(ps, ct_p, marker='v', label='Commute Time (/100)', color='purple', linestyle=':')
    ax2.axhline(y=2/(n_fixed+2), color='red', linestyle='--', alpha=0.5, label='ER Theoretical')
    ax2.set_xlabel('Internal density probability (p)')
    ax2.set_ylabel('Metric Value')
    ax2.set_title(f'Exp B: Varying Internal Density (n={n_fixed}, m={m_fixed})\nAll metrics are entirely blind to density')
    ax2.legend()
    ax2.grid(True)
    
    os.makedirs('results', exist_ok=True)
    plt.tight_layout()
    plt.savefig('results/sweep_density_and_m.png', dpi=300)
    print("Saved figure to results/sweep_density_and_m.png")

if __name__ == "__main__":
    run_sweep()
