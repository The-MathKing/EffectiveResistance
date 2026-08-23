import networkx as nx
import numpy as np

from graphs import build_Gc, build_dense_Gc, build_connected_dense_Gc
from metrics import waf3_exact, effective_resistance

def build_density_sweep_Gc(n, m, p):
    """
    Builds Gc with n, m, but adds edges between the m nodes of each u_i 
    with probability p.
    p=0 is original Gc (dangling leaves).
    p=1 is build_dense_Gc (cliques).
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

if __name__ == "__main__":
    n = 4
    m = 10
    
    ps = [0.0, 0.2, 0.5, 0.8, 1.0]
    
    print(f"Sweep density p at fixed n={n}, m={m}")
    for p in ps:
        G = build_density_sweep_Gc(n, m, p)
        r = effective_resistance(G, "s", "t")
        w = waf3_exact(G, "s", "t")
        print(f"p={p:.1f} | ER={r:.4f} | WAF3={w:.4f}")
