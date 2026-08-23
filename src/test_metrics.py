import networkx as nx
from metrics import waf3_exact, effective_resistance, link_resistance_curvature, compute_ollivier_ricci
from graphs import build_Gc
import numpy as np

def test_metrics_on_Gc():
    print("Testing metrics on G_c counterexample graphs...")
    print(f"{'n':<5} | {'m':<5} | {'WAF3':<10} | {'Eff. Resist':<15} | {'Link Res Curv':<15} | {'Ollivier-Ricci':<15}")
    print("-" * 75)
    
    n = 4 # Keep n fixed
    for m in [2, 4, 8, 16, 32, 64]:
        G = build_Gc(n, m)
        s, t = "s", "t"
        
        # Calculate WAF3
        w3 = waf3_exact(G, s, t)
        
        # Calculate Effective Resistance
        R = effective_resistance(G, s, t)
        
        # Calculate Link Resistance Curvature
        lrc = link_resistance_curvature(G, s, t)
        
        # Calculate Ollivier-Ricci
        orc_dict = compute_ollivier_ricci(G)
        orc = orc_dict.get((s, t), 0)
        
        print(f"{n:<5} | {m:<5} | {w3:<10.4f} | {R:<15.4f} | {lrc:<15.4f} | {orc:<15.4f}")

if __name__ == "__main__":
    test_metrics_on_Gc()
