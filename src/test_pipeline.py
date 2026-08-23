import networkx as nx
from metrics import waf3_exact, compute_ollivier_ricci, effective_resistance
from jacobian import SimpleGCN, compute_jacobian_norms
from mosr import calculate_mosr
from graphs import build_Gc
import torch

def test_pipeline():
    print("Testing pipeline on small G_c graph...")
    
    # 1. Build graph
    G = build_Gc(n=4, m=4)
    print(f"Graph generated: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # 2. Curvatures
    print("Computing WAF3...")
    waf3_curvs = {}
    for u, v in G.edges():
        c = waf3_exact(G, u, v)
        waf3_curvs[(u, v)] = c
        waf3_curvs[(v, u)] = c
        
    print("Computing Ollivier-Ricci...")
    orc_curvs = compute_ollivier_ricci(G)
    
    print("Computing Effective Resistance (negative as curvature proxy)...")
    res_curvs = {}
    for u, v in G.edges():
        # we negate it so smaller resistance = higher 'curvature' / less bottleneck
        c = -effective_resistance(G, u, v)
        res_curvs[(u, v)] = c
        res_curvs[(v, u)] = c
        
    # 3. Jacobian
    print("Computing Jacobian ground truth...")
    model = SimpleGCN(in_channels=16, hidden_channels=16, num_layers=2)
    model.eval() # Eval mode
    jaco_norms = compute_jacobian_norms(G, model, feature_dim=16)
    
    # 4. MOSR
    print("Calculating MOSR...")
    try:
        mosr_waf3 = calculate_mosr(G, waf3_curvs, jaco_norms, q=25)
        print(f"MOSR25 WAF3: {mosr_waf3:.4f}")
    except Exception as e:
        print("MOSR WAF3 failed:", e)
        
    try:
        mosr_orc = calculate_mosr(G, orc_curvs, jaco_norms, q=25)
        print(f"MOSR25 ORC: {mosr_orc:.4f}")
    except Exception as e:
        print("MOSR ORC failed:", e)
        
    try:
        mosr_res = calculate_mosr(G, res_curvs, jaco_norms, q=25)
        print(f"MOSR25 EFF RESIST: {mosr_res:.4f}")
    except Exception as e:
        print("MOSR EFF RESIST failed:", e)

if __name__ == "__main__":
    test_pipeline()
