import numpy as np
import networkx as nx

def calculate_mosr(G_full: nx.Graph, G_sub: nx.Graph, curvatures: dict, jacobian_norms: dict, q: int = 25):
    """
    Calculates the Missed Over-Squashing Ratio (MOSR) using a global rank-based threshold.
    """
    global_curv = []
    global_jaco = []
    for u, v in G_full.edges():
        c = curvatures.get((u, v), curvatures.get((v, u)))
        j = jacobian_norms.get((u, v), jacobian_norms.get((v, u)))
        if c is not None and j is not None:
            global_curv.append(c)
            global_jaco.append(j)
            
    if not global_curv:
        return 0.0
        
    # Global thresholds: bottom q percentile
    j_threshold = np.percentile(global_jaco, q)
    c_threshold = np.percentile(global_curv, q)
    
    numerator = 0
    denominator = 0
    
    for u, v in G_sub.edges():
        c = curvatures.get((u, v), curvatures.get((v, u)))
        j = jacobian_norms.get((u, v), jacobian_norms.get((v, u)))
        
        if c is not None and j is not None:
            is_true_bottleneck = (j <= j_threshold)
            is_flagged = (c <= c_threshold)
            
            if is_true_bottleneck:
                denominator += 1
                if not is_flagged:
                    numerator += 1
                    
    if denominator == 0:
        return 0.0, numerator, denominator
        
    return (numerator / denominator), numerator, denominator
