import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score

def calculate_mosr(G_full: nx.Graph, G_sub: nx.Graph, curvatures: dict, jacobian_norms: dict, q: int = 25, lower_is_bottleneck: bool = True):
    """
    Calculates the Missed Over-Squashing Ratio (MOSR) using a global rank-based threshold.
    Also computes AUROC and Average Precision for detecting ground-truth bottlenecks within the sub-graph.
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
        return 0.0, 0, 0, 0.5, 0.0
        
    # Global thresholds
    j_threshold = np.percentile(global_jaco, q)
    if lower_is_bottleneck:
        c_threshold = np.percentile(global_curv, q)
    else:
        c_threshold = np.percentile(global_curv, 100 - q)
    
    numerator = 0
    denominator = 0
    
    stratum_curv = []
    stratum_jaco_binary = []
    
    for u, v in G_sub.edges():
        c = curvatures.get((u, v), curvatures.get((v, u)))
        j = jacobian_norms.get((u, v), jacobian_norms.get((v, u)))
        
        if c is not None and j is not None:
            is_true_bottleneck = (j <= j_threshold)
            if lower_is_bottleneck:
                is_flagged = (c <= c_threshold)
                # Predictor score: lower c indicates bottleneck. So negate for AUROC.
                stratum_curv.append(-c)
            else:
                is_flagged = (c >= c_threshold)
                # Predictor score: higher c indicates bottleneck.
                stratum_curv.append(c)
                
            stratum_jaco_binary.append(int(is_true_bottleneck))
            
            if is_true_bottleneck:
                denominator += 1
                if not is_flagged:
                    numerator += 1
                    
    mosr = (numerator / denominator) if denominator > 0 else 0.0
    
    auroc = 0.5
    ap = 0.0
    if len(set(stratum_jaco_binary)) > 1:
        auroc = roc_auc_score(stratum_jaco_binary, stratum_curv)
        ap = average_precision_score(stratum_jaco_binary, stratum_curv)
        
    return mosr, numerator, denominator, auroc, ap
