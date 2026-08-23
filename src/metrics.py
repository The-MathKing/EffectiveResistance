import networkx as nx
import numpy as np
from GraphRicciCurvature.OllivierRicci import OllivierRicci

def waf3_exact(G: nx.Graph, u, v, f=lambda x: 1/(1+x)):
    """
    Computes exact Weighted Augmented Forman-3 (WAF3) Curvature for edge (u, v).
    """
    # N(v) is open neighborhood, B(v) is closed neighborhood
    N_u = set(G.neighbors(u))
    N_v = set(G.neighbors(v))
    B_u = N_u.union({u})
    B_v = N_v.union({v})
    
    intersection = B_u.intersection(B_v)
    term1 = sum(f(G.degree(i)) for i in intersection)
    
    N_u_minus_B_v = N_u - B_v
    N_v_minus_B_u = N_v - B_u
    
    term2 = sum(f(G.degree(i)) for i in N_u_minus_B_v)
    term3 = sum(f(G.degree(i)) for i in N_v_minus_B_u)
    
    return term1 - (term2 + term3)

def effective_resistance(G: nx.Graph, u, v):
    """
    Computes exact pairwise effective resistance R(u, v) using networkx.
    For small graphs, this uses the Moore-Penrose pseudoinverse internally.
    """
    return nx.resistance_distance(G, u, v)

def commute_time(G: nx.Graph, u, v):
    """
    Computes commute time C(u, v) = vol(G) * R(u, v).
    vol(G) is the sum of degrees of all nodes.
    """
    vol = sum(d for _, d in G.degree())
    r = effective_resistance(G, u, v)
    return vol * r

def link_resistance_curvature(G: nx.Graph, u, v):
    """
    Computes the 'Link resistance curvature' as defined in Chen et al. (Table 1).
    LinkResistanceCurvature(u, v) = (2 - sum_{i~u} w_ui - sum_{j~v} w_vj) / w_uv
    where W is the pseudoinverse of the symmetric-normalized Laplacian.
    """
    # 1. Compute symmetric-normalized Laplacian L_sym
    L_sym = nx.normalized_laplacian_matrix(G).toarray()
    # 2. Compute Moore-Penrose pseudoinverse W = L_sym^+
    W = np.linalg.pinv(L_sym)
    
    # 3. Create a mapping from node to index
    nodelist = list(G.nodes())
    node_idx = {node: i for i, node in enumerate(nodelist)}
    
    u_idx, v_idx = node_idx[u], node_idx[v]
    
    w_uv = W[u_idx, v_idx]
    
    sum_u = sum(W[u_idx, node_idx[i]] for i in G.neighbors(u))
    sum_v = sum(W[v_idx, node_idx[j]] for i, j in [(v, j) for j in G.neighbors(v)])
    
    if w_uv == 0:
        return 0 # Handle potential division by zero
        
    return (2 - sum_u - sum_v) / w_uv

def compute_ollivier_ricci(G: nx.Graph):
    """
    Computes Ollivier-Ricci curvature for all edges in G.
    Returns a dictionary mapping edges to curvature values.
    """
    orc = OllivierRicci(G, alpha=0.5, verbose="ERROR")
    orc.compute_ricci_curvature()
    # the curvature is stored as an edge attribute 'ricciCurvature'
    curvatures = {}
    for u, v, d in orc.G.edges(data=True):
        curvatures[(u, v)] = d.get('ricciCurvature', 0)
        curvatures[(v, u)] = d.get('ricciCurvature', 0)
    return curvatures
