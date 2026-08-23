import networkx as nx

def build_Gc(n: int, m: int) -> nx.Graph:
    """
    Constructs the G_c counterexample family graph from Chen et al. Theorem 5.
    
    Args:
        n (int): Branching factor at depth 1 (number of 1-hop nodes u_i connecting to s and t)
        m (int): Branching factor at depth 2 (number of 2-hop leaf nodes hanging off each u_i)
        
    Returns:
        nx.Graph: The constructed counterexample graph.
    """
    G = nx.Graph()
    s, t = "s", "t"
    G.add_edge(s, t)
    
    for i in range(n):
        u_i = f"u{i}"
        G.add_edge(s, u_i)
        G.add_edge(t, u_i)
        for j in range(m):
            v_ij = f"v{i}_{j}"
            G.add_edge(u_i, v_ij)
            
    return G

def build_dense_Gc(n: int, m: int) -> nx.Graph:
    """
    Constructs a variant of G_c where the m leaf nodes for each u_i form a dense clique.
    This demonstrates that Effective Resistance is blind to intra-cluster density
    due to potential symmetry.
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
        
        # Add internal clique connectivity
        for j1 in range(m):
            for j2 in range(j1+1, m):
                G.add_edge(v_nodes[j1], v_nodes[j2])
                
    return G

def build_connected_dense_Gc(n: int, m: int) -> nx.Graph:
    """
    Constructs a variant of G_c where a single pool of m nodes connects to all u_i 
    and forms a dense clique. This demonstrates blindness to fully connected symmetric 
    intra-cluster bottlenecks.
    """
    G = nx.Graph()
    s, t = "s", "t"
    G.add_edge(s, t)
    
    u_nodes = [f"u{i}" for i in range(n)]
    for u in u_nodes:
        G.add_edge(s, u)
        G.add_edge(t, u)
        
    v_nodes = [f"v_{j}" for j in range(m)]
    for v in v_nodes:
        for u in u_nodes:
            G.add_edge(u, v)
            
    for j1 in range(m):
        for j2 in range(j1+1, m):
            G.add_edge(v_nodes[j1], v_nodes[j2])
            
    return G

if __name__ == "__main__":
    # Small test
    g = build_Gc(n=2, m=2)
    print(f"Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}")
