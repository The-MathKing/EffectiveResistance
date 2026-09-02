import networkx as nx
from torch_geometric.datasets import WebKB
from torch_geometric.utils import to_networkx

def main():
    dataset = WebKB(root='/tmp/Texas', name='Texas')
    data = dataset[0]
    G_raw = to_networkx(data, to_undirected=True)
    largest_cc = max(nx.connected_components(G_raw), key=len)
    G_raw = G_raw.subgraph(largest_cc).copy()
    G_raw.remove_edges_from(nx.selfloop_edges(G_raw))
    
    bridges = list(nx.bridges(G_raw))
    print(f"Total cut-edges (bridges) in Texas LCC: {len(bridges)}")
    
    degree_1_bridges = 0
    non_leaf_bridges = 0
    
    for u, v in bridges:
        if G_raw.degree(u) == 1 or G_raw.degree(v) == 1:
            degree_1_bridges += 1
        else:
            non_leaf_bridges += 1
            
    print(f"  - Degree-1 leaf edges: {degree_1_bridges}")
    print(f"  - Non-leaf bridges: {non_leaf_bridges}")

if __name__ == '__main__':
    main()
