import networkx as nx
from torch_geometric.datasets import Planetoid, WebKB, Amazon
from torch_geometric.utils import to_networkx
from ogb.nodeproppred import NodePropPredDataset

def get_core_frac(dataset_name):
    if dataset_name in ['Cora', 'CiteSeer']:
        dataset = Planetoid(root='/tmp/' + dataset_name, name=dataset_name)
    elif dataset_name in ['Texas', 'Cornell', 'Wisconsin']:
        dataset = WebKB(root='/tmp/' + dataset_name, name=dataset_name)
    elif dataset_name in ['Chameleon', 'Squirrel']:
        from torch_geometric.datasets import WikipediaNetwork
        dataset = WikipediaNetwork(root='/tmp/' + dataset_name, name=dataset_name)
    elif dataset_name == 'Photo':
        dataset = Amazon(root='/tmp/Photo', name='Photo')
    elif dataset_name == 'ogbn-arxiv':
        dataset = NodePropPredDataset(name='ogbn-arxiv')
        data = dataset[0]
        G_raw = to_networkx(data, to_undirected=True)
        largest_cc = max(nx.connected_components(G_raw), key=len)
        G_raw = G_raw.subgraph(largest_cc).copy()
        G_raw.remove_edges_from(nx.selfloop_edges(G_raw))
        core_2 = nx.k_core(G_raw, k=2)
        pruned = G_raw.number_of_nodes() - core_2.number_of_nodes()
        return pruned, G_raw.number_of_nodes(), pruned / G_raw.number_of_nodes()
    
    data = dataset[0]
    G_raw = to_networkx(data, to_undirected=True)
    largest_cc = max(nx.connected_components(G_raw), key=len)
    G_raw = G_raw.subgraph(largest_cc).copy()
    G_raw.remove_edges_from(nx.selfloop_edges(G_raw))
    core_2 = nx.k_core(G_raw, k=2)
    pruned = G_raw.number_of_nodes() - core_2.number_of_nodes()
    return pruned, G_raw.number_of_nodes(), pruned / G_raw.number_of_nodes()

for name in ['Cora', 'CiteSeer', 'Texas', 'Cornell', 'Wisconsin', 'Chameleon', 'Squirrel', 'Photo']:
    p, t, f = get_core_frac(name)
    print(f"{name}: {p} / {t} ({f*100:.1f}%)")
