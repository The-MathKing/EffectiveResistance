import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import networkx as nx
from torch_geometric.utils import from_networkx
import numpy as np

class SimpleGCN(nn.Module):
    def __init__(self, in_channels=64, hidden_channels=64, num_layers=2, normalize=True):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(pyg_nn.GCNConv(in_channels, hidden_channels, normalize=normalize))
        for _ in range(num_layers - 1):
            self.convs.append(pyg_nn.GCNConv(hidden_channels, hidden_channels, normalize=normalize))
            
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = torch.relu(x)
        return x

def compute_jacobian_norms(G: nx.Graph, model: nn.Module, feature_dim=64):
    """
    Computes the Frobenius norm of the Jacobian ∂h_v^(L) / ∂h_u^(0) for all pairs (u, v) in G.
    Returns a dictionary mapping (u, v) to the Jacobian norm.
    """
    # Convert to PyG data
    data = from_networkx(G)
    
    # Initialize random features
    x = torch.randn((G.number_of_nodes(), feature_dim), requires_grad=True)
    
    # Forward pass
    out = model(x, data.edge_index)
    
    jacobian_norms = {}
    nodes = list(G.nodes())
    node_idx = {node: i for i, node in enumerate(nodes)}
    
    # For every target node v, we compute the gradient of its output w.r.t input x
    for v_idx, v in enumerate(nodes):
        # We want grad of out[v_idx] w.r.t x
        # Since out[v_idx] is a vector, we compute jacobian for each component
        # Alternatively, using autograd to compute the full jacobian matrix for node v
        
        # We can just sum the components of out[v_idx] and backward, 
        # but to get the true Frobenius norm of the Jacobian matrix, we need each column.
        # Let's iterate over the feature dimension of the output.
        
        out_dim = out.shape[1]
        v_jacobian = torch.zeros((out_dim, G.number_of_nodes(), feature_dim))
        
        for d in range(out_dim):
            grad_out = torch.zeros_like(out)
            grad_out[v_idx, d] = 1.0
            
            # retain_graph=True because we do multiple backward passes on the same graph
            x.grad = None
            out.backward(grad_out, retain_graph=True)
            
            # x.grad contains ∂ out[v_idx, d] / ∂ x[u_idx, c]
            v_jacobian[d] = x.grad.clone()
            
        # v_jacobian has shape (out_dim, num_nodes, in_dim)
        # Frobenius norm for each node u is the norm over (out_dim, in_dim)
        for u_idx, u in enumerate(nodes):
            norm = torch.linalg.matrix_norm(v_jacobian[:, u_idx, :], ord='fro').item()
            jacobian_norms[(u, v)] = norm
            
    symmetric_norms = {}
    for u in nodes:
        for v in nodes:
            symmetric_norms[(u, v)] = 0.5 * (jacobian_norms[(u, v)] + jacobian_norms[(v, u)])
            
    return symmetric_norms

def compute_diffusion_scores(G: nx.Graph, k_steps=2):
    """
    Computes a parameter-free structural propagation score based on the 
    k-step random walk transition matrix.
    """
    nodes = list(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes)
    D = np.sum(A, axis=1)
    D_inv = np.zeros_like(D)
    D_inv[D > 0] = 1.0 / D[D > 0]
    P = np.diag(D_inv) @ A
    P_k = np.linalg.matrix_power(P, k_steps)
    
    diffusion_scores = {}
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            diffusion_scores[(u, v)] = 0.5 * (P_k[i, j] + P_k[j, i])
            
    return diffusion_scores
