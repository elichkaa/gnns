import torch
import torch.nn as nn


def pairwise_euclidean_distances(X: torch.Tensor, dim=-1):
    if X.dim() == 2:
        # [1, num_nodes, feature_dim]
        X = X.unsqueeze(0)

    # [batch, num_nodes, num_nodes]
    D = torch.cdist(X, X) ** 2
    return D

def pairwise_poincare_distances(X: torch.Tensor, dim=-1):
    # Poincarè disk distance r=1 (Hyperbolic)
    if X.dim() == 2:
        # [1, num_nodes, feature_dim]
        X = X.unsqueeze(0)
    X_norm = (X**2).sum(dim, keepdim=True)
    X_norm = (X_norm.sqrt()-1).relu() + 1
    X = X/(X_norm*(1+1e-2))
    X_norm = (X**2).sum(dim, keepdim=True)

    pq = torch.cdist(X, X)**2
    D = torch.arccosh(
        1e-6+1+2*pq/((1-X_norm)*(1-X_norm.transpose(-1, -2))))**2
    return D

class Identity(nn.Module):

    def __init__(self, retparam=None):
        super(Identity, self).__init__()
        self.retparam = retparam

    def forward(self, x, edge_index=None):
        if self.retparam is not None:
            params = (x, edge_index) if edge_index is not None else (x,)
            return params[self.retparam]
        return x

class MLP(nn.Module):
    def __init__(self, layer_sizes, final_activation=False, dropout=0.0):
        super(MLP, self).__init__()
        layers = []

        for i in range(1, len(layer_sizes)):
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))

            if i < len(layer_sizes) - 1 or final_activation:
                layers.append(nn.LeakyReLU(0.1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x, edge_index=None):
        return self.mlp(x)


class cDGM(nn.Module):
    def __init__(self, embed_f, distance="euclidean"):
        super(cDGM, self).__init__()
        # f_θ
        self.embed_f = embed_f
        if distance == "euclidean":
            self.distance = pairwise_euclidean_distances
        else:
            self.distance = pairwise_poincare_distances
        # temperature t controls sharpness of connections
        self.t = nn.Parameter(torch.tensor(
            1.0 if distance == "hyperbolic" else 4.0))
        # threshold T controls connection radius
        self.T = nn.Parameter(torch.tensor(0.5))
        # normalization
        self.scale = nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        self.centroid = None
    
    def forward(self, X: torch.Tensor, A_0=None):
        # X̃ = f_θ(X)
        X_tilde = self.embed_f(X, A_0)
        # [batch, num_nodes, feature_dim]
        if X_tilde.dim() == 2:
            X_tilde = X_tilde.unsqueeze(0)
        
        if self.scale < 0 or self.centroid is None:
            self.centroid = X_tilde.mean(dim=(0, 1), keepdim=True).detach()
            self.scale.data = (0.9 / (X_tilde - self.centroid).abs().max()).detach()
        
        X_norm = (X_tilde - self.centroid) * self.scale
        # distances
        D = self.distance(X_norm)
        # adjacency matrix A using sigmoid thresholding
        # a_ij = σ(t * (T - d_ij²))
        temp = torch.clamp(self.t, -5, 5)
        A = torch.sigmoid(temp * (self.T.abs() - D))
        return X_tilde, A.squeeze(0) if X_tilde.size(0) == 1 else A, None


class dDGM(nn.Module):
    def __init__(self, embed_f, k=5, distance="euclidean", sparse=True):
        super(dDGM, self).__init__()

        # f_θ
        self.embed_f = embed_f
        self.k = k
        if distance == "euclidean":
            self.distance = pairwise_euclidean_distances
        else:
            self.distance = pairwise_poincare_distances
        self.sparse = sparse

        # temperature
        self.t = nn.Parameter(torch.tensor(
            1.0 if distance == "hyperbolic" else 4.0))

    def forward(self, X: torch.Tensor, A_0=None):
        # X̃ = f_θ(X)
        X_tilde = self.embed_f(X, A_0)

        # [batch, num_nodes, feature_dim]
        if X_tilde.dim() == 2:
            X_tilde = X_tilde.unsqueeze(0)

        batch_size, num_nodes, feature_dim = X_tilde.shape

        # distances
        D = pairwise_euclidean_distances(X_tilde)

        edge_index, logprobs = self._sample_knn_graph(D, batch_size, num_nodes)

        return X_tilde, edge_index, logprobs

    def _sample_knn_graph(self, D, batch_size, num_nodes):
        temp = torch.clamp(self.t, -5, 5)
        logits = D * torch.exp(temp) 
        
        # gumbel-max
        q = torch.rand_like(logits) + 1e-8

        # gumbel-top-k
        gumbel_noise = -torch.log(-torch.log(q))
        perturbed_logits = logits - gumbel_noise

        # top-k neighbors (smallest distances)
        _, indices = torch.topk(-perturbed_logits, self.k, dim=2)

        logprobs = -logits.gather(2, indices)

        rows = torch.arange(
            num_nodes, device=logits.device).view(1, num_nodes, 1)
        rows = rows.repeat(batch_size, 1, self.k)

        # [batch, 2, num_nodes * k]
        edges = torch.stack([indices.view(batch_size, -1),
                            rows.view(batch_size, -1)], dim=1)

        if self.sparse:
            offset = torch.arange(batch_size, device=logits.device) * num_nodes
            offset = offset.view(batch_size, 1, 1)

            edges_flat = edges + offset 
            edge_index = edges_flat.transpose(0, 1).reshape(
                2, -1)  # [2, batch * num_nodes * k]
        else:
            edge_index = edges

        return edge_index, logprobs
