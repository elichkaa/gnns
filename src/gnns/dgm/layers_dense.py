import torch
from gnns.dgm.layers import *
from torch import nn


class DGM_d(nn.Module):
    def __init__(self, embed_f, k=5, distance=pairwise_euclidean_distances, sparse=True):
        super(DGM_d, self).__init__()

        self.sparse = sparse

        self.temperature = nn.Parameter(torch.tensor(
            1. if distance == "hyperbolic" else 4.).float())
        self.embed_f = embed_f
        self.centroid = None
        self.scale = None
        self.k = k

        self.debug = False
        if distance == 'euclidean':
            self.distance = pairwise_euclidean_distances
        else:
            self.distance = pairwise_poincare_distances

    def forward(self, x, A, not_used=None, fixedges=None):
        # A: [2, num_edges] (edge list)
        b, n, d = x.shape  # [batch_size, num_nodes, hidden_dim]
        device = x.device

        # initial edges
        if A is None or (isinstance(A, list) and (len(A) == 0 or A[0] is None)):
            batch_offset = torch.arange(
                b, device=device).view(b, 1) * n  # [b, 1]
            node_ids = torch.arange(n, device=device).view(1, n)  # [1, n]
            global_node_ids = (batch_offset + node_ids).flatten()  # [b*n]

            A = torch.stack([global_node_ids, global_node_ids],
                            dim=0)  # [2, b*n]

        x_flat = x.view(b * n, d)  # [b*n, d]
        x_out = self.embed_f(x_flat, A)  # [b*n, d']
        x = x_out.view(b, n, -1)  # [b, n, d']

        # sample k-nearest neighbors per document
        if self.training:
            if fixedges is not None:
                return x, fixedges, torch.zeros(
                    b, n, self.k, dtype=torch.float, device=device
                )

            # distance matrix per document
            D, _x = self.distance(x)
            edges_hat, logprobs = self.sample_without_replacement(D, b, n)

        else:
            with torch.no_grad():
                if fixedges is not None:
                    return x, fixedges, torch.zeros(
                        b, n, self.k, dtype=torch.float, device=device
                    )
                D, _x = self.distance(x)  # D: [b, n, n]
                edges_hat, logprobs = self.sample_without_replacement(D, b, n)

        if self.debug:
            self.D = D
            self.edges_hat = edges_hat
            self.logprobs = logprobs
            self.x = x

        return x, edges_hat, logprobs

    def sample_without_replacement(self, logits, b, n):
        # logits shape: [b, n, n] - separate distance matrix per document
        logits = logits * torch.exp(torch.clamp(self.temperature, -5, 5))

        q = torch.rand_like(logits) + 1e-8
        lq = (logits - torch.log(-torch.log(q)))
        # indices: [batch_size, num_nodes, k] - k nearest neighbors for each node (local indices 0 to n-1)
        logprobs, indices = torch.topk(-lq, self.k, dim=-1)  # [b, n, k]

        if self.sparse:
            # we need to convert local indices (0 to n-1) to global indices

            # source_nodes: [b, n, k]
            source_nodes = torch.arange(n, device=logits.device).view(
                1, n, 1).expand(b, n, self.k)

            batch_offset = torch.arange(
                b, device=logits.device).view(b, 1, 1) * n

            source_global = (
                source_nodes + batch_offset).reshape(-1)  # [b*n*k]
            target_global = (indices + batch_offset).reshape(-1)  # [b*n*k]

            # [2, b*n*k]
            edges = torch.stack([source_global, target_global], dim=0)

            # edges: [2, b*n*k], logprobs: [b, n, k]
            return edges, logprobs

        # indices: [b, n, k], logprobs: [b, n, k]
        return indices, logprobs
