import torch
from pykeops.torch import LazyTensor
from torch import nn


def pairwise_euclidean_distances(x, dim=-1):
    dist = torch.cdist(x, x)**2
    return dist, x


def pairwise_poincare_distances(x, dim=-1):
    x_norm = (x**2).sum(dim, keepdim=True)
    x_norm = (x_norm.sqrt()-1).relu() + 1
    x = x/(x_norm*(1+1e-2))
    x_norm = (x**2).sum(dim, keepdim=True)

    pq = torch.cdist(x, x)**2
    dist = torch.arccosh(
        1e-6+1+2*pq/((1-x_norm)*(1-x_norm.transpose(-1, -2))))**2
    return dist, x


def sparse_eye(size):
    """
    Returns the identity matrix as a sparse matrix
    """
    indices = torch.arange(0, size).long().unsqueeze(0).expand(2, size)
    values = torch.tensor(1.0).float().expand(size)
    cls = getattr(torch.sparse, values.type().split(".")[-1])
    return cls(indices, values, torch.Size([size, size]))


class DGM_d(nn.Module):
    def __init__(self, embed_f, k=5, distance="euclidean", sparse=True):
        super(DGM_d, self).__init__()

        self.sparse = sparse

        self.temperature = nn.Parameter(torch.tensor(
            1. if distance == "hyperbolic" else 4.).float())
        self.embed_f = embed_f
        self.centroid = None
        self.scale = None
        self.k = k
        self.distance = distance

        self.debug = False

    def forward(self, x, A, not_used=None, fixedges=None):
        # NOTE: b is batch_size
        b, n, d = x.shape
        device = x.device

        if A is None or (isinstance(A, list) and (len(A) == 0 or A[0] is None)):
            A = None

        # NOTE: apply embedding function to each document separately
        if isinstance(self.embed_f, nn.Module) and hasattr(self.embed_f, 'weight'):
            # MLP can process all at once
            x_flat = x.view(b * n, d)
            x_out = self.embed_f(x_flat, A)
            x = x_out.view(b, n, -1) if x_out.dim() == 2 else x_out
        else:
            # GNN processes each document separately
            x_list = []
            for i in range(b):
                doc_edges = A[:, A[0] >= i*n]
                # encoding
                doc_edges = doc_edges[:, doc_edges[0] <
                                      (i+1)*n]
                # renormalize [0, n)
                doc_edges_local = doc_edges - i*n

                x_doc = x[i]  # [n, d]
                x_doc_out = self.embed_f(x_doc, doc_edges_local)  # [n, d']
                x_list.append(x_doc_out)

            x = torch.stack(x_list, dim=0)  # [b, n, d']

        if self.training:
            if fixedges is not None:
                return x, fixedges, torch.zeros(b, n, self.k, dtype=torch.float, device=device)
            edges_hat, logprobs = self.sample_without_replacement(x)

        else:
            with torch.no_grad():
                if fixedges is not None:
                    return x, fixedges, torch.zeros(b, n, self.k, dtype=torch.float, device=device)
                edges_hat, logprobs = self.sample_without_replacement(x)

        if self.debug:
            if self.distance == "euclidean":
                D, _x = pairwise_euclidean_distances(x)
            if self.distance == "hyperbolic":
                D, _x = pairwise_poincare_distances(x)

            self.D = (
                D * torch.exp(torch.clamp(self.temperature, -5, 5))).detach().cpu()
            self.edges_hat = edges_hat.detach().cpu()
            self.logprobs = logprobs.detach().cpu()

        return x, edges_hat, logprobs

    def sample_without_replacement(self, x):

        b, n, _ = x.shape

        if self.distance == "euclidean":
            G_i = LazyTensor(x[:, :, None, :])    # (b, n, 1, d)
            X_j = LazyTensor(x[:, None, :, :])    # (b, 1, n, d)

            mD = ((G_i - X_j) ** 2).sum(-1)

            lq = mD * torch.exp(torch.clamp(self.temperature, -5, 5))
            indices = lq.argKmin(self.k, dim=2)  # [b, n, k]

            x1 = torch.gather(
                x, 1, indices.view(b, -1, 1).repeat(1, 1, x.shape[-1])).view(b, n, self.k, -1)
            x2 = x[:, :, None, :].repeat(1, 1, self.k, 1)
            logprobs = ((x1-x2).pow(2).sum(-1) * torch.exp(torch.clamp(
                self.temperature, -5, 5)))  # [b, n, k]

        if self.distance == "hyperbolic":
            x_norm = (x**2).sum(-1, keepdim=True)
            x_norm = (x_norm.sqrt()-1).relu() + 1
            x = x/(x_norm*(1+1e-2))
            x_norm = (x**2).sum(-1, keepdim=True)

            G_i = LazyTensor(x[:, :, None, :])    # (M**2, 1, 2)
            X_j = LazyTensor(x[:, None, :, :])    # (1, N, 2)

            G_i2 = LazyTensor(1-x_norm[:, :, None, :])    # (M**2, 1, 2)
            X_j2 = LazyTensor(1-x_norm[:, None, :, :])    # (1, N, 2)

            pq = ((G_i - X_j) ** 2).sum(-1)
            N = (G_i2*X_j2)
            XX = (1e-6+1+2*pq/N)
            mD = (XX+(XX**2-1).sqrt()).log()**2

            lq = mD * torch.exp(torch.clamp(self.temperature, -5, 5))
            indices = lq.argKmin(self.k, dim=2)  # [b, n, k]

            x1 = torch.gather(
                x, 1, indices.view(b, -1, 1).repeat(1, 1, x.shape[-1])).view(b, n, self.k, -1)
            x2 = x[:, :, None, :].repeat(1, 1, self.k, 1)

            x1_n = torch.gather(
                x_norm, 1, indices.view(b, -1, 1).repeat(1, 1, 1)).view(b, n, self.k, -1)
            x2_n = x_norm[:, :, None, :].repeat(1, 1, self.k, 1)

            pq = (x1-x2).pow(2).sum(-1)
            pqn = ((1-x1_n)*(1-x2_n)).sum(-1)
            XX = 1e-6+1+2*pq/pqn
            dist = torch.log(XX+(XX**2-1).sqrt())**2
            logprobs = (-dist * torch.exp(torch.clamp(self.temperature, -5, 5)))

            if self.debug:
                self._x = x.detach().cpu()+0

        source_nodes = torch.arange(n, device=x.device).view(
            1, n, 1).expand(b, n, self.k)
        batch_offset = torch.arange(b, device=x.device).view(b, 1, 1) * n

        source_global = (source_nodes + batch_offset).reshape(-1)
        target_global = (indices + batch_offset).reshape(-1)

        edges = torch.stack([source_global, target_global],
                            dim=0)  # [2, b*n*k]

        if self.sparse:
            return edges, logprobs  # [2, b*n*k], [b, n, k]

        return indices, logprobs


class DGM_c(nn.Module):
    input_dim = 4
    debug = False

    def __init__(self, embed_f, k=None, distance="euclidean"):
        super(DGM_c, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(1).float())
        self.threshold = nn.Parameter(torch.tensor(0.5).float())
        self.embed_f = embed_f
        self.centroid = None
        self.scale = None
        self.distance = distance

        self.scale = nn.Parameter(
            torch.tensor(-1).float(), requires_grad=False)
        self.centroid = nn.Parameter(torch.zeros(
            (1, 1, DGM_c.input_dim)).float(), requires_grad=False)

    def forward(self, x, A, not_used=None, fixedges=None):

        x = self.embed_f(x, A)

        if self.scale < 0:
            self.centroid.data = x.mean(-2, keepdim=True).detach()
            self.scale.data = (0.9/(x-self.centroid).abs().max()).detach()

        if self.distance == "hyperbolic":
            D, _x = pairwise_poincare_distances((x-self.centroid)*self.scale)
        else:
            D, _x = pairwise_euclidean_distances((x-self.centroid)*self.scale)

        A = torch.sigmoid(self.temperature*(self.threshold.abs()-D))

        if DGM_c.debug:
            self.A = A.data.cpu()
            self._x = _x.data.cpu()

#         self.A=A
#         A = A/A.sum(-1,keepdim=True)
        return x, A, None


class MLP(nn.Module):
    def __init__(self, layers_size, final_activation=False, dropout=0):
        super(MLP, self).__init__()
        layers = []
        for li in range(1, len(layers_size)):
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(layers_size[li-1], layers_size[li]))
            if li == len(layers_size)-1 and not final_activation:
                continue
            layers.append(nn.LeakyReLU(0.1))

        self.MLP = nn.Sequential(*layers)

    def forward(self, x, e=None):
        x = self.MLP(x)
        return x


class Identity(nn.Module):
    def __init__(self, retparam=None):
        self.retparam = retparam
        super(Identity, self).__init__()

    def forward(self, *params):
        if self.retparam is not None:
            return params[self.retparam]
        return params
