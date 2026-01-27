import torch
from gnns.dgm.layers import *
from torch import nn

class DGM_d(nn.Module):
    def __init__(self, embed_f, k=5, distance=pairwise_euclidean_distances, sparse=True):
        super(DGM_d, self).__init__()
        self.sparse = sparse
        # NOTE: lower from .4
        self.temperature = nn.Parameter(torch.tensor(1.).float())
        self.embed_f = embed_f
        self.k = k
        self.debug = False
        
        if distance == 'euclidean':
            self.distance = pairwise_euclidean_distances
        else:
            self.distance = pairwise_poincare_distances
    
    def forward(self, x, A, not_used=None, fixedges=None):
        # x: [b, n, d]
        b, n, d = x.shape
        device = x.device
        
        if A is None:
            batch_offset = torch.arange(b, device=device).view(b, 1) * n
            node_ids = torch.arange(n, device=device).view(1, n)
            global_node_ids = (batch_offset + node_ids).flatten()
            A = torch.stack([global_node_ids, global_node_ids], dim=0)
        
        x_flat = x.view(b * n, d)
        
        x_flat = torch.clamp(x_flat, min=-1e6, max=1e6)
        
        x_out = self.embed_f(x_flat, A)
        x = x_out.view(b, n, -1) if x_out.dim() == 2 else x_out
        
        x = torch.clamp(x, min=-1e6, max=1e6)
        
        if self.training:
            if fixedges is not None:
                return x, fixedges, torch.zeros(b, n, self.k, dtype=torch.float, device=device)
            
            # D: [b, n, n]
            D, _x = self.distance(x) 
            
            D = torch.where(torch.isnan(D) | torch.isinf(D),
                          torch.tensor(1e6, device=D.device, dtype=D.dtype),
                          D)
            
            edges_hat, logprobs = self.sample_without_replacement(D, b, n)
        else:
            with torch.no_grad():
                if fixedges is not None:
                    return x, fixedges, torch.zeros(b, n, self.k, dtype=torch.float, device=device)
                D, _x = self.distance(x)
                
                D = torch.where(torch.isnan(D) | torch.isinf(D),
                              torch.tensor(1e6, device=D.device, dtype=D.dtype),
                              D)
                
                edges_hat, logprobs = self.sample_without_replacement(D, b, n)
        
        return x, edges_hat, logprobs
    
    def sample_without_replacement(self, logits, b, n):
        # logits: [b, n, n]
        logits = torch.clamp(logits, min=-1e10, max=1e10)
        
        temp = torch.exp(torch.clamp(self.temperature, -5, 5))
        logits = logits * temp
        
        # NOTE: prevent log(0)
        q = torch.rand_like(logits)
        q = torch.clamp(q, min=1e-10, max=1.0 - 1e-10)
        
        log_q = torch.log(q)
        log_neg_log_q = torch.log(-log_q)
        lq = logits - log_neg_log_q
        
        lq = torch.where(torch.isnan(lq) | torch.isinf(lq),
                        torch.tensor(-1e10, device=lq.device, dtype=lq.dtype),
                        lq)
        # [b, n, k]
        logprobs, indices = torch.topk(-lq, self.k, dim=-1)  
        
        logprobs = -logprobs
        
        logprobs = torch.clamp(logprobs, min=-1e10, max=0)
        
        if self.sparse:
            source_nodes = torch.arange(n, device=logits.device).view(1, n, 1).expand(b, n, self.k)
            batch_offset = torch.arange(b, device=logits.device).view(b, 1, 1) * n
            
            source_global = (source_nodes + batch_offset).reshape(-1)  # [b*n*k]
            target_global = (indices + batch_offset).reshape(-1)  # [b*n*k]
            
            edges = torch.stack([source_global, target_global], dim=0)  # [2, b*n*k]
            
            return edges, logprobs  # [2, b*n*k], [b, n, k]
        
        return indices, logprobs  # [b, n, k], [b, n, k]