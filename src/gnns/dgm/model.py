import os
import torch
from torch.nn import ModuleList
from torch_geometric.nn import EdgeConv, DenseGCNConv, DenseGATConv, GCNConv, GATConv
import pytorch_lightning as pl
from argparse import Namespace

from gnns.dgm.layers import *
if (not os.environ.get("USE_KEOPS")) or os.environ.get("USE_KEOPS") == "False":
    from gnns.dgm.layers_dense import *


class DGM_Model(pl.LightningModule):
    def __init__(self, hparams):
        super(DGM_Model, self).__init__()

        if type(hparams) is not Namespace:
            hparams = Namespace(**hparams)

        self.save_hyperparameters(hparams)
        conv_layers = hparams.conv_layers
        fc_layers = hparams.fc_layers
        dgm_layers = hparams.dgm_layers
        k = hparams.k

        self.use_continuous = getattr(hparams, 'use_continuous_dgm', False)
        self.pooling = getattr(hparams, 'pooling', "mean")

        self.graph_f = ModuleList()
        self.node_g = ModuleList()
        for i, (dgm_l, conv_l) in enumerate(zip(dgm_layers, conv_layers)):
            if len(dgm_l) > 0:
                if self.use_continuous:
                    # cDGM: fully differentiable soft adjacency matrix
                    if 'ffun' not in hparams or hparams.ffun == 'gcn':
                        self.graph_f.append(
                            DGM_c(GCNConv(dgm_l[0], dgm_l[-1]), k=hparams.k, distance=hparams.distance))
                    if hparams.ffun == 'gat':
                        self.graph_f.append(
                            DGM_c(GATConv(dgm_l[0], dgm_l[-1]), k=hparams.k, distance=hparams.distance))
                    if hparams.ffun == 'mlp':
                        self.graph_f.append(
                            DGM_c(MLP(dgm_l), k=hparams.k, distance=hparams.distance))
                    if hparams.ffun == 'knn':
                        self.graph_f.append(
                            DGM_c(Identity(retparam=0), k=hparams.k, distance=hparams.distance))
                else:
                    # dDGM: sparse k-NN with Gumbel sampling
                    if 'ffun' not in hparams or hparams.ffun == 'gcn':
                        self.graph_f.append(
                            DGM_d(GCNConv(dgm_l[0], dgm_l[-1]), k=hparams.k, distance=hparams.distance))
                    elif hparams.ffun == 'gat':
                        self.graph_f.append(
                            DGM_d(GATConv(dgm_l[0], dgm_l[-1]), k=hparams.k, distance=hparams.distance))
                    elif hparams.ffun == 'mlp':
                        self.graph_f.append(
                            DGM_d(MLP(dgm_l), k=hparams.k, distance=hparams.distance))
                    elif hparams.ffun == 'knn':
                        self.graph_f.append(
                            DGM_d(Identity(retparam=0), k=hparams.k, distance=hparams.distance))
            else:
                # NOTE: no embeddings in the beginning
                self.graph_f.append(Identity())

            # GNN Diffusion layer
            if self.use_continuous:
                # dense GNN layers for cDGM
                if hparams.gfun == 'gcn':
                    self.node_g.append(DenseGCNConv(conv_l[0], conv_l[1]))
                elif hparams.gfun == 'gat':
                    self.node_g.append(DenseGATConv(
                        conv_l[0], conv_l[1], heads=1))
                elif hparams.gfun == 'edgeconv':
                    self.node_g.append(DenseGCNConv(conv_l[0], conv_l[1]))
            else:
                # sparse GNN layers for dDGM
                if hparams.gfun == 'edgeconv':
                    conv_l_copy = conv_l.copy()
                    conv_l_copy[0] = conv_l_copy[0]*2
                    self.node_g.append(
                        EdgeConv(MLP(conv_l_copy), hparams.pooling))
                elif hparams.gfun == 'gcn':
                    self.node_g.append(GCNConv(conv_l[0], conv_l[1]))
                elif hparams.gfun == 'gat':
                    self.node_g.append(GATConv(conv_l[0], conv_l[1]))

        self.fc = MLP(fc_layers, final_activation=False)
        if hparams.pre_fc is not None and len(hparams.pre_fc) > 0:
            self.pre_fc = MLP(hparams.pre_fc, final_activation=True)
        self.avg_accuracy = None

        # torch lightning specific
        self.automatic_optimization = False
        self.debug = False

        self.edges = None

    def global_pooling(self, x, attention_mask):
        if self.pooling == 'mean':
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (x * mask_expanded).sum(dim=1)
            # [batch, 1]
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = sum_embeddings / sum_mask

        elif self.pooling == 'max':
            mask_expanded = attention_mask.unsqueeze(-1).float()
            x_masked = x.masked_fill(mask_expanded == 0, float('-inf'))
            pooled = x_masked.max(dim=1)[0]

        elif self.pooling == 'cls':
            pooled = x[:, 0, :]

        elif self.pooling == 'sum':
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask_expanded).sum(dim=1)

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return pooled

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=0.001
        )
        return optimizer

    def forward(self, x, attention_mask=None, edges=None, return_adj=False):
        if self.hparams.pre_fc is not None and len(self.hparams.pre_fc) > 0:
            x = self.pre_fc(x)

        b, n, d = x.shape

        graph_x = x.detach()  # [b, n, d]
        lprobslist = []

        # Start with no edges
        current_edges = None  # Will be [2, b*n*k] when created

        for layer_idx, (f, g) in enumerate(zip(self.graph_f, self.node_g)):

            # Apply DGM to ENTIRE batch at once
            graph_x, new_edges, lprobs = f(graph_x, current_edges, None)
            # if new_edges is not None:
            #     print(f"\n{'='*60}")
            #     print(f"Layer {layer_idx}: Generated {new_edges.shape[1]} edges")
            #     print(f"Edges shape: {new_edges.shape}")
            #     print(f"Edges min/max: {new_edges.min()}/{new_edges.max()}")
            #     print(f"Logprobs shape: {lprobs.shape if lprobs is not None else None}")
            #     print(f"Logprobs mean: {lprobs.mean() if lprobs is not None else None}")
            #     print(f"{'='*60}\n")

            # Update edges
            if new_edges is not None:
                current_edges = new_edges

            # Now split for GNN diffusion
            if current_edges is None:
                # No edges yet, skip diffusion
                pass
            elif self.use_continuous:
                # cDGM: dense adjacency [b, n, n]
                if current_edges.dim() == 2:
                    adj = current_edges.unsqueeze(0).repeat(b, 1, 1)
                else:
                    adj = current_edges

                x_gnn = torch.nn.functional.relu(g(x, adj))

                if x.shape[-1] == x_gnn.shape[-1]:
                    x = x + x_gnn
                else:
                    x = x_gnn
            else:
                # dDGM: sparse edges [2, b*n*k]
                # NOTE: we process each document separately for GNN
                current_edges_long = current_edges.long()

                x_list = []
                for i in range(b):
                    # extract edges for document i
                    doc_mask = (
                        current_edges_long[0] >= i*n) & (current_edges_long[0] < (i+1)*n)
                    doc_edges = current_edges_long[:, doc_mask]
                    doc_edges_local = doc_edges - i * \
                        n  # renormalize to [0, n-1]

                    x_doc = x[i]  # [n, d]
                    x_doc = torch.dropout(
                        x_doc, self.hparams.dropout, train=self.training)

                    x_doc_out = torch.nn.functional.relu(
                        g(x_doc, doc_edges_local))

                    if x_doc.shape[-1] == x_doc_out.shape[-1]:
                        x_doc = x_doc + x_doc_out
                    else:
                        x_doc = x_doc_out

                    x_list.append(x_doc)

                # stack back to batch
                x = torch.stack(x_list, dim=0)  # [b, n, d']

            # concat for next DGM layer
            graph_x = torch.cat([graph_x, x.detach()], dim=-1)

            if lprobs is not None:
                lprobslist.append(lprobs)

        if attention_mask is not None:
            x_pooled = self.global_pooling(x, attention_mask)
        else:
            x_pooled = x.mean(dim=1)

        logits = self.fc(x_pooled)

        if return_adj:
            if self.use_continuous:
                return logits, current_edges
            else:
                adj = torch.zeros(b, n, n, device=x.device)
                if current_edges is not None:
                    edges_long = current_edges.long()
                    batch_idx = edges_long[0] // n
                    src_idx = edges_long[0] % n
                    tgt_idx = edges_long[1] % n
                    adj[batch_idx, src_idx, tgt_idx] = 1.0
                return logits, adj

        return logits, torch.stack(lprobslist, -1) if len(lprobslist) > 0 else None

    def training_step(self, train_batch, batch_idx):
        optimizer = self.optimizers(use_pl_optimizer=True)
        optimizer.zero_grad()
    
        X = train_batch['node_features']
        y = train_batch['label']
        attention_mask = train_batch['attention_mask']
    
        pred, logprobs = self(X, attention_mask)
    
        loss = torch.nn.functional.cross_entropy(pred, y, label_smoothing=0.1)
        total_loss = loss
    
        correct_t = (pred.argmax(-1) == y).float().mean().item()
    
        # REGULARIZATION - MUCH WEAKER
        if logprobs is not None and not torch.isnan(logprobs).any():
            corr_pred = (pred.argmax(-1) == y).float().detach()
    
            if self.avg_accuracy is None:
                self.avg_accuracy = 0.5
    
            point_w = (corr_pred - self.avg_accuracy)
            graph_loss_per_sample = -logprobs.mean(dim=[1, 2])
            graph_loss = (point_w * graph_loss_per_sample).mean()
    
            edge_probs = torch.exp(logprobs)
            
            sparsity_loss = 0.001 * edge_probs.mean()
            connectivity_loss = -0.001 * torch.log(edge_probs.mean() + 1e-8)
    
            entropy = -(edge_probs * logprobs + (1-edge_probs) * torch.log(1-edge_probs + 1e-8))
            entropy_loss = -0.001 * entropy.mean()
    
            locality_loss = 0.001 * edge_probs.mean() if self.hparams.k >= 3 else 0
    
            total_graph_loss = graph_loss + sparsity_loss + connectivity_loss + entropy_loss + locality_loss
            
            if not torch.isnan(total_graph_loss):
                total_loss = total_loss + 0.1 * total_graph_loss
    
                self.log('train_graph_loss', graph_loss.detach())
                self.log('train_sparsity_loss', sparsity_loss.detach())
                self.log('train_connectivity_loss', connectivity_loss.detach())
    
            self.avg_accuracy = self.avg_accuracy * 0.95 + 0.05 * corr_pred.mean().item()
    
        total_loss.backward()
        optimizer.step()
    
        self.log('train_acc', 100 * correct_t, prog_bar=True)
        self.log('train_loss', total_loss.detach(), prog_bar=True)
    
        return total_loss

    def validation_step(self, train_batch, batch_idx):
        X = train_batch['node_features']
        y = train_batch['label']
        attention_mask = train_batch['attention_mask']

        pred, _ = self(X, attention_mask)
        pred = pred.softmax(-1)

        for i in range(1, self.hparams.test_eval):
            pred_, _ = self(X, attention_mask)
            pred += pred_.softmax(-1)

        pred = pred / self.hparams.test_eval

        loss = torch.nn.functional.cross_entropy(
            pred.log(), y, label_smoothing=0.1)
        correct_t = (pred.argmax(-1) == y).float().mean().item()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', 100 * correct_t, prog_bar=True)

        return loss

    def test_step(self, train_batch, batch_idx):
        X = train_batch['node_features']
        y = train_batch['label']
        attention_mask = train_batch['attention_mask']

        pred, _ = self(X, attention_mask)
        pred = pred.softmax(-1)

        for i in range(1, self.hparams.test_eval):
            pred_, _ = self(X, attention_mask)
            pred += pred_.softmax(-1)

        pred = pred / self.hparams.test_eval

        loss = torch.nn.functional.cross_entropy(
            pred.log(), y, label_smoothing=0.1)
        correct_t = (pred.argmax(-1) == y).float().mean().item()

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', 100 * correct_t, prog_bar=True)

        return loss
