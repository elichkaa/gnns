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

        graph_x = x.detach()
        lprobslist = []

        # no initial edges
        current_edges = None

        for layer_idx, (f, g) in enumerate(zip(self.graph_f, self.node_g)):
            try:
                graph_x, new_edges, lprobs = f(
                    graph_x, current_edges, attention_mask)

                # update edges
                if new_edges is not None:
                    current_edges = new_edges

            except Exception as e:
                print(f"ERROR in DGM layer {layer_idx}: {e}")
                print(f"graph_x shape: {graph_x.shape}")
                print(
                    f"current_edges: {current_edges.shape if current_edges is not None else None}")
                raise

            # if layer_idx == 0 and current_edges is not None:
            #     self.edges = current_edges.clone()

            if current_edges is None:
                # no edges yet
                pass
            elif self.use_continuous:
                # cDGM: edges is dense adjacency matrix [b, n, n]
                if current_edges.dim() == 2:
                    # single adjacency matrix -> broadcast to batch
                    adj = current_edges.unsqueeze(0).repeat(b, 1, 1)
                else:
                    adj = current_edges

                x_gnn = torch.nn.functional.relu(g(x, adj))

                if x.shape[-1] == x_gnn.shape[-1]:
                    x = x + x_gnn
                else:
                    x = x_gnn
            else:
                # dDGM: sparse edge_index [2, b*n*k]
                current_edges_long = current_edges.long()

                b_curr, n_curr, d_curr = x.shape

                x_flat = x.view(b_curr * n_curr, d_curr)
                x_flat = torch.dropout(
                    x_flat, self.hparams.dropout, train=self.training)

                try:
                    x_gnn = torch.nn.functional.relu(
                        g(x_flat, current_edges_long))
                    x_gnn = x_gnn.view(b_curr, n_curr, -1)

                    # residual connection
                    if x.shape[-1] == x_gnn.shape[-1]:
                        x = x + x_gnn
                    else:
                        x = x_gnn

                except Exception as e:
                    print(f"x_flat shape: {x_flat.shape}")
                    print(f"edges shape: {current_edges_long.shape}")
                    print(
                        f"edges min/max: {current_edges_long.min()}/{current_edges_long.max()}")
                    raise

            graph_x = torch.cat([graph_x, x.detach()], -1)

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
        loss.backward()

        correct_t = (pred.argmax(-1) == y).float().mean().item()

        # REGULARIZATION
        if logprobs is not None:
            corr_pred = (pred.argmax(-1) == y).float().detach()

            if self.avg_accuracy is None:
                self.avg_accuracy = 0.5

            point_w = (corr_pred - self.avg_accuracy)
            graph_loss_per_sample = -logprobs.mean(dim=[1, 2])
            graph_loss = (point_w * graph_loss_per_sample).mean()

            # sparsity - penalize too many edges
            edge_probs = torch.exp(logprobs)
            sparsity_loss = 0.1 * edge_probs.mean()

            # connectivity - penalize too few edges
            connectivity_loss = -0.01 * torch.log(edge_probs.mean() + 1e-8)

            # entropy - encourage decisive edges
            entropy = -(edge_probs * logprobs + (1-edge_probs)
                        * torch.log(1-edge_probs + 1e-8))
            entropy_loss = -0.01 * entropy.mean()

            if self.hparams.k >= 3:
                locality_loss = 0.05 * edge_probs.mean()
            else:
                locality_loss = 0

            total_graph_loss = graph_loss + sparsity_loss + \
                connectivity_loss + entropy_loss + locality_loss
            total_graph_loss.backward()

            self.log('train_graph_loss', graph_loss.detach().cpu())
            self.log('train_sparsity_loss', sparsity_loss.detach().cpu())
            self.log('train_connectivity_loss',
                     connectivity_loss.detach().cpu())

            self.avg_accuracy = self.avg_accuracy * 0.95 + 0.05 * corr_pred.mean().item()

        optimizer.step()

        self.log('train_acc', 100 * correct_t)
        self.log('train_loss', loss.detach().cpu())

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

        self.log('val_loss', loss.detach())
        self.log('val_acc', 100 * correct_t)

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

        self.log('test_loss', loss.detach().cpu())
        self.log('test_acc', 100 * correct_t)
