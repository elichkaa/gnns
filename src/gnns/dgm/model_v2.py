import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
from argparse import Namespace
from gnns.dgm.dgm_layers import MLP, cDGM, dDGM, Identity

# Encoder -> DGM Layers -> GNN Layers -> Classifier
class DGM_Model(pl.LightningModule):

    def __init__(self, hparams):
        super(DGM_Model, self).__init__()

        if not isinstance(hparams, Namespace):
            hparams = Namespace(**hparams)

        self.save_hyperparameters(hparams)

        dgm_layers = hparams.dgm_layers
        conv_layers = hparams.conv_layers
        fc_layers = hparams.fc_layers
        k = hparams.k
        self.task = getattr(hparams, 'task', 'classification')

        if hasattr(hparams, 'pre_fc') and hparams.pre_fc is not None and len(hparams.pre_fc) > 0:
            self.pre_fc = MLP(hparams.pre_fc, final_activation=True)
        else:
            self.pre_fc = None

        self.graph_f = nn.ModuleList()
        self.node_g = nn.ModuleList()

        for i, (dgm_l, conv_l) in enumerate(zip(dgm_layers, conv_layers)):
            if len(dgm_l) > 0:
                # f_θ
                if not hasattr(hparams, 'ffun') or hparams.ffun == 'gcn':
                    embed_f = GCNConv(dgm_l[0], dgm_l[-1])
                elif hparams.ffun == 'gat':
                    embed_f = GATConv(dgm_l[0], dgm_l[-1])
                elif hparams.ffun == 'mlp':
                    embed_f = MLP(dgm_l)
                elif hparams.ffun == 'knn':
                    embed_f = Identity(retparam=0)
                else:
                    embed_f = MLP(dgm_l)

                # DGM module
                if hparams.dgm_type == 'continuous' or hparams.dgm_type == 'cDGM':
                    self.graph_f.append(
                        cDGM(embed_f, distance=hparams.distance))
                else:
                    self.graph_f.append(
                        dDGM(embed_f, k=k, distance=hparams.distance))
            else:
                self.graph_f.append(Identity())

            # GNN layer
            if hparams.gfun == 'edgeconv':
                conv_l_copy = conv_l.copy()
                conv_l_copy[0] = conv_l_copy[0] * 2
                self.node_g.append(EdgeConv(MLP(conv_l_copy), hparams.pooling))
            elif hparams.gfun == 'gcn':
                self.node_g.append(GCNConv(conv_l[0], conv_l[1]))
            elif hparams.gfun == 'gat':
                self.node_g.append(GATConv(conv_l[0], conv_l[1]))
            else:
                self.node_g.append(GCNConv(conv_l[0], conv_l[1]))

        self.fc = MLP(fc_layers, final_activation=False)

        self.avg_accuracy = None
        self.automatic_optimization = False

    def forward(self, X, edge_index=None, batch_size=None, return_adj=False):
        if self.pre_fc is not None:
            X = self.pre_fc(X)

        if X.dim() == 2:
            X = X.unsqueeze(0)

        batch_sz, num_nodes, feature_dim = X.shape

        graph_x = X.detach()
        logprobs_list = []

        for dgm, gnn in zip(self.graph_f, self.node_g):
            graph_x, A_or_edge_index, logprobs = dgm(graph_x, edge_index)

            if self.hparams.dgm_type == "continuous":
                A = A_or_edge_index
        
                if A is not None:
                    if A.dim() == 2:
                        A = A.unsqueeze(0)
        
                    edge_index_list = []
                    for b in range(A.size(0)):
                        src, tgt = torch.nonzero(A[b] > 0.5, as_tuple=True)
                        ei = torch.stack([src, tgt], dim=0)
                        ei += b * num_nodes
                        edge_index_list.append(ei)
        
                    edge_index = torch.cat(edge_index_list, dim=1)
                else:
                    edge_index = None
            else:
                edge_index = A_or_edge_index

            X_flat = X.view(-1, X.shape[-1])
            X_out = F.relu(gnn(
                F.dropout(X_flat, p=self.hparams.dropout,
                          training=self.training),
                edge_index
            ))
            X = X_out.view(batch_sz, num_nodes, -1)

            graph_x = torch.cat([graph_x, X.detach()], dim=-1)

            if logprobs is not None:
                logprobs_list.append(logprobs)

        logits = self.fc(X)

        logprobs_stacked = torch.stack(
            logprobs_list, dim=-1) if len(logprobs_list) > 0 else None

        if return_adj:
            if self.hparams.dgm_type == 'continuous':
                _, A, _ = self.graph_f[0](X.detach(), edge_index)
                return logits, A
            else:
                adj = torch.zeros(batch_sz, num_nodes, num_nodes, device=X.device)
                if edge_index is not None:
                    src = edge_index[0] % num_nodes
                    tgt = edge_index[1] % num_nodes
                    adj[:, src, tgt] = 1.0
                    adj[:, tgt, src] = 1.0
                return logits, adj

        return logits, logprobs_list

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        X = batch['node_features']  # [batch, max_len, feature_dim]
        labels = batch['label']  # [batch]
        attention_mask = batch['attention_mask']  # [batch, max_len]

        batch_size = X.shape[0]
        num_classes = self.hparams.fc_layers[-1]

        logits, logprobs = self(X, edge_index=None)
        mask_expanded = attention_mask.unsqueeze(-1)  # [batch, max_len, 1]

        logits_masked = logits * mask_expanded
        pooled_logits = logits_masked.sum(
            dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        if self.task == 'classification':
            y_one_hot = F.one_hot(labels, num_classes=num_classes).float()
            cls_loss = F.cross_entropy(pooled_logits, labels)
            predictions = pooled_logits.argmax(dim=-1)
            self.manual_backward(cls_loss)
            correct = (predictions == labels).float()
            train_acc = correct.mean().item()
            self.log('train_loss', cls_loss.detach())
            self.log('train_acc', train_acc * 100)
        else:  # regression
            preds = pooled_logits.squeeze(-1)
            cls_loss = F.mse_loss(preds, labels)
            self.manual_backward(cls_loss)
            mae = torch.abs(preds - labels)
            correct = 1.0 - mae
            self.log('train_loss', cls_loss.detach())
            self.log('train_mae', mae.mean())

        # backward cls loss
        
        # predictions = pooled_logits.argmax(dim=-1)
        # correct = (predictions == labels).float()
        # train_acc = correct.mean().item()

        # NOTE: only for ddgm
        if logprobs is not None and len(logprobs) > 0 and self.hparams.dgm_type != 'continuous':
            logprobs_tensor = torch.stack(logprobs, dim=-1)  # [batch, num_nodes, k, num_layers]
            
            if self.avg_accuracy is None or self.avg_accuracy.size(0) != batch_size:
                self.avg_accuracy = torch.ones(batch_size, device=correct.device) * 0.5
            
            reward = self.avg_accuracy.to(correct.device) - correct
            
            graph_loss = (reward.view(batch_size, 1, 1, 1) *
                          logprobs_tensor.mean(dim=[1, 2])).mean()
            
            # self.manual_backward(graph_loss)
            
            self.avg_accuracy = self.avg_accuracy.to(
                correct.device) * 0.95 + 0.05 * correct
            
            self.log('train_graph_loss', graph_loss.detach())

        optimizer.step()

        # self.log('train_loss', cls_loss.detach())
        # self.log('train_acc', train_acc * 100)

        return cls_loss

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, 'test')

    def _eval_step(self, batch, batch_idx, prefix):
        X = batch['node_features']
        labels = batch['label']
        attention_mask = batch['attention_mask']
        num_classes = self.hparams.fc_layers[-1]
        
        num_eval = 8 if self.hparams.dgm_type == 'discrete' else 1
        
        pooled_sum = None
        for _ in range(num_eval):
            logits, _ = self(X, edge_index=None)
            
            mask_expanded = attention_mask.unsqueeze(-1)
            logits_masked = logits * mask_expanded
            pooled_logits = logits_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            
            if pooled_sum is None:
                pooled_sum = pooled_logits
            else:
                pooled_sum += pooled_logits
        
        pooled_logits = pooled_sum / num_eval
        
        if self.task == 'classification':
            loss = F.cross_entropy(pooled_logits, labels)
            predictions = pooled_logits.argmax(dim=-1)
            accuracy = (predictions == labels).float().mean().item()
            self.log(f'{prefix}_loss', loss)
            self.log(f'{prefix}_acc', accuracy * 100)
        else:  # regression
            preds = pooled_logits.squeeze(-1)
            loss = F.mse_loss(preds, labels)
            mae = torch.abs(preds - labels).mean()
            ss_res = ((labels - preds) ** 2).sum()
            ss_tot = ((labels - labels.mean()) ** 2).sum()
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            self.log(f'{prefix}_loss', loss)
            self.log(f'{prefix}_mae', mae)
            self.log(f'{prefix}_r2', r2)
        
        # self.log(f'{prefix}_loss', loss)
        # self.log(f'{prefix}_acc', accuracy * 100)
        
        return loss

    def configure_optimizers(self):
        """Configure Adam optimizer."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr
        )
        return optimizer
