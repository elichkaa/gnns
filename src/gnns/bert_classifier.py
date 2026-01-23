import torch
from torch import nn
import pytorch_lightning as pl
from argparse import Namespace


class SimpleBERTClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super(SimpleBERTClassifier, self).__init__()
        if type(hparams) is not Namespace:
            hparams = Namespace(**hparams)

        self.save_hyperparameters(hparams)

        input_dim = getattr(hparams, 'input_dim', 768)

        # classifier
        fc_dims = [input_dim] + hparams.fc_layers
        fc_layers = []
        for i in range(len(fc_dims) - 1):
            fc_layers.append(nn.Linear(fc_dims[i], fc_dims[i+1]))
            if i < len(fc_dims) - 2:
                fc_layers.append(nn.LeakyReLU(0.1))
                if hparams.dropout > 0:
                    fc_layers.append(nn.Dropout(hparams.dropout))
        self.classifier = nn.Sequential(*fc_layers)

        self.pooling = hparams.pooling if hasattr(
            hparams, 'pooling') else 'mean'

    def global_pooling(self, x, attention_mask):
        if self.pooling == 'mean':
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (x * mask_expanded).sum(dim=1)
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

    def forward(self, x, attention_mask):
        """
        Args:
            x: [batch, seq_len, input_dim] - token embeddings
            attention_mask: [batch, seq_len]

        Returns:
            logits: [batch_size, num_classes]
        """
        x_pooled = self.global_pooling(x, attention_mask)
        logits = self.classifier(x_pooled)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr,
            # removed because very bad results with it
            weight_decay=self.hparams.weight_decay
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        X = batch['node_features']
        labels = batch['label']
        attention_mask = batch['attention_mask']

        logits = self(X, attention_mask)
        loss = nn.functional.cross_entropy(logits, labels)

        acc = (logits.argmax(-1) == labels).float().mean()

        self.log('train_loss', loss)
        self.log('train_acc', 100 * acc)

        return loss

    def validation_step(self, batch, batch_idx):
        X = batch['node_features']
        labels = batch['label']
        attention_mask = batch['attention_mask']

        logits = self(X, attention_mask)
        loss = nn.functional.cross_entropy(logits, labels)

        acc = (logits.argmax(-1) == labels).float().mean()

        self.log('val_loss', loss)
        self.log('val_acc', 100 * acc)

        return loss

    def test_step(self, batch, batch_idx):
        X = batch['node_features']
        labels = batch['label']
        attention_mask = batch['attention_mask']

        logits = self(X, attention_mask)
        loss = nn.functional.cross_entropy(logits, labels)

        acc = (logits.argmax(-1) == labels).float().mean()

        self.log('test_loss', loss)
        self.log('test_acc', 100 * acc)

        return loss
