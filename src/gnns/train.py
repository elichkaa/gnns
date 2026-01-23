from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from argparse import ArgumentParser
import pytorch_lightning as pl
from gnns.bert_classifier import SimpleBERTClassifier
from gnns.datasets import NewsGroupsGraphDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import sys
import torch
# sys.path.insert(0, './keops')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["USE_KEOPS"] = "True"


def collate_fn(batch):
    # NOTE: for batching
    node_features = torch.stack([b['node_features'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    labels = torch.stack([b['label'] for b in batch])
    texts = [b['text'] for b in batch]

    return {
        'node_features': node_features,
        'attention_mask': attention_mask,
        'label': labels,
        'text': texts
    }


def get_encoder_dim(encoder_name, use_rnn, rnn_hidden_size=None, rnn_bidirectional=False):
    """Get the output dimension of the encoder."""
    if use_rnn:
        if rnn_hidden_size is None:
            raise ValueError(
                "Must specify rnn_hidden_size when use_rnn_encoder=True")
        return rnn_hidden_size * (2 if rnn_bidirectional else 1)

    dim_map = {
        'distilbert-base-uncased': 768,
        'bert-base-uncased': 768,
        'bert-large-uncased': 1024,
        'roberta-base': 768,
        'roberta-large': 1024,
    }
    return dim_map.get(encoder_name, 768)


def run_training_process(run_params):
    encoder_dim = get_encoder_dim(
        run_params.encoder_name,
        run_params.use_rnn_encoder,
        getattr(run_params, 'rnn_hidden_size', None),
        getattr(run_params, 'rnn_bidirectional', False)
    )

    # Setup RNN encoder if needed
    rnn_encoder = None
    word_embeddings = None
    if run_params.use_rnn_encoder:

        from gnns.idgl.layers.common import EncoderRNN

        vocab_size = 30522  # BERT vocab size for tokenizer compatibility
        emb_dim = run_params.rnn_embedding_dim

        word_embeddings = nn.Embedding(vocab_size, emb_dim)
        rnn_encoder = EncoderRNN(
            input_size=emb_dim,
            hidden_size=run_params.rnn_hidden_size,
            bidirectional=run_params.rnn_bidirectional,
            num_layers=run_params.rnn_num_layers,
            rnn_type=run_params.rnn_type,
            device='cuda'
        )

    dataset_kwargs = {
        'max_length': run_params.max_length,
        'device': 'cuda',
        'use_cache': True,
        'val_split': run_params.val_split,
        'freeze_encoder': run_params.freeze_encoder,
        'remove_punctualization': run_params.remove_punctualization,
    }

    if run_params.use_rnn_encoder:
        dataset_kwargs.update({
            'use_rnn_encoder': True,
            'rnn_encoder': rnn_encoder,
            'word_embeddings': word_embeddings,
        })
    else:
        dataset_kwargs.update({
            'encoder_name': run_params.encoder_name,
        })

    train_data = NewsGroupsGraphDataset(split='train', **dataset_kwargs)
    val_data = NewsGroupsGraphDataset(split='val', **dataset_kwargs)
    test_data = NewsGroupsGraphDataset(split='test', **dataset_kwargs)

    train_loader = DataLoader(train_data, batch_size=run_params.batch_size,
                              collate_fn=collate_fn, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=run_params.batch_size,
                            collate_fn=collate_fn, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=run_params.batch_size,
                             collate_fn=collate_fn, shuffle=False, num_workers=4)

    class MyDataModule(pl.LightningDataModule):
        def setup(self, stage=None):
            pass

        def train_dataloader(self):
            return train_loader

        def val_dataloader(self):
            return val_loader

        def test_dataloader(self):
            return test_loader

    # MODEL
    if run_params.model_type == 'baseline':
        args.input_dim = encoder_dim  # Set input dimension
        args.fc_layers = [64, 32, train_data.num_classes]
        model = SimpleBERTClassifier(args)
        model_name = f"baseline_{args.encoder_name}_pool{args.pooling}"
    # elif args.model_type == 'dgm':
    #     if args.pre_fc is None or len(args.pre_fc) == 0:
    #         if len(args.dgm_layers[0]) > 0:
    #             args.dgm_layers[0][0] = encoder_dim
    #         args.conv_layers[0][0] = encoder_dim
    #     else:
    #         args.pre_fc[0] = encoder_dim
    #         first_hidden = args.pre_fc[-1]
    #         if len(args.dgm_layers[0]) > 0:
    #             args.dgm_layers[0][0] = first_hidden
    #         args.conv_layers[0][0] = first_hidden

    #     args.fc_layers[-1] = train_data.num_classes

    #     model = DGM_Model(args)
    #     dgm_type = "cDGM" if args.use_continuous_dgm else "dDGM"
    #     encoder_str = "rnn" if args.use_rnn_encoder else args.encoder_name.split(
    #         '/')[-1]
    #     model_name = f"{dgm_type}_{encoder_str}_k{args.k}_{args.gfun}_{args.distance}_pool{args.pooling}"
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        min_delta=0.001,
        patience=args.patience,
        verbose=True,
        mode='max'
    )
    callbacks = [early_stop_callback]

    if val_data == test_data:
        callbacks = None

    logger = TensorBoardLogger("../logs/", name=model_name)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        deterministic=False,
        enable_progress_bar=True
    )

    trainer.fit(model, datamodule=MyDataModule())
    trainer.test(model, datamodule=MyDataModule())


if __name__ == "__main__":
    parser = ArgumentParser()

    # Model type
    parser.add_argument("--model_type", type=str, default='dgm',
                        choices=['baseline', 'dgm'],
                        help="Model type: 'baseline' (BERT only) or 'dgm' (with graph learning)")

    # Encoder configuration
    parser.add_argument("--encoder_name", type=str, default='distilbert-base-uncased',
                        help="Transformer encoder name (e.g., 'bert-base-uncased', 'roberta-base')")
    parser.add_argument("--use_rnn_encoder", action='store_true',
                        help="Use RNN encoder instead of transformer")
    parser.add_argument("--rnn_embedding_dim", type=int, default=300,
                        help="RNN word embedding dimension")
    parser.add_argument("--rnn_hidden_size", type=int, default=384,
                        help="RNN hidden size (will be doubled if bidirectional)")
    parser.add_argument("--rnn_bidirectional", action='store_true',
                        help="Use bidirectional RNN")
    parser.add_argument("--rnn_num_layers", type=int, default=2,
                        help="Number of RNN layers")
    parser.add_argument("--rnn_type", type=str, default='lstm',
                        choices=['lstm', 'gru'],
                        help="RNN type")

    # Data configuration
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max sequence length for documents")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of train data for validation")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--freeze_encoder", action='store_true',
                        help="Freeze encoder weights")
    parser.add_argument("--remove_punctualization", action='store_true',
                        help="Remove punctuation from text")

    # DGM architecture
    parser.add_argument("--conv_layers", type=lambda x: eval(x),
                        default=[[768, 256], [256, 128], [128, 64]],
                        help="GNN layer dimensions")
    parser.add_argument("--dgm_layers", type=lambda x: eval(x),
                        default=[[768, 256, 64], [], []],
                        help="DGM layer dimensions")
    parser.add_argument("--fc_layers", type=lambda x: eval(x),
                        default=[64, 32, 20],
                        help="Classifier dimensions")
    parser.add_argument("--pre_fc", type=lambda x: eval(x),
                        default=[],
                        help="Pre-processing MLP ([] to disable)")

    # Graph learning
    parser.add_argument("--k", type=int, default=5,
                        help="Number of neighbors per node")
    parser.add_argument("--distance", type=str, default='euclidean',
                        choices=['euclidean', 'hyperbolic'],
                        help="Distance metric")
    parser.add_argument("--use_continuous_dgm", action='store_true',
                        help="Use continuous DGM (cDGM) instead of discrete (dDGM)")
    parser.add_argument("--gfun", type=str, default='gat',
                        choices=['gcn', 'gat', 'edgeconv'],
                        help="GNN architecture")
    parser.add_argument("--ffun", type=str, default='gcn',
                        choices=['gcn', 'gat', 'mlp', 'knn'],
                        help="DGM encoder function")

    # Regularization
    parser.add_argument("--lambda_sparse", type=float, default=0.1,
                        help="Sparsity regularization weight")
    parser.add_argument("--lambda_connect", type=float, default=0.01,
                        help="Connectivity regularization weight")
    parser.add_argument("--lambda_entropy", type=float, default=0.01,
                        help="Edge entropy regularization weight")

    # Training
    parser.add_argument("--pooling", type=str, default='mean',
                        choices=['mean', 'max', 'cls', 'sum'],
                        help="Pooling method")
    parser.add_argument("--dropout", type=float, default=0.4,
                        help="Dropout rate")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=100,
                        help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--test_eval", type=int, default=5,
                        help="Number of forward passes for test ensemble")
    parser.add_argument("--weight_decay", type=float, default=0.01)

    args = parser.parse_args()

    run_training_process(args)
