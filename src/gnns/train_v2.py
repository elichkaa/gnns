from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from argparse import ArgumentParser
import pytorch_lightning as pl
from gnns.bert_classifier import SimpleBERTClassifier
from gnns.datasets import MRDGraphDataset, NewsGroupsGraphDataset
from torch.utils.data import DataLoader
import torch
from gnns.dgm.model_v2 import DGM_Model

TEST_ONLY: bool = False


def collate_fn(batch):
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


def get_encoder_dim(encoder_name):
    dim_map = {
        'distilbert-base-uncased': 768,
        'google/embeddinggemma-300m': 768
    }
    return dim_map.get(encoder_name, 768)


def run_training_process(run_params):
    encoder_dim = get_encoder_dim(run_params.encoder_name)

    dataset_kwargs = {
        'encoder_name': run_params.encoder_name,
        'max_length': run_params.max_length,
        'device': 'cuda',
        'use_cache': False,
        'val_split': run_params.val_split,
        'freeze_encoder': run_params.freeze_encoder,
        'remove_punctualization': run_params.remove_punctualization,
    }

    if run_params.dataset == "20news":
        train_data = NewsGroupsGraphDataset(split='train', **dataset_kwargs)
        val_data = NewsGroupsGraphDataset(split='val', **dataset_kwargs)
        test_data = NewsGroupsGraphDataset(split='test', **dataset_kwargs)
    else:
        dataset_kwargs['max_length'] = 512
        train_data = MRDGraphDataset(split='train', **dataset_kwargs)
        val_data = MRDGraphDataset(split='val', **dataset_kwargs)
        test_data = MRDGraphDataset(split='test', **dataset_kwargs)

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
        args.input_dim = encoder_dim
        # final_val = 1 if run_params.task == "regression" else train_data.num_classes
        # args.fc_layers = [final_val]
        args.task = run_params.task
        model = SimpleBERTClassifier(args)
        model_name = f"baseline_{args.encoder_name}_pool{args.pooling}"
    elif args.model_type in ['dgm', 'cdgm']:
        if args.use_continuous_dgm or args.model_type == 'cdgm':
            args.dgm_type = 'continuous'
            dgm_type = "cDGM"
        else:
            args.dgm_type = 'discrete'
            dgm_type = "dDGM"
        if args.pre_fc is None or len(args.pre_fc) == 0:
            if len(args.dgm_layers[0]) > 0:
                args.dgm_layers[0] = [encoder_dim] + args.dgm_layers[0][1:]
            args.conv_layers[0] = [encoder_dim] + args.conv_layers[0][1:]
        else:
            args.pre_fc[0] = encoder_dim
            first_hidden = args.pre_fc[-1]
            if len(args.dgm_layers[0]) > 0:
                args.dgm_layers[0][0] = first_hidden
            args.conv_layers[0][0] = first_hidden
        
        args.fc_layers[-1] = 1 if run_params.task == "regression" else train_data.num_classes
        args.task = run_params.task
        
        model = DGM_Model(args)
        encoder_str = args.encoder_name.replace("/", "-")
        model_name = f"{dgm_type}_{encoder_str}_k{args.k}_{args.gfun}_{args.distance}_pool{args.pooling}"
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    if run_params.task == 'regression':
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=args.patience,
            verbose=True,
            mode='min'
        )
    else:
        early_stop_callback = EarlyStopping(
            monitor='val_acc',
            min_delta=0.001,
            patience=args.patience,
            verbose=True,
            mode='max'
        )
    callbacks = [early_stop_callback, checkpoint_callback]

    if val_data == test_data:
        callbacks = None
    print("LOGGING")
    logger = TensorBoardLogger(f"./logs_{run_params.dataset}/", name=model_name.replace("/", "-"))
    if TEST_ONLY:
        prefix_dir = "./logs/dDGM_google-embeddinggemma-300m_k5_gat_euclidean_poolmean/"
        version = "version_50"
        checkpoint_dir = "epoch=99-step=56600.ckpt"
        model = DGM_Model.load_from_checkpoint(
            f'{prefix_dir}{version}/checkpoints/{checkpoint_dir}')

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        deterministic=False,
        enable_progress_bar=True,
    )

    if not TEST_ONLY:
        if args.resume_from_checkpoint:
            trainer.fit(model, datamodule=MyDataModule(),
                        ckpt_path=args.resume_from_checkpoint)
        else:
            trainer.fit(model, datamodule=MyDataModule())

    trainer.test(model, datamodule=MyDataModule())


if __name__ == "__main__":
    parser = ArgumentParser()

    # Model type
    parser.add_argument("--model_type", type=str, default='dgm',
                        choices=['baseline', 'dgm'],
                        help="Model type: 'baseline' (BERT only) or 'dgm' (with graph learning)")

    # Encoder configuration
    parser.add_argument("--encoder_name", type=str, default='google/embeddinggemma-300m',
                        help="Transformer encoder name (e.g., 'bert-base-uncased', 'roberta-base')")
    parser.add_argument("--dataset", type=str, default='20news',
                        choices=['20news', 'mrd'],
                        help="Dataset name")
    parser.add_argument("--task", type=str, default='classification',
                        choices=['classification', 'regression'],
                        help="Task Type")

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
                        default=[[64, 128]],
                        help="GNN layer dimensions")
    parser.add_argument("--dgm_layers", type=lambda x: eval(x),
                        default=[[256, 128, 64]],
                        help="DGM layer dimensions")
    parser.add_argument("--fc_layers", type=lambda x: eval(x),
                        default=[128, 64, 20],
                        help="Classifier dimensions")
    parser.add_argument("--pre_fc", type=lambda x: eval(x),
                        default=[768, 256],
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
    parser.add_argument("--ffun", type=str, default='mlp',
                        choices=['gcn', 'gat', 'mlp', 'knn'],
                        help="DGM encoder function")

    # Regularization
    parser.add_argument("--lambda_sparse", type=float, default=0.01,
                        help="Sparsity regularization weight")
    parser.add_argument("--lambda_connect", type=float, default=0.001,
                        help="Connectivity regularization weight")
    parser.add_argument("--lambda_entropy", type=float, default=0.001,
                        help="Edge entropy regularization weight")
    parser.add_argument("--lambda_locality", type=float, default=0.01,
                        help="Locality regularization weight")
    parser.add_argument("--graph_loss_weight", type=float, default=0.01)

    # Training
    parser.add_argument("--pooling", type=str, default='mean',
                        choices=['mean', 'max', 'cls', 'sum'],
                        help="Pooling method")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=100,
                        help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--test_eval", type=int, default=10,
                        help="Number of forward passes for test ensemble")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--resume_from_checkpoint", type=str,
                        default=None, help="Path to checkpoint to resume training from")

    args = parser.parse_args()

    run_training_process(args)
