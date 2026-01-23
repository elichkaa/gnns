from gnns.idgl.model_handler import ModelHandler
import yaml
import argparse
from pathlib import Path
import sys
import json
import matplotlib.pyplot as plt
import logging
import torch

torch.backends.cudnn.enabled = False


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_results(config, best_metrics, test_metrics, out_dir):
    serializable_config = {}
    for key, value in config.items():
        if hasattr(value, '__class__') and value.__class__.__module__ == 'torch':
            serializable_config[key] = str(value)
        else:
            try:
                json.dumps(value)
                serializable_config[key] = value
            except (TypeError, ValueError):
                serializable_config[key] = str(value)

    results = {
        'config': serializable_config,
        'best_validation_metrics': best_metrics,
        'test_metrics': test_metrics if test_metrics else {}
    }

    results_file = out_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_file}")


def setup_logging(out_dir):
    """Setup additional logging to results directory"""
    log_dir = out_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / 'training.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train IDGL model')
    parser.add_argument(
        '--config',
        type=str,
        default='../idgl/config/idgl.yml',
        help='Path to config file'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default=None,
        help='Override dataset name from config'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Override data directory from config'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default=None,
        help='Override output directory from config'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Override random seed from config'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=None,
        help='Override max epochs from config'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Override batch size from config'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='Override learning rate from config'
    )
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=None,
        help='Override hidden size from config'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=None,
        help='Override dropout from config'
    )
    parser.add_argument(
        '--cuda_id',
        type=int,
        default=None,
        help='Override CUDA device ID from config'
    )
    parser.add_argument(
        '--no_cuda',
        action='store_true',
        help='Disable CUDA'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    if args.dataset_name is not None:
        config['dataset_name'] = args.dataset_name
    if args.data_dir is not None:
        config['data_dir'] = args.data_dir
    if args.out_dir is not None:
        config['out_dir'] = args.out_dir
    if args.seed is not None:
        config['seed'] = args.seed
    if args.max_epochs is not None:
        config['max_epochs'] = args.max_epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if args.hidden_size is not None:
        config['hidden_size'] = args.hidden_size
    if args.dropout is not None:
        config['dropout'] = args.dropout
    if args.cuda_id is not None:
        config['cuda_id'] = args.cuda_id
    if args.no_cuda:
        config['no_cuda'] = True

    out_dir = Path(config['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(out_dir)

    config_items = [
        ("Dataset", config['dataset_name']),
        ("Data directory", config['data_dir']),
        ("Output directory", config['out_dir']),
        ("Model", config['model_name']),
        ("Hidden size", config['hidden_size']),
        ("Dropout", config['dropout']),
        ("Graph learning", config['graph_learn']),
        ("Graph metric", config['graph_metric_type']),
        ("Max epochs", config['max_epochs']),
        ("Batch size", config['batch_size']),
        ("Learning rate", config['learning_rate']),
        ("Optimizer", config['optimizer']),
        ("Seed", config['seed']),
        ("CUDA", not config['no_cuda']),
    ]

    for key, value in config_items:
        print(f"{key}: {value}")
        logger.info(f"{key}: {value}")

    if not config['no_cuda']:
        print(f"CUDA device: {config['cuda_id']}")
        logger.info(f"CUDA device: {config['cuda_id']}")

    model_handler = ModelHandler(config)
    best_metrics = model_handler.train()

    logger.info(f"Best validation metrics: {best_metrics}")

    test_metrics = None
    if model_handler.test_loader is not None:
        test_metrics = model_handler.test()

        for metric_name, metric_value in test_metrics.items():
            msg = f"{metric_name.upper()}: {metric_value:.4f}"
            print(msg)
            logger.info(msg)

    save_results(config, best_metrics, test_metrics, out_dir)
    logger.info(f"Results saved to {out_dir / 'results.json'}")


if __name__ == '__main__':
    main()
