import json
import os
import re
import codecs
import string
from collections import defaultdict
import numpy as np
from nltk.tokenize import wordpunct_tokenize
from sklearn import preprocessing
import torch
from gnns.utils.generic_utils import to_cuda, normalize_sparse_adj, sparse_mx_to_torch_sparse_tensor
import scipy.sparse as sp


def tokenize(s): return wordpunct_tokenize(
    re.sub('[%s]' % re.escape(string.punctuation), ' ', s))


def load_data(config):
    data_split = [
        float(x) for x in config['data_split_ratio'].replace(' ', '').split(',')]
    if config['dataset_name'] == 'mrd':
        file_path = os.path.join(config['data_dir'], 'mrd.txt')
        train_set, dev_set, test_set = load_mrd_data(
            file_path, data_split, config.get('data_seed', 1234))
    elif config['dataset_name'] == '20news':
        train_set, dev_set, test_set = load_20news_data(
            config['data_dir'], data_split, config.get('data_seed', 1234))
    else:
        raise ValueError('Unknown dataset_name: {}'.format(
            config['dataset_name']))

    return train_set, dev_set, test_set


def load_mrd_data(file_path, data_split, seed):
    '''Loads the Movie Review Data (https://www.cs.cornell.edu/people/pabo/movie-review-data/).'''

    all_instances = []
    all_seq_len = []
    with open(file_path, 'r') as fp:
        for line in fp:
            idx, rating, subj = line.split('\t')
            word_list = tokenize(subj.lower())
            all_instances.append([word_list, float(rating)])
            all_seq_len.append(len(word_list))

    print('[ Max seq length: {} ]'.format(np.max(all_seq_len)))
    print('[ Min seq length: {} ]'.format(np.min(all_seq_len)))
    print('[ Mean seq length: {} ]'.format(int(np.mean(all_seq_len))))

    # Random data split
    train_ratio, dev_ratio, test_ratio = data_split
    assert train_ratio + dev_ratio + test_ratio == 1
    n_train = int(len(all_instances) * train_ratio)
    n_dev = int(len(all_instances) * dev_ratio)
    n_test = len(all_instances) - n_train - n_dev

    random = np.random.RandomState(seed)
    random.shuffle(all_instances)

    train_instances = all_instances[:n_train]
    dev_instances = all_instances[n_train: n_train + n_dev]
    test_instances = all_instances[-n_test:]
    return train_instances, dev_instances, test_instances


def load_20news_data(data_dir, data_split, seed):
    train_dev_instances, train_dev_seq_len, train_dev_labels = data_load_helper(
        os.path.join(data_dir, "20news-bydate-train"))
    test_instances, test_seq_len, test_labels = data_load_helper(
        os.path.join(data_dir, "20news-bydate-test"))

    all_seq_len = train_dev_seq_len + test_seq_len
    print('[ Max seq length: {} ]'.format(np.max(all_seq_len)))
    print('[ Min seq length: {} ]'.format(np.min(all_seq_len)))
    print('[ Mean seq length: {} ]'.format(int(np.mean(all_seq_len))))

    le = preprocessing.LabelEncoder()
    le.fit(train_dev_labels + test_labels)
    nclass = len(list(le.classes_))
    print('[# of classes: {}] '.format(nclass))

    train_dev_labels = le.transform(train_dev_labels)
    test_labels = le.transform(test_labels)
    train_dev_instances = list(zip(train_dev_instances, train_dev_labels))
    test_instances = list(zip(test_instances, test_labels))

    # Random data split
    train_ratio, dev_ratio = data_split
    assert train_ratio + dev_ratio == 1
    n_train = int(len(train_dev_instances) * train_ratio)

    random = np.random.RandomState(seed)
    random.shuffle(train_dev_instances)

    train_instances = train_dev_instances[:n_train]
    dev_instances = train_dev_instances[n_train:]
    return train_instances, dev_instances, test_instances


def data_load_helper(data_dir):
    all_instances = []
    all_seq_len = []
    all_labels = []
    files = get_all_files(data_dir, recursive=True)
    for filename in files:
        # with open(filename, 'r') as fp:
        with codecs.open(filename, 'r', encoding='UTF-8', errors='ignore') as fp:
            text = fp.read().lower()
            word_list = tokenize(text)

            parent_name, child_name = os.path.split(filename)
            doc_name = os.path.split(parent_name)[-1] + '_' + child_name
            label = doc_name.split('_')[0]

            all_instances.append(word_list)
            all_seq_len.append(len(word_list))
            all_labels.append(label)

    return all_instances, all_seq_len, all_labels


def get_all_files(corpus_path, recursive=False):
    if recursive:
        return [os.path.join(root, file) for root, dirnames, filenames in os.walk(corpus_path) for file in filenames if os.path.isfile(os.path.join(root, file)) and not file.startswith('.')]
    else:
        return [os.path.join(corpus_path, filename) for filename in os.listdir(corpus_path) if os.path.isfile(os.path.join(corpus_path, filename)) and not filename.startswith('.')]


def prepare_datasets(config):
    """
    Prepares datasets based on the data type specified in config.

    Args:
        config: Configuration dictionary

    Returns:
        dict: Dictionary containing datasets
    """

    data_type = config.get('data_type', 'text')

    if data_type == 'text':
        # Load text data
        train_set, dev_set, test_set = load_data(config)

        return {
            'train': train_set,
            'dev': dev_set,
            'test': test_set
        }
    elif data_type in ('network', 'uci'):
        # Load network/graph data
        dataset_name = config['dataset_name']
        data_dir = config['data_dir']

        # Load features, labels, and adjacency matrix
        adj_path = os.path.join(data_dir, f'{dataset_name}_adj.npz')
        features_path = os.path.join(data_dir, f'{dataset_name}_features.npz')
        labels_path = os.path.join(data_dir, f'{dataset_name}_labels.npy')

        if os.path.exists(adj_path):
            adj = sp.load_npz(adj_path)
        else:
            # Create empty adjacency matrix if not found
            adj = sp.csr_matrix((0, 0))

        if os.path.exists(features_path):
            features = sp.load_npz(features_path)
            if sp.issparse(features):
                features = features.toarray()
        else:
            features = np.array([])

        if os.path.exists(labels_path):
            labels = np.load(labels_path)
        else:
            labels = np.array([])

        # Load train/val/test splits
        idx_train = np.load(os.path.join(
            data_dir, f'{dataset_name}_idx_train.npy'))
        idx_val = np.load(os.path.join(
            data_dir, f'{dataset_name}_idx_val.npy'))
        idx_test = np.load(os.path.join(
            data_dir, f'{dataset_name}_idx_test.npy'))

        # Convert to torch tensors
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)

        # Normalize adjacency matrix
        if config.get('normalize_adj', False):
            adj = normalize_sparse_adj(adj)

        adj = sparse_mx_to_torch_sparse_tensor(adj).to_dense()

        # Move to device if specified
        device = config.get('device', None)
        if device:
            features = to_cuda(features, device)
            labels = to_cuda(labels, device)
            adj = to_cuda(adj, device)

        return {
            'adj': adj,
            'features': features,
            'labels': labels,
            'idx_train': torch.LongTensor(idx_train),
            'idx_val': torch.LongTensor(idx_val),
            'idx_test': torch.LongTensor(idx_test)
        }
    else:
        raise ValueError(f'Unknown data_type: {data_type}')


class DataStream(object):
    """Data stream for batching text data"""

    def __init__(self, data, vocab, config, isShuffle=False, isLoop=False, isSort=True, batch_size=None):
        self.data = data
        self.vocab = vocab
        self.config = config
        self.isShuffle = isShuffle
        self.isLoop = isLoop
        self.isSort = isSort
        self.batch_size = batch_size if batch_size is not None else config.get(
            'batch_size', 32)
        self.num_examples = len(data)
        self.cur_pointer = 0

        # Sort by sequence length for efficient batching
        if isSort:
            self.data = sorted(
                self.data, key=lambda x: len(x[0]), reverse=True)

        # Shuffle if needed
        if isShuffle:
            self._shuffle()

    def _shuffle(self):
        np.random.shuffle(self.data)

    def nextBatch(self):
        """Get next batch of data"""
        if self.cur_pointer >= self.num_examples:
            if not self.isLoop:
                return None
            self.cur_pointer = 0
            if self.isShuffle:
                self._shuffle()

        end_pointer = min(self.cur_pointer +
                          self.batch_size, self.num_examples)
        batch = self.data[self.cur_pointer:end_pointer]
        self.cur_pointer = end_pointer

        return batch

    def get_num_batch(self):
        """Get number of batches"""
        return (self.num_examples + self.batch_size - 1) // self.batch_size

    def reset(self):
        """Reset the data stream"""
        self.cur_pointer = 0
        if self.isShuffle:
            self._shuffle()


def vectorize_input(batch, config, training=True, device=None, vocab=None):
    """
    Vectorize a batch of text data into tensors.

    Args:
        batch: List of examples
        config: Configuration dictionary
        training: Whether in training mode
        device: Device to place tensors on
        vocab: Vocabulary object for converting words to indices

    Returns:
        dict: Dictionary containing vectorized batch
    """
    import torch
    from gnns.utils.generic_utils import to_cuda, create_mask

    if not batch:
        return None

    batch_size = len(batch)

    # Extract sequences and labels
    sequences = [item[0] for item in batch]
    labels = [item[1] if len(item) > 1 else 0 for item in batch]

    # Get sequence lengths
    seq_lens = [len(seq) for seq in sequences]
    max_len = max(seq_lens)

    # Pad sequences and convert words to indices if needed
    padded_seqs = []
    for seq in sequences:
        # Check if seq contains strings (words) or integers (indices)
        if len(seq) > 0 and isinstance(seq[0], str):
            # Convert words to indices using vocab
            if vocab is None:
                raise ValueError(
                    "vocab must be provided when sequences contain words (strings)")
            indexed_seq = [vocab.getIndex(word) for word in seq]
        else:
            # Already indexed
            indexed_seq = seq

        # Pad sequence
        padded_seq = indexed_seq + [0] * (max_len - len(indexed_seq))
        padded_seqs.append(padded_seq)

    # Convert to tensors
    context = torch.LongTensor(padded_seqs)
    context_lens = torch.LongTensor(seq_lens)

    # Handle different task types
    task_type = config.get('task_type', 'classification')
    if task_type == 'classification':
        targets = torch.LongTensor(labels)
    elif task_type == 'regression':
        targets = torch.FloatTensor(labels)
    else:
        targets = torch.LongTensor(labels)

    # Create mask
    context_mask = create_mask(context_lens, max_len, device=device)

    # Move to device
    if device:
        context = to_cuda(context, device)
        context_lens = to_cuda(context_lens, device)
        targets = to_cuda(targets, device)
        context_mask = to_cuda(context_mask, device)

    return {
        'context': context,
        'context_lens': context_lens,
        'targets': targets,
        'context_mask': context_mask,
        'batch_size': batch_size
    }
