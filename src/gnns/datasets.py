import os
import pickle
import string
import re
from typing import Literal, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.conftest import fetch_20newsgroups
from sklearn.model_selection import train_test_split

class NewsGroupsGraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: Literal['train', 'val', 'test'] = 'train',
        max_length: int = 128,
        cache_dir: str = '../data/newsgroups_cache',
        device: str = 'cpu',
        categories: Optional[list] = None,
        remove: tuple = ('headers', 'footers', 'quotes'),
        use_cache: bool = True,
        val_split: float = 0.2,
        freeze_encoder: bool = True,
        remove_punctualization: bool = False,
        encoder_name: str = 'distilbert-base-uncased',
        encoder_class=None
    ):
        self.split = split
        self.max_length = max_length
        self.device = device
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.val_split = val_split
        self.remove_punctualization = remove_punctualization
        self.encoder_name = encoder_name

        os.makedirs(cache_dir, exist_ok=True)

        if encoder_class is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
            self.encoder = encoder_class.from_pretrained(encoder_name)
        else:

            self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
            self.encoder = AutoModel.from_pretrained(encoder_name)

        self.encoder.eval()
        self.encoder.to(device)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        data_split = 'train' if split in ['train', 'val'] else 'test'
        print(f"Loading 20newsgroups {data_split} data...")
        self.data = fetch_20newsgroups(
            subset=data_split, categories=categories, shuffle=True, random_state=42, remove=remove)

        self.X = self.data.data
        self.y = self.data.target
        self.num_classes = len(self.data.target_names)
        self.target_names = self.data.target_names

        if split in ['train', 'val']:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                self.X, self.y, test_size=val_split, random_state=42, stratify=self.y)

            if split == 'train':
                self.texts = train_texts
                self.labels = torch.LongTensor(train_labels)
            else:
                self.texts = val_texts
                self.labels = torch.LongTensor(val_labels)
        else:
            self.texts = self.X
            self.labels = torch.LongTensor(self.y)

        encoder_key = self.encoder_name.replace("/", "_")
        cache_file = os.path.join(
            cache_dir, f'newsgroups_{split}_maxlen{max_length}_valsplit{val_split}_{encoder_key}.pkl')

        if use_cache and os.path.exists(cache_file):
            print(f"Loading cached embeddings from {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.embeddings = cache_data['embeddings']
                self.attention_masks = cache_data['attention_masks']
                self.num_nodes_list = cache_data['num_nodes_list']
        else:
            self._preprocess_all_documents()

            if use_cache:
                print(f"Saving to cache: {cache_file}")
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'embeddings': self.embeddings,
                        'attention_masks': self.attention_masks,
                        'num_nodes_list': self.num_nodes_list
                    }, f)

    def clean_text(self, text: str) -> str:
        no_pct = text.translate(str.maketrans('', '', string.punctuation))
        return re.sub(r'\s+', ' ', no_pct)

    def _preprocess_all_documents(self):
        self.embeddings = []
        self.attention_masks = []
        self.num_nodes_list = []

        with torch.no_grad():
            for idx, text in enumerate(self.texts):
                if idx % 500 == 0:
                    print(f"Processing document {idx}/{len(self.texts)}...")

                if self.remove_punctualization:
                    text = self.clean_text(text)

                encoded = self.tokenizer(
                    text, max_length=self.max_length, padding='max_length',
                    truncation=True, return_tensors='pt')

                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)

                outputs = self.encoder(
                    input_ids=input_ids, attention_mask=attention_mask)
                token_embeddings = outputs.last_hidden_state.squeeze(0)

                num_actual_tokens = attention_mask.sum().item()

                self.embeddings.append(token_embeddings.cpu())
                self.attention_masks.append(attention_mask.squeeze(0).cpu())
                self.num_nodes_list.append(num_actual_tokens)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'node_features': self.embeddings[idx],
            'attention_mask': self.attention_masks[idx],
            'num_nodes': self.num_nodes_list[idx],
            'label': self.labels[idx],
            'edge_index': None,
            'text': self.texts[idx]
        }

    def get_sample_tokens(self, idx):
        encoded = self.tokenizer(
            self.texts[idx], max_length=self.max_length, truncation=True)
        return self.tokenizer.convert_ids_to_tokens(encoded['input_ids'])


class MRDGraphDataset(torch.utils.data.Dataset):
    """Movie Review Data dataset for sentiment regression."""

    def __init__(
        self,
        split: Literal['train', 'val', 'test'] = 'train',
        max_length: int = 1000,
        cache_dir: str = '../data/mrd_cache',
        device: str = 'cpu',
        use_cache: bool = True,
        freeze_encoder: bool = True,
        remove_punctualization: bool = False,
        encoder_name: str = 'distilbert-base-uncased',
        encoder_class=None,
        file_path: str = '../data/mrd/mrd.txt',
        val_split: float = 0.2,
        seed: int = 42
    ):
        self.split = split
        self.max_length = max_length
        self.device = device
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.remove_punctualization = remove_punctualization
        self.encoder_name = encoder_name
        self.file_path = file_path
        self.val_split = val_split
        self.seed = seed

        os.makedirs(cache_dir, exist_ok=True)

        if encoder_class is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
            self.encoder = encoder_class.from_pretrained(encoder_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
            self.encoder = AutoModel.from_pretrained(encoder_name)

        self.encoder.eval()
        self.encoder.to(device)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        print(f"Loading MRD {split} data...")
        train_instances, dev_instances, test_instances = self._load_mrd_data(
            file_path, val_split, seed)

        if split == 'train':
            instances = train_instances
        elif split == 'val':
            instances = dev_instances
        else:  # test
            instances = test_instances

        self.texts = [' '.join(word_list) for word_list, rating in instances]
        self.scores = torch.FloatTensor(
            [rating for word_list, rating in instances])

        print(f"Loaded {len(self.texts)} {split} instances")

        encoder_key = self.encoder_name.replace("/", "_")
        cache_file = os.path.join(
            cache_dir,
            f'mrd_{split}_maxlen{max_length}_split{val_split}_seed{seed}_{encoder_key}.pkl'
        )

        if use_cache and os.path.exists(cache_file):
            print(f"Loading cached embeddings from {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.embeddings = cache_data['embeddings']
                self.attention_masks = cache_data['attention_masks']
                self.num_nodes_list = cache_data['num_nodes_list']
        else:
            self._preprocess_all_documents()

            if use_cache:
                print(f"Saving to cache: {cache_file}")
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'embeddings': self.embeddings,
                        'attention_masks': self.attention_masks,
                        'num_nodes_list': self.num_nodes_list
                    }, f)

    @staticmethod
    def _load_mrd_data(file_path, data_split, seed):
        # from IDGL code
        def tokenize(text):
            """Simple whitespace tokenization."""
            return text.split()

        all_instances = []
        all_seq_len = []

        with open(file_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue
                idx, rating, subj = parts
                word_list = tokenize(subj.lower())
                all_instances.append([word_list, float(rating)])
                all_seq_len.append(len(word_list))

        print(f'[ Max seq length: {np.max(all_seq_len)} ]')
        print(f'[ Min seq length: {np.min(all_seq_len)} ]')
        print(f'[ Mean seq length: {int(np.mean(all_seq_len))} ]')

        # Random data split
        train_ratio = 0.6
        dev_ratio = data_split
        test_ratio = 1 - train_ratio - dev_ratio
        assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6

        n_train = int(len(all_instances) * train_ratio)
        n_dev = int(len(all_instances) * dev_ratio)
        n_test = len(all_instances) - n_train - n_dev

        random = np.random.RandomState(seed)
        random.shuffle(all_instances)

        train_instances = all_instances[:n_train]
        dev_instances = all_instances[n_train: n_train + n_dev]
        test_instances = all_instances[-n_test:]

        print(
            f'[ Train: {len(train_instances)}, Dev: {len(dev_instances)}, Test: {len(test_instances)} ]')

        return train_instances, dev_instances, test_instances

    def clean_text(self, text: str) -> str:
        no_pct = text.translate(str.maketrans('', '', string.punctuation))
        return re.sub(r'\s+', ' ', no_pct)

    def _preprocess_all_documents(self):
        self.embeddings = []
        self.attention_masks = []
        self.num_nodes_list = []

        with torch.no_grad():
            for idx, text in enumerate(self.texts):
                if idx % 500 == 0:
                    print(f"Processing document {idx}/{len(self.texts)}...")

                if self.remove_punctualization:
                    text = self.clean_text(text)

                encoded = self.tokenizer(
                    text, max_length=self.max_length, padding='max_length',
                    truncation=True, return_tensors='pt')

                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)

                outputs = self.encoder(
                    input_ids=input_ids, attention_mask=attention_mask)
                token_embeddings = outputs.last_hidden_state.squeeze(0)

                num_actual_tokens = attention_mask.sum().item()

                self.embeddings.append(token_embeddings.cpu())
                self.attention_masks.append(attention_mask.squeeze(0).cpu())
                self.num_nodes_list.append(num_actual_tokens)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'node_features': self.embeddings[idx],  # [max_length, 768]
            'attention_mask': self.attention_masks[idx],  # [max_length]
            'num_nodes': self.num_nodes_list[idx],  # int
            'label': self.scores[idx],  # float (regression score)
            'edge_index': None,
            'text': self.texts[idx]
        }

    def get_sample_tokens(self, idx):
        encoded = self.tokenizer(
            self.texts[idx], max_length=self.max_length, truncation=True)
        return self.tokenizer.convert_ids_to_tokens(encoded['input_ids'])
