import torch
import matplotlib.pyplot as plt
import networkx as nx
from transformers import DistilBertTokenizer
from gnns.dgm.model import DGM_Model
import numpy as np
from gnns.datasets import NewsGroupsGraphDataset
import os
from transformers import AutoTokenizer, AutoModel

prefix_dir = "../new/dgm2/logs/baseline_google-embeddinggemma-300m_poolmean/"
save_dir = f"{prefix_dir}visualizations"
version = "version_1"
encoder_name = 'google/embeddinggemma-300m'


def visualize_dgm_graphs(dataset, model, num_samples=5, save_dir=save_dir):
    os.makedirs(save_dir, exist_ok=True)

    device = next(model.parameters()).device
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    model.eval()

    for idx in range(min(num_samples, len(dataset))):
        sample = dataset[idx]
        text = sample['text']
        true_label = sample['label'].item()

        node_features = sample['node_features'].unsqueeze(0).to(device)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
        num_nodes = sample['num_nodes']

        with torch.no_grad():
            logits, adj_matrix = model(
                node_features, attention_mask, return_adj=True)
            pred_label = logits.argmax(-1).item()

        encoded = tokenizer(
            text, max_length=dataset.max_length, truncation=True)
        tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])

        adj = adj_matrix[0, :num_nodes, :num_nodes].cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        im = ax1.imshow(adj, cmap='viridis', aspect='auto')
        ax1.set_xticks(range(min(30, num_nodes)))
        ax1.set_yticks(range(min(30, num_nodes)))
        ax1.set_xticklabels(
            tokens[:min(30, num_nodes)], rotation=90, fontsize=8)
        ax1.set_yticklabels(tokens[:min(30, num_nodes)], fontsize=8)
        ax1.set_title('Learned Adjacency Matrix (first 30 tokens)')
        plt.colorbar(im, ax=ax1)

        G = nx.Graph()

        for i in range(min(30, num_nodes)):
            G.add_node(i, label=tokens[i])

        threshold = np.percentile(adj.flatten(), 95)
        for i in range(min(30, num_nodes)):
            for j in range(i+1, min(30, num_nodes)):
                if adj[i, j] > threshold:
                    G.add_edge(i, j, weight=adj[i, j])

        pos = nx.spring_layout(G, k=2, iterations=50)
        node_labels = {i: tokens[i] for i in G.nodes()}

        nx.draw_networkx_nodes(G, pos, node_size=300,
                               node_color='lightblue', ax=ax2)
        nx.draw_networkx_labels(G, pos, node_labels, font_size=8, ax=ax2)

        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w*3 for w in weights],
                               alpha=0.6, edge_color='gray', ax=ax2)

        ax2.set_title(
            f'Learned Graph (top 5% edges)\nTrue: {dataset.target_names[true_label]} | Pred: {dataset.target_names[pred_label]}')
        ax2.axis('off')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/graph_sample{idx}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n=== Sample {idx} ===")
        print(
            f"True: {dataset.target_names[true_label]} | Pred: {dataset.target_names[pred_label]}")
        print(
            f"Graph stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        degrees = dict(G.degree())
        if degrees:
            top_hubs = sorted(
                degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"Top hubs: {[(tokens[i], deg) for i, deg in top_hubs]}")

        if edges:
            top_edges = sorted(zip(edges, weights),
                               key=lambda x: x[1], reverse=True)[:5]
            print(f"Strongest edges: {[f'{tokens[u]}↔{tokens[v]} ({w:.3f})' for (
                u, v), w in top_edges]}")


def visualize_dgm_token_importance(dataset, model, num_samples=5, save_dir=save_dir):
    os.makedirs(save_dir, exist_ok=True)

    device = next(model.parameters()).device
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model.eval()

    for idx in range(min(num_samples, len(dataset))):
        sample = dataset[idx]
        text = sample['text']
        true_label = sample['label'].item()

        node_features = sample['node_features'].unsqueeze(0).to(device)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
        num_nodes = sample['num_nodes']

        baseline_emb = torch.zeros_like(node_features)
        num_steps = 50
        attributions = torch.zeros_like(node_features)

        for step in range(num_steps):
            alpha = step / num_steps
            interpolated = baseline_emb + alpha * \
                (node_features - baseline_emb)
            interpolated.requires_grad_(True)

            logits = model(interpolated, attention_mask)[0]
            pred_label = logits.argmax(-1).item()
            score = logits[0, pred_label]

            score.backward()
            attributions += interpolated.grad
            model.zero_grad()

        attributions = (node_features - baseline_emb) * \
            attributions / num_steps
        importance = attributions[0].norm(dim=-1).cpu().numpy()

        encoded = tokenizer(
            text, max_length=dataset.max_length, truncation=True)
        tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])

        top_k = min(20, num_nodes)
        top_indices = np.argsort(importance[:num_nodes])[-top_k:][::-1]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Blues(
            importance[top_indices] / importance[top_indices].max())
        ax.barh(range(top_k), importance[top_indices], color=colors)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([tokens[i] for i in top_indices], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Integrated Gradient Attribution (DGM)')
        ax.set_title(
            f'Top {top_k} tokens (with graph)\nTrue: {dataset.target_names[true_label]} | Pred: {dataset.target_names[pred_label]}')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/importance_sample{idx}.png', dpi=150)
        plt.close()

        print(
            f"Sample {idx} top tokens: {[tokens[i] for i in top_indices[:5]]}")


dgm_model = DGM_Model.load_from_checkpoint(
    f'{prefix_dir}{version}/checkpoints/epoch=29-step=8490.ckpt')

test_data = NewsGroupsGraphDataset(
    split='test', max_length=512, device='cuda', use_cache=True)

visualize_dgm_graphs(test_data, dgm_model, num_samples=5)
visualize_dgm_token_importance(test_data, dgm_model, num_samples=5)
