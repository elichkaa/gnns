import torch
import matplotlib.pyplot as plt
import networkx as nx
from transformers import DistilBertTokenizer
from gnns.dgm.model import DGM_Model
import numpy as np
from gnns.datasets import NewsGroupsGraphDataset
import os
from transformers import AutoTokenizer, AutoModel

prefix_dir = "../logs/dDGM_distilbert-base-uncased_k5_gat_euclidean_poolmean/"
save_dir = f"{prefix_dir}visualizations"
version = "version_1"
encoder_name = 'distilbert-base-uncased'
# encoder_name = 'google/embeddinggemma-300m'
checkpoint = 'epoch=30-step=35092.ckpt'
# checkpoint = 'epoch=29-step=16980.ckpt'
# checkpoint = "epoch=49-step=28300.ckpt"

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
        
        print(f"\n=== Sample {idx} ===")
        print(f"True: {dataset.target_names[true_label]} | Pred: {dataset.target_names[pred_label]}")
        print(f"Num nodes: {num_nodes}, Total edges in adj: {(adj > 0).sum() // 2}")
        
        G = nx.Graph()
        for i in range(min(num_nodes, len(tokens))):
            G.add_node(i, label=tokens[i])
        
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if adj[i, j] > 0:
                    G.add_edge(i, j, weight=adj[i, j])
        
        print(f"Graph stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        degrees = dict(G.degree())
        if degrees:
            top_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"Top 10 hubs: {[(tokens[i], deg) for i, deg in top_hubs]}")
        
        edges = list(G.edges())
        if edges:
            weights = [G[u][v]['weight'] for u, v in edges]
            top_edges = sorted(zip(edges, weights), key=lambda x: x[1], reverse=True)[:10]
            print(f"Top 10 edges: {[f'{tokens[u]}↔{tokens[v]} ({w:.3f})' for (u, v), w in top_edges]}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        viz_size = min(50, num_nodes)
        im = ax1.imshow(adj[:viz_size, :viz_size], cmap='viridis', aspect='auto')
        ax1.set_xticks(range(0, viz_size, 5))
        ax1.set_yticks(range(0, viz_size, 5))
        ax1.set_xticklabels(tokens[:viz_size:5], rotation=90, fontsize=6)
        ax1.set_yticklabels(tokens[:viz_size:5], fontsize=6)
        ax1.set_title(f'Adjacency Matrix (first {viz_size} tokens)')
        plt.colorbar(im, ax=ax1)
        
        top_node_ids = [i for i, _ in top_hubs[:30]]
        G_sub = G.subgraph(top_node_ids)
        
        pos = nx.spring_layout(G_sub, k=2, iterations=50)
        node_labels = {i: tokens[i] for i in G_sub.nodes()}
        
        nx.draw_networkx_nodes(G_sub, pos, node_size=300, node_color='lightblue', ax=ax2)
        nx.draw_networkx_labels(G_sub, pos, node_labels, font_size=6, ax=ax2)
        
        edges_sub = G_sub.edges()
        if edges_sub:
            weights_sub = [G_sub[u][v]['weight'] for u, v in edges_sub]
            nx.draw_networkx_edges(G_sub, pos, width=[w*2 for w in weights_sub],
                                   alpha=0.6, edge_color='gray', ax=ax2)
        
        ax2.set_title(f'Subgraph (top 30 hubs)\nTrue: {dataset.target_names[true_label]} | Pred: {dataset.target_names[pred_label]}')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/graph_sample{idx}.png', dpi=150, bbox_inches='tight')
        plt.close()


def visualize_dgm_token_importance(dataset, model, num_samples=5, save_dir=save_dir):
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
    f'{prefix_dir}{version}/checkpoints/{checkpoint}')

test_data = NewsGroupsGraphDataset(
    split='test', max_length=512, device='cuda', use_cache=True)

visualize_dgm_graphs(test_data, dgm_model, num_samples=10)
# visualize_dgm_token_importance(test_data, dgm_model, num_samples=5)
