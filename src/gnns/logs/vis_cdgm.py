import torch
import matplotlib.pyplot as plt
import networkx as nx
from transformers import AutoTokenizer
from gnns.dgm.cdgm import cDGM_GNN
import numpy as np
from gnns.datasets import NewsGroupsGraphDataset
import os
from argparse import Namespace

prefix_dir = "../logs/dDGM_google-embeddinggemma-300m_k5_gat_euclidean_poolmean/"
version = "version_49"
save_dir = f"{prefix_dir}{version}/visualizations"
encoder_name = 'google/embeddinggemma-300m'
# epoch=99-step=56600.ckpt
# epoch=49-step=14150.ckpt
checkpoint = 'epoch=99-step=56600.ckpt'


def visualize_cdgm_graphs(dataset, model, num_samples=5, save_dir=save_dir):
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
            logits, adjacencies = model(node_features, attention_mask)
            pred_label = logits.argmax(-1).item()
            
            adj_matrix = adjacencies[-1]  # [batch, num_nodes, num_nodes]
        
        encoded = tokenizer(text, max_length=dataset.max_length, truncation=True)
        tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
        tokens = tokens[:num_nodes] + ['[PAD]'] * max(0, num_nodes - len(tokens))
        
        adj = adj_matrix[0, :num_nodes, :num_nodes].cpu().numpy()
        
        print(f"\n=== Sample {idx} ===")
        print(f"True: {dataset.target_names[true_label]} | Pred: {dataset.target_names[pred_label]}")
        print(f"Num nodes: {num_nodes}")
        print(f"Adjacency stats: min={adj.min():.3f}, max={adj.max():.3f}, mean={adj.mean():.3f}")
        
        G = nx.Graph()
        for i in range(num_nodes):
            G.add_node(i, label=tokens[i])
        
        threshold = 0.5
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if adj[i, j] > threshold:
                    G.add_edge(i, j, weight=adj[i, j])
        
        print(f"Graph stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (threshold={threshold})")
        
        degrees = dict(G.degree())
        if degrees:
            top_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"Top 10 hubs: {[(tokens[i] if i < len(tokens) else '[UNK]', deg) for i, deg in top_hubs]}")
        
        edges = list(G.edges())
        if edges:
            weights = [G[u][v]['weight'] for u, v in edges]
            top_edges = sorted(zip(edges, weights), key=lambda x: x[1], reverse=True)[:10]
            # print(f"Top 10 edges: {[f'{tokens[u] if u < len(tokens) else \"[UNK]\"}↔{tokens[v] if v < len(tokens) else \"[UNK]\"} ({w:.3f})' for (u, v), w in top_edges]}")
        
        fig, axes = plt.subplots(1, len(adjacencies) + 1, figsize=(6 * (len(adjacencies) + 1), 6))
        
        for layer_idx, adj_layer in enumerate(adjacencies):
            ax = axes[layer_idx]
            adj_viz = adj_layer[0, :num_nodes, :num_nodes].cpu().numpy()
            
            viz_size = min(50, num_nodes)
            im = ax.imshow(adj_viz[:viz_size, :viz_size], cmap='viridis', aspect='auto', vmin=0, vmax=1)
            ax.set_title(f'Layer {layer_idx+1} Adjacency\n(first {viz_size} tokens)')
            ax.set_xticks(range(0, viz_size, 10))
            ax.set_yticks(range(0, viz_size, 10))
            ax.set_xticklabels([tokens[i] if i < len(tokens) else '[UNK]' for i in range(0, viz_size, 10)], 
                              rotation=90, fontsize=6)
            ax.set_yticklabels([tokens[i] if i < len(tokens) else '[UNK]' for i in range(0, viz_size, 10)], 
                              fontsize=6)
            plt.colorbar(im, ax=ax)
        
        ax = axes[-1]
        if len(top_hubs) > 0:
            top_node_ids = [i for i, _ in top_hubs[:min(30, len(top_hubs))]]
            G_sub = G.subgraph(top_node_ids)
            
            if G_sub.number_of_nodes() > 0:
                pos = nx.spring_layout(G_sub, k=2, iterations=50)
                node_labels = {i: tokens[i] if i < len(tokens) else '[UNK]' for i in G_sub.nodes()}
                
                node_colors = ['lightgreen' if i == true_label else 'lightblue' for i in G_sub.nodes()]
                nx.draw_networkx_nodes(G_sub, pos, node_size=300, node_color='lightblue', ax=ax)
                nx.draw_networkx_labels(G_sub, pos, node_labels, font_size=7, ax=ax)
                
                edges_sub = G_sub.edges()
                if edges_sub:
                    weights_sub = [G_sub[u][v]['weight'] for u, v in edges_sub]
                    nx.draw_networkx_edges(G_sub, pos, width=[w*3 for w in weights_sub],
                                         alpha=0.6, edge_color='gray', ax=ax)
        
        correct = "✓" if pred_label == true_label else "✗"
        ax.set_title(f'Graph Structure (top 30 hubs)\nTrue: {dataset.target_names[true_label]}\nPred: {dataset.target_names[pred_label]} {correct}')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/cdgm_sample{idx}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization to {save_dir}/cdgm_sample{idx}.png")


cdgm_model = cDGM_GNN.load_from_checkpoint(
    f'{prefix_dir}{version}/checkpoints/{checkpoint}')
cdgm_model = cdgm_model.cuda()
cdgm_model.eval()

test_data = NewsGroupsGraphDataset(
    split='test', 
    max_length=256, 
    device='cuda', 
    use_cache=True, 
    encoder_name=encoder_name
)

visualize_cdgm_graphs(test_data, cdgm_model, num_samples=10)