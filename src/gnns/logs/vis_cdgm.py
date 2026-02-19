import torch
import matplotlib.pyplot as plt
import networkx as nx
from transformers import AutoTokenizer
from gnns.dgm.model_v2 import DGM_Model
import numpy as np
from gnns.datasets import NewsGroupsGraphDataset, MRDGraphDataset
import os
from argparse import Namespace

# prefix_dir = "../logs_mrd/cDGM_google-embeddinggemma-300m_k15_gat_euclidean_poolmean/"
prefix_dir = "../logs_20news/cDGM_google-embeddinggemma-300m_k10_gat_euclidean_poolmean/"
version = "version_1"
save_dir = f"{prefix_dir}{version}/visualizations"
encoder_name = 'google/embeddinggemma-300m'
# epoch=99-step=56600.ckpt
# epoch=49-step=14150.ckpt
checkpoint = 'epoch=32-step=9339.ckpt'


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
        num_nodes = attention_mask.sum().item()
        
        with torch.no_grad():
            logits, adjacencies = model(node_features, edge_index=None, return_adj=True)
            adj = adjacencies
            
            mask_expanded = attention_mask.unsqueeze(-1)
            logits_masked = logits * mask_expanded
            pooled_logits = logits_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            pred_label = pooled_logits.argmax(-1).item()
            # pred_label = pooled_logits.squeeze(-1).item()

            if adj.dim() == 2:
                adj = adj.unsqueeze(0)
        
        encoded = tokenizer(text, max_length=dataset.max_length, truncation=True)
        tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
        tokens = tokens[:num_nodes]
        
        adj_np = adj[0, :num_nodes, :num_nodes].cpu().numpy()
        
        print(f"\n=== Sample {idx} ===")
        print(f"True: {dataset.target_names[true_label]} | Pred: {dataset.target_names[pred_label]}")
        # print(f"True: {true_label} | Pred: {pred_label}")
        print(f"Num nodes: {num_nodes}")
        print(f"Adjacency stats: min={adj_np.min():.3f}, max={adj_np.max():.3f}, mean={adj_np.mean():.3f}")
        
        G = nx.Graph()
        for i in range(num_nodes):
            G.add_node(i, label=tokens[i] if i < len(tokens) else '[UNK]')
        
        threshold = 0.5
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if adj_np[i, j] > threshold:
                    G.add_edge(i, j, weight=adj_np[i, j])
        
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (threshold={threshold})")
        
        degrees = dict(G.degree())
        if degrees:
            top_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"Top hubs: {[(tokens[i] if i < len(tokens) else '[UNK]', deg) for i, deg in top_hubs]}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        viz_size = min(50, num_nodes)
        im = ax1.imshow(adj_np[:viz_size, :viz_size], cmap='viridis', vmin=0, vmax=1)
        ax1.set_title(f'Adjacency Matrix (first {viz_size} tokens)')
        plt.colorbar(im, ax=ax1)
        
        if len(top_hubs) > 0:
            top_nodes = [i for i, _ in top_hubs[:min(30, len(top_hubs))]]
            G_sub = G.subgraph(top_nodes)
            
            if G_sub.number_of_nodes() > 0:
                pos = nx.spring_layout(G_sub, k=2, iterations=50)
                node_labels = {i: tokens[i] if i < len(tokens) else '[UNK]' for i in G_sub.nodes()}
                
                nx.draw_networkx_nodes(G_sub, pos, node_size=300, node_color='lightblue', ax=ax2)
                nx.draw_networkx_labels(G_sub, pos, node_labels, font_size=7, ax=ax2)
                
                if G_sub.edges():
                    weights = [G_sub[u][v]['weight'] for u, v in G_sub.edges()]
                    nx.draw_networkx_edges(G_sub, pos, width=[w*3 for w in weights],
                                         alpha=0.6, edge_color='gray', ax=ax2)
        
        ax2.set_title(f'Graph (top 30 hubs)\nTrue: {dataset.target_names[true_label]} | Pred: {dataset.target_names[pred_label]}')
        # ax2.set_title(f'Graph (top 30 hubs)\nTrue: {true_label} | Pred: {pred_label}')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/cdgm_sample{idx}.png', dpi=150)
        plt.close()


cdgm_model = DGM_Model.load_from_checkpoint(
    f'{prefix_dir}{version}/checkpoints/{checkpoint}')
cdgm_model = cdgm_model.cuda()
cdgm_model.eval()

test_data = NewsGroupsGraphDataset(
    split='test', 
    max_length=512, 
    device='cuda', 
    use_cache=False, 
    encoder_name=encoder_name
)

# test_data = MRDGraphDataset(split='test', device='cuda', file_path="../../data/mrd/mrd.txt", max_length=512, use_cache=False, encoder_name=encoder_name)

visualize_cdgm_graphs(test_data, cdgm_model, num_samples=10)