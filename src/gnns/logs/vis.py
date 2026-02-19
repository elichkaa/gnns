import torch
import matplotlib.pyplot as plt
import networkx as nx
from transformers import DistilBertTokenizer
from gnns.dgm.model_v2 import DGM_Model
import numpy as np
from gnns.datasets import NewsGroupsGraphDataset, MRDGraphDataset
import os
from transformers import AutoTokenizer, AutoModel

# dDGM_distilbert-base-uncased_k10_gat_euclidean_poolmean
# dDGM_google-embeddinggemma-300m_k10_gat_euclidean_poolmean
prefix_dir = "../logs_20news/dDGM_google-embeddinggemma-300m_k15_gat_euclidean_poolmean/"
version = "version_3"
task = "classification"
save_dir = f"{prefix_dir}{version}/visualizations"
# encoder_name = 'distilbert-base-uncased'
encoder_name = 'google/embeddinggemma-300m'
# checkpoint = 'epoch=20-step=35092.ckpt'
# checkpoint = 'epoch=24-step=14150.ckpt'
# checkpoint = "epoch=49-step=28300.ckpt"
# epoch=13-step=7924.ckpt
# epoch=99-step=56600.ckpt
# epoch=29-step=16980.ckpt
# checkpoint = "epoch=17-step=1692.ckpt"
checkpoint = "epoch=17-step=5094.ckpt"


def visualize_dgm_graphs(dataset, model, num_samples=5, save_dir=save_dir, show_errors=False):
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
            logits_list = []
            adj_list = []
            for _ in range(5):
                logits, adj = model(node_features, return_adj=True)
                
                mask_expanded = attention_mask.unsqueeze(-1)
                logits_masked = logits * mask_expanded
                pooled_logits = logits_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
                
                logits_list.append(pooled_logits)
                adj_list.append(adj)
            
            logits = torch.stack(logits_list).mean(0)
            adj_matrix = torch.stack(adj_list).mean(0)
            pred_label = logits.squeeze(-1).item() if task == "regression" else logits.argmax(-1).item()
        if show_errors and pred_label == true_label:
            idx += 1
            continue
        encoded = tokenizer(text, max_length=dataset.max_length, truncation=True)
        tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
        
        tokens = tokens[:num_nodes] + ['[PAD]'] * max(0, num_nodes - len(tokens))
        
        adj = adj_matrix[0, :num_nodes, :num_nodes].cpu().numpy()
        
        print(f"\n=== Sample {idx} ===")
        print(f"True: {dataset.target_names[true_label]} | Pred: {dataset.target_names[pred_label]}")
        print(f"Num nodes: {num_nodes}, Total edges in adj: {(adj > 0.1).sum() // 2}")
        
        G = nx.Graph()
        for i in range(num_nodes):
            G.add_node(i, label=tokens[i])
        
        # Only add edges with weight > threshold
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if adj[i, j] > 0.1:  # Threshold for averaged probabilities
                    G.add_edge(i, j, weight=adj[i, j])
        
        print(f"Graph stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        degrees = dict(G.degree())
        if degrees:
            top_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"Top 10 hubs: {[(tokens[i] if i < len(tokens) else '[UNK]', deg) for i, deg in top_hubs]}")
        
        edges = list(G.edges())
        if edges:
            weights = [G[u][v]['weight'] for u, v in edges]
            top_edges = sorted(zip(edges, weights), key=lambda x: x[1], reverse=True)[:10]
            # print(f"Top 10 edges: {[f'{tokens[u] if u < len(tokens) else "[UNK]"}↔{tokens[v] if v < len(tokens) else "[UNK]"} ({w:.3f})' for (u, v), w in top_edges]}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        viz_size = min(50, num_nodes)
        im = ax1.imshow(adj[:viz_size, :viz_size], cmap='viridis', aspect='auto')
        ax1.set_xticks(range(0, viz_size, 5))
        ax1.set_yticks(range(0, viz_size, 5))
        ax1.set_xticklabels([tokens[i] if i < len(tokens) else '[UNK]' for i in range(0, viz_size, 5)], rotation=90, fontsize=6)
        ax1.set_yticklabels([tokens[i] if i < len(tokens) else '[UNK]' for i in range(0, viz_size, 5)], fontsize=6)
        ax1.set_title(f'Adjacency Matrix (first {viz_size} tokens, averaged over 5 samples)')
        plt.colorbar(im, ax=ax1)
        
        if len(top_hubs) > 0:
            top_node_ids = [i for i, _ in top_hubs[:min(30, len(top_hubs))]]
            G_sub = G.subgraph(top_node_ids)
            
            if G_sub.number_of_nodes() > 0:
                pos = nx.spring_layout(G_sub, k=2, iterations=50)
                node_labels = {i: tokens[i] if i < len(tokens) else '[UNK]' for i in G_sub.nodes()}
                
                nx.draw_networkx_nodes(G_sub, pos, node_size=300, node_color='lightblue', ax=ax2)
                nx.draw_networkx_labels(G_sub, pos, node_labels, font_size=6, ax=ax2)
                
                edges_sub = G_sub.edges()
                if edges_sub:
                    weights_sub = [G_sub[u][v]['weight'] for u, v in edges_sub]
                    nx.draw_networkx_edges(G_sub, pos, width=[w*2 for w in weights_sub],
                                           alpha=0.6, edge_color='gray', ax=ax2)
        
        ax2.set_title(f'True: {dataset.target_names[true_label]} | Pred: {dataset.target_names[pred_label]}')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/graph_sample{idx}.png', dpi=150, bbox_inches='tight')
        plt.close()


dgm_model = DGM_Model.load_from_checkpoint(
    f'{prefix_dir}{version}/checkpoints/{checkpoint}')
dgm_model = dgm_model.cuda()

test_data = NewsGroupsGraphDataset(
    split='test', max_length=512, device='cuda', use_cache=False, encoder_name=encoder_name)

# test_data = MRDGraphDataset(split='test', device='cuda', file_path="../../data/mrd/mrd.txt", max_length=512, use_cache=False, encoder_name=encoder_name)

visualize_dgm_graphs(test_data, dgm_model, num_samples=30, show_errors=False)