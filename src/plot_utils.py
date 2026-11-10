import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import networkx as nx
import torch
import umap
import matplotlib.patheffects as PathEffects
import os
def plot_learning_curve(history_df, save_path, dataset_name=None, dpi=300):
    plt.figure(figsize=(12, 5))
    
    loss_title = f'Training Loss ({dataset_name})' if dataset_name else 'Training Loss'
    acc_title = f'Training Accuracy ({dataset_name})' if dataset_name else 'Training Accuracy'
    
    plt.subplot(1, 2, 1)
    plt.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss', color='tab:blue')
    plt.title(loss_title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history_df['epoch'], history_df['train_acc'], label='Train Acc', color='tab:orange')
    plt.title(acc_title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
def split_two_lines(name, max_len=14):
    if len(name) <= max_len:
        return name
    spaces = [i for i, c in enumerate(name) if c == ' ']
    if not spaces:
        return name
    mid = len(name) / 2.0
    best_space = min(spaces, key=lambda s: abs(s - mid))
    return name[:best_space] + '\n' + name[best_space+1:]

def plot_confusion_matrix(y_true, y_pred, classes, save_path, dataset_name=None, dpi=300):
    cm = confusion_matrix(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    formatted_classes = [split_two_lines(str(c), max_len=14) for c in classes]

    plt.figure(figsize=(10, 8.5))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=formatted_classes, yticklabels=formatted_classes)

    if dataset_name:
        clean_title = dataset_name.replace('_Processed', '')
        plt.title(clean_title, fontsize=14, pad=12)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_score, n_classes, classes, save_path, dataset_name=None, dpi=300):
    y_test_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        if np.sum(y_test_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{classes[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    if dataset_name:
        plt.title(dataset_name, fontsize=14, pad=12)
    else:
        plt.title('', fontsize=14, pad=12)
        
    plt.legend(loc="lower right", fontsize='small')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

def plot_umap(latent, labels, classes, save_path, dataset_name=None, dpi=300, add_text_labels=True):
    
    reducer = umap.UMAP(
        n_neighbors=15, 
        min_dist=0.5, 
        n_components=2, 
        metric='cosine', 
        random_state=42
    )
    
    X_embedded = reducer.fit_transform(latent)
    df_umap = pd.DataFrame(X_embedded, columns=['UMAP1', 'UMAP2'])
    
    if isinstance(labels, list): 
        labels = np.array(labels)
    if isinstance(classes, torch.Tensor):
        classes = classes.cpu().numpy()
    label_map = {i: str(name) for i, name in enumerate(classes)}
    df_umap['Cell Type'] = [label_map[l] for l in labels]

    num_points = latent.shape[0]
    s_val = 120000 / num_points 
    s_val = min(max(s_val, 1), 20)

    unique_types = np.unique(df_umap['Cell Type'])
    n_types = len(unique_types)
    if n_types <= 10:
        palette_list = sns.color_palette("tab10", n_types)
    elif n_types <= 20:
        palette_list = sns.color_palette("tab20", n_types)
    else:
        palette_list = sns.color_palette("husl", n_types)
    color_map = {ctype: palette_list[i] for i, ctype in enumerate(unique_types)}
    
    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        data=df_umap, 
        x='UMAP1', 
        y='UMAP2', 
        hue='Cell Type', 
        palette=color_map,
        hue_order=unique_types,
        s=s_val, 
        alpha=0.8, 
        edgecolor=None, 
        legend=False if add_text_labels else 'auto'
    )
    
    if add_text_labels:
        unique_labels = np.unique(labels)
        for label_idx in unique_labels:
            indices = np.where(labels == label_idx)[0]
            if len(indices) == 0: 
                continue
            
            x_center = np.median(X_embedded[indices, 0])
            y_center = np.median(X_embedded[indices, 1])
            
            label_text = label_map[label_idx]
            
            txt = plt.text(
                x_center, y_center, 
                label_text, 
                fontsize=11, 
                fontweight='bold',
                ha='center', va='center',
                color='black'
            )
            txt.set_path_effects([
                PathEffects.withStroke(linewidth=3, foreground='w', alpha=0.8)
            ])

    plt.axis('off')
    
    if dataset_name:
        plt.title(f'{dataset_name}', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  -> UMAP 已保存: {save_path}")
def plot_gene_skeleton(matrix, gene_names, save_path, dataset_name=None, dpi=300, top_k=300):
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().detach().numpy()

    if matrix.shape[0] != matrix.shape[1]:
        matrix_abs = np.abs(matrix)
        adj_matrix = np.dot(matrix_abs, matrix_abs.T)
    else:
        adj_matrix = np.abs(matrix)

    np.fill_diagonal(adj_matrix, 0)

    max_val = adj_matrix.max()

    if max_val == 0:
        print("Warning: This cell has no active expression in the current feature set (all-zero matrix), skipping plotting.")
        return

    rows, cols = np.where(adj_matrix > 0)
    weights = adj_matrix[rows, cols]
    edges_zip = zip(rows, cols, weights)
    sorted_edges = sorted(edges_zip, key=lambda x: x[2], reverse=True)
    
    G = nx.Graph()
    count = 0
    for u, v, w in sorted_edges:
        if u == v: 
            continue
        if not G.has_edge(u, v):
            G.add_edge(u, v, weight=w)
            count += 1
        if count >= top_k:
            break

    mapping = {i: str(name) for i, name in enumerate(gene_names)}
    G = nx.relabel_nodes(G, mapping)
    
    if G.number_of_nodes() == 0:
        print("Warning: No edges in the graph, skipping plotting.")
        return

    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, seed=42, k=0.3, iterations=50)

    edges = G.edges(data=True)
    raw_weights = np.array([d['weight'] for u, v, d in edges])
    if len(raw_weights) > 0:
        w_min, w_max = raw_weights.min(), raw_weights.max()
        if w_max > w_min:
            widths = 0.5 + 2.5 * (raw_weights - w_min) / (w_max - w_min)
        else:
            widths = 1.0
    else:
        widths = 1.0

    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='#87CEFA', alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='#333333')
    nx.draw_networkx_edges(G, pos, width=widths, edge_color='gray', alpha=0.4)

    if dataset_name:
        plt.title(f'Global Gene Topological Skeleton ({dataset_name}, Top {top_k} Links)', fontsize=24)
    else:
        plt.title(f'Global Gene Topological Skeleton (Top {top_k} Links)', fontsize=24)
        
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
