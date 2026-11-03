"""
Script Function: Evaluates representation generalization under an extreme 50% train / 50% unseen test split. Conducts an automated grid search for optimal Louvain clustering, computes biological conservation metrics (AvgBIO, NMI, ARI, ASW), and plots comparative UMAPs and bar charts.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pathjoin = os.path.join

from model import CSGNet
from datasets import MyDataset

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

seed_everything(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_data_dir = os.path.expanduser("~/scMORETA/dataset/pre_data/PBMC_10k_Processed")
output_dir = os.path.expanduser("~/scMORETA/result/PBMC_10k_Processed_50_50")
models_dir = pathjoin(output_dir, 'models')
plot_dir = pathjoin(output_dir, 'eval_plots')
os.makedirs(plot_dir, exist_ok=True)
# Perform forward inference through model encoder to extract high-dimensional latent cell embeddings
def extract_features(model, data_x):
    loader = DataLoader(MyDataset(data_x, np.zeros(len(data_x))), batch_size=512, shuffle=False)
    latents = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            lat, _ = model(x, get_latent_variable=True)
            latents.append(lat.cpu().numpy())
    return np.vstack(latents)
# Grid search optimal Louvain resolution (0.1~2.0) to evaluate Silhouette (ASW), NMI, ARI, and the composite AvgBIO score
def evaluate_subset(latents, labels, split_name="Test"):
    raw_asw = silhouette_score(latents, labels, metric='euclidean')
    asw_cell = (raw_asw + 1.0) / 2.0
    adata = sc.AnnData(X=latents)
    sc.pp.neighbors(adata, use_rep='X', n_neighbors=15, metric='euclidean')

    best_nmi = -1.0
    best_clusters = None
    best_res = 0.1

    for res in np.arange(0.1, 2.1, 0.1):
        res_val = round(float(res), 1)
        sc.tl.louvain(adata, resolution=res_val, key_added='cluster_temp', random_state=42)
        pred = adata.obs['cluster_temp'].values
        curr_nmi = normalized_mutual_info_score(labels, pred)
        if curr_nmi > best_nmi:
            best_nmi = curr_nmi
            best_clusters = pred.copy()
            best_res = res_val

    nmi_cell = best_nmi
    ari_cell = adjusted_rand_score(labels, best_clusters)
    avg_bio = (ari_cell + nmi_cell + asw_cell) / 3.0
    labels_int = labels.astype(int)
    adata.obs['celltype'] = pd.Categorical(
        labels_int, 
        categories=np.sort(np.unique(labels_int))
    )

    clusters_int = best_clusters.astype(int)
    adata.obs['Louvain_best'] = pd.Categorical(
        clusters_int, 
        categories=np.sort(np.unique(clusters_int))
    )
    
    print(f"   ↳ [{split_name} best Louvain (res={best_res})]: NMI={nmi_cell:.4f}, ARI={ari_cell:.4f}, ASW={asw_cell:.4f}, AvgBIO={avg_bio:.4f}")

    sc.tl.umap(adata, min_dist=0.3, random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sc.pl.umap(adata, color='celltype', ax=axes[0], show=False, title=f'{split_name} Set: Ground Truth', frameon=False)
    sc.pl.umap(adata, color='Louvain_best', ax=axes[1], show=False, title=f'{split_name} Set: Louvain Clusters (res={best_res})', frameon=False)
    plt.tight_layout()
    
    base_save_name = f'umap_{split_name.lower()}_comparison'
    png_path = pathjoin(plot_dir, f'{base_save_name}.png')
    svg_path = pathjoin(plot_dir, f'{base_save_name}.svg')
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"PNG : {png_path}")
    print(f"SVG : {svg_path}")

    return {
        'Split': split_name,
        'AvgBIO': round(avg_bio, 4),
        'NMI_cell': round(nmi_cell, 4),
        'ARI_cell': round(ari_cell, 4),
        'ASW_cell': round(asw_cell, 4),
        'Best_Res': best_res
    }

def main():
    print("=" * 60)
    split_file = pathjoin(output_dir, 'split_indices_50_50.npz')
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"error: {split_file}，please run train_50_50.py！")
    split_data = np.load(split_file)
    tr_idx, te_idx = split_data['train_idx'], split_data['test_idx']
    raw_rna = np.load(pathjoin(base_data_dir, 'RNA_matrix.npy')).astype(np.float32)
    labels = np.load(pathjoin(base_data_dir, 'Labels.npy')).astype(np.int64)
    model_path = pathjoin(models_dir, 'tcsgn_50_50_best_model.pth')
    adj_path = pathjoin(models_dir, 'global_adj_50_50.pt')

    model = torch.load(model_path, map_location=device, weights_only=False)
    global_adj = torch.load(adj_path, map_location=device)
    model.register_buffer('global_H', global_adj)
    model.eval()
    train_latents = extract_features(model, raw_rna[tr_idx])
    train_labels = labels[tr_idx]
    test_latents = extract_features(model, raw_rna[te_idx])
    test_labels = labels[te_idx]
    res_train = evaluate_subset(train_latents, train_labels, split_name="Train_50pct")
    res_test = evaluate_subset(test_latents, test_labels, split_name="Test_50pct_Unseen")
    df_results = pd.DataFrame([res_train, res_test])
    print("=" * 60)
    print(df_results.to_string(index=False))
    print("=" * 60)

    csv_path = pathjoin(plot_dir, "tcsgn_50_50_evaluation_metrics.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"saved: {csv_path}")
    metrics = ['AvgBIO', 'NMI_cell', 'ARI_cell', 'ASW_cell']
    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, [res_train[m] for m in metrics], width, label='Train Set (50%)', color='#6C8EBF')
    plt.bar(x + width/2, [res_test[m] for m in metrics], width, label='Test Set (50% Unseen)', color='#D79B00')
    
    plt.ylabel('Score', fontsize=12)
    plt.title('T-CSGN Representation Generalization (50% Train vs 50% Test)', fontsize=14)
    plt.xticks(x, metrics, fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(frameon=True, fontsize=11)
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

    for i in x:
        plt.text(i - width/2, res_train[metrics[i]] + 0.02, f"{res_train[metrics[i]]:.3f}", ha='center', fontsize=9)
        plt.text(i + width/2, res_test[metrics[i]] + 0.02, f"{res_test[metrics[i]]:.3f}", ha='center', fontsize=9)

    plt.tight_layout()
    bar_png = pathjoin(plot_dir, 'train_vs_test_50_50_metrics_barplot.png')
    bar_svg = pathjoin(plot_dir, 'train_vs_test_50_50_metrics_barplot.svg')
    
    plt.savefig(bar_png, dpi=600, bbox_inches='tight')
    plt.savefig(bar_svg, format='svg', bbox_inches='tight')
    plt.close()

    print(f"PNG : {bar_png}")
    print(f"SVG : {bar_svg}")

if __name__ == '__main__':
    main()
