"""
Script Function: In silico single-gene digital knockout perturbation analysis. Suppresses the expression of canonical Naive B cell marker genes (TCL1A, CD79A, IL4R, MS4A1) by 99% to quantify the retention/decay of predicted cell-type probabilities with Wilcoxon signed-rank tests, generating both a 2x2 panel and individual boxplots (Figure 6C).
"""
import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

plt.rcParams['svg.fonttype'] = 'none'  
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

data_dir = "../dataset/pre_data/PBMC_10k_Processed"
model_path = "../result/PBMC_10k_Processed/models/tcsgn_PBMC_10k_Processed_full_model2.pth"
target_celltype = "Naive B"
TOP_K = 4
DPI = 300 

if not os.path.exists(model_path):
    alt_model = model_path.replace("model2", "model1")
    if os.path.exists(alt_model):
        model_path = alt_model

output_fig_dir = "./Figures_output"
os.makedirs(output_fig_dir, exist_ok=True)
X_rna = np.load(os.path.join(data_dir, "RNA_matrix.npy"))
labels = np.load(os.path.join(data_dir, "Labels.npy"))
gene_names = np.loadtxt(os.path.join(data_dir, "gene_names.csv"), dtype=str)
label_map = np.load(os.path.join(data_dir, "label_map.npy"), allow_pickle=True)
# Map target cell type (Naive B) to its integer class index and extract corresponding single-cell indices
if isinstance(label_map, dict):
    inv_map = {v: k for k, v in label_map.items()}
    target_class_idx = inv_map[target_celltype]
else:
    target_class_idx = np.where(label_map == target_celltype)[0][0]

cell_indices = np.where(labels == target_class_idx)[0]
print(f"in target cell find {len(cell_indices)}  [{target_celltype}] ")

candidate_markers = ['TCL1A', 'CD79A', 'IL4R', 'MS4A1']

target_genes_list = [g for g in candidate_markers if g in gene_names]
print(f"\nSelected Naive B marker genes for in silico knockout: {target_genes_list}")

if len(target_genes_list) < 4:
    missing = set(candidate_markers) - set(target_genes_list)
    print(f"Warning: Gene(s) {missing} not found in current preprocessed features. Please verify")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()
X_baseline = torch.tensor(X_rna[cell_indices]).float().to(device)

with torch.no_grad():
    logits_base = model(X_baseline)
    probs_base = torch.softmax(logits_base, dim=-1)
    prob_b_base = probs_base[:, target_class_idx].cpu().numpy()

base_mean = np.mean(prob_b_base)
base_median = np.median(prob_b_base)
print(f" μ = {base_mean:.4f}, median = {base_median:.4f}")

fig_all, axes = plt.subplots(2, 2, figsize=(13, 11))
axes = axes.flatten()
# Iterate over marker genes, downscale expression by 0.01 to simulate in silico knockout, and record retention rate (%)
for rank, (target_gene, ax) in enumerate(zip(target_genes_list, axes), 1):
    gene_idx = np.where(gene_names == target_gene)[0][0]

    X_knockout = X_rna[cell_indices].copy()
    X_knockout[:, gene_idx] = X_knockout[:, gene_idx] * 0.01
    X_knockout_tensor = torch.tensor(X_knockout).float().to(device)
    
    with torch.no_grad():
        logits_ko = model(X_knockout_tensor)
        probs_ko = torch.softmax(logits_ko, dim=-1)
        prob_b_ko = probs_ko[:, target_class_idx].cpu().numpy()
        
    ko_mean = np.mean(prob_b_ko)
    ko_median = np.median(prob_b_ko)
    drop_pct = (base_mean - ko_mean) / base_mean * 100
    retention_pct = 100.0 - drop_pct

    try:
        stat_w, p_val = wilcoxon(prob_b_base, prob_b_ko)
        p_text = f"p = {p_val:.2e}" if p_val < 0.001 else f"p = {p_val:.3f} (n.s.)"
    except:
        p_text = "p ≈ 1.00 (n.s.)"
        
    df_plot = pd.DataFrame({
        'Condition': ['Original'] * len(prob_b_base) + [f'Knockdown'] * len(prob_b_ko),
        'Probability': np.concatenate([prob_b_base, prob_b_ko])
    })

    sns.boxplot(
        x='Condition',
        y='Probability',
        hue='Condition',
        data=df_plot,
        palette=['#7C9D8E', '#D4A373'],  
        ax=ax,
        width=0.52,
        linewidth=1.6,
        fliersize=5.0,
        flierprops=dict(marker='o', markerfacecolor='white', markeredgecolor='#444444', markeredgewidth=1.1, alpha=0.8),
        legend=False
    )
    
    ax.text(
        0, 0.76, 
        f"Mean: {base_mean:.4f}\nMedian: {base_median:.4f}", 
        ha='center', va='center', fontsize=11.5, fontweight='medium', color='#1A3B2E',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F1EC', edgecolor='#7C9D8E', lw=1.1, alpha=0.9)
    )
    ax.text(
        1, 0.76, 
        f"Mean: {ko_mean:.4f}\nMedian: {ko_median:.4f}", 
        ha='center', va='center', fontsize=12, fontweight='medium', color='#4A2C11',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F9F2EA', edgecolor='#D4A373', lw=1.1, alpha=0.9)
    )
    ax.set_title(f"{target_gene}", fontsize=15.5, pad=12)
    ax.set_ylabel(f"Predicted Probability", fontsize=13.0,labelpad=8)
    ax.set_xlabel("")
    ax.set_ylim(-0.02, 1.06)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    ax.tick_params(axis='x', labelsize=13.5)
    ax.tick_params(axis='y', labelsize=13)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig_single, ax_s = plt.subplots(figsize=(6, 5))
    sns.boxplot(
        x='Condition', y='Probability', hue='Condition', data=df_plot,
        palette=['#7C9D8E', '#D4A373'], ax=ax_s, width=0.52, linewidth=1.6,
        fliersize=5.0, flierprops=dict(marker='o', markerfacecolor='white', markeredgecolor='#444444', markeredgewidth=1.1, alpha=0.8),
        legend=False
    )
    ax_s.text(0, 0.76, f"Mean: {base_mean:.4f}\nMedian: {base_median:.4f}", ha='center', va='center', fontsize=12.5, fontweight='medium', color='#1A3B2E',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F1EC', edgecolor='#7C9D8E', lw=1.1, alpha=0.9))
    ax_s.text(1, 0.76, f"Mean: {ko_mean:.4f}\nMedian: {ko_median:.4f}", ha='center', va='center', fontsize=12.5, fontweight='medium', color='#4A2C11',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='#F9F2EA', edgecolor='#D4A373', lw=1.1, alpha=0.9))
    
    ax_s.set_title(f"{target_gene}", fontsize=15.5, pad=12)
    ax_s.set_ylabel(f"Predicted Probability ({target_celltype})", fontsize=13.0, labelpad=8)
    ax_s.set_xlabel("")
    ax_s.set_ylim(-0.02, 1.06)
    ax_s.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_s.tick_params(axis='x', labelsize=13)
    ax_s.tick_params(axis='y', labelsize=13)
    ax_s.spines['top'].set_visible(False)
    ax_s.spines['right'].set_visible(False)
    
    fig_single.tight_layout()
    single_name = f"Figure6C_Single_Top{rank}_{target_gene}"
    fig_single.savefig(os.path.join(output_fig_dir, f"{single_name}.png"), dpi=DPI, bbox_inches='tight')
    fig_single.savefig(os.path.join(output_fig_dir, f"{single_name}.svg"), format='svg', bbox_inches='tight')
    plt.close(fig_single)

fig_all.tight_layout()
combined_name = "Figure6C_Combined_with_Metrics"
combined_png_path = os.path.join(output_fig_dir, f"{combined_name}.png")
combined_svg_path = os.path.join(output_fig_dir, f"{combined_name}.svg")
fig_all.savefig(combined_png_path, dpi=DPI, bbox_inches='tight')
fig_all.savefig(combined_svg_path, format='svg', bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print(f"saved [{output_fig_dir}]")
print(f"PNG: {combined_png_path}")
print(f"SVG: {combined_svg_path}")
print("=" * 70)
