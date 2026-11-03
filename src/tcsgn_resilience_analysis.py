"""
Script Function: Cellular Identity Resilience (ISS) analysis, including topological fingerprint heatmap, global resilience UMAP projection, lymphocyte differentiation dynamics violin plots, and monocyte fragility mechanistic dissection (DEGs, GO enrichment, volcano plot, and inflammatory stress regression).
"""
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from model import CSGNet
import textwrap

def seed_everything_strict(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
seed_everything_strict(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_path = "/home/horizon/scMORETA/dataset/pre_data/PBMC_10k_Processed"
DROP_RATES = [0.5]
SAVE_DIR = "Figures_output"
os.makedirs(SAVE_DIR, exist_ok=True) 

def save_dual_figures(fig_name):
    png_path = os.path.join(SAVE_DIR, f"{fig_name}.png")
    svg_path = os.path.join(SAVE_DIR, f"{fig_name}.svg")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, dpi=300, bbox_inches='tight')
    print(f"saved: {png_path} and {svg_path}")
    
class MyDataset(Dataset):
    def __init__(self, rna, labels):
        self.rna = torch.tensor(rna, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.rna[idx], self.labels[idx]

def load_label_map(path):
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.size > 1:
        return {i: name for i, name in enumerate(loaded)}
    elif hasattr(loaded, 'item'):
        return loaded.item()
    return loaded
# Extract single-cell dynamic hyperedge activation, average across lineages to plot topological regulatory fingerprint heatmap (Figure 7A)
def plot_topological_fingerprint(model, adata, dataset_name, device):
    model.eval()
    all_h_activation = []
    data_tensor = torch.tensor(adata.X, dtype=torch.float32)
    tmp_loader = DataLoader(data_tensor, batch_size=256, shuffle=False)
    with torch.no_grad():
        for batch_x in tmp_loader:
            _ = model(batch_x.to(device))
            h_act = model.current_dynamic_H.abs().mean(dim=1).cpu().numpy()
            all_h_activation.append(h_act)
    full_h_activation = np.vstack(all_h_activation)
    df_h = pd.DataFrame(full_h_activation)
    df_h['cell_type'] = adata.obs['cell_type'].values
    type_fingerprint = df_h.groupby('cell_type').mean()
    plt.figure(figsize=(16, 10))
    hm = sns.heatmap(
        type_fingerprint,
        cmap="magma",
        xticklabels=20,
        cbar_kws={
            "label": "Regulatory Module Activation",
            "shrink": 0.75,
            "aspect": 25,
            "pad": 0.02,
        },
    )

    cbar = hm.collections[0].colorbar
    cbar.ax.set_ylabel("Regulatory Module Activation", fontsize=14)
    cbar.ax.tick_params(labelsize=13)
    cbar.solids.set_rasterized(True)
    cbar.solids.set_edgecolor("face")

    plt.xlabel("Hyperedge Index (Regulatory Modules)", fontsize=15)
    plt.xticks(fontsize=14) 
    plt.ylabel("Cell Lineages", fontsize=15)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    save_dual_figures(f"Figure7A_topological_fingerprint_{dataset_name}")

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

def run_global_analysis():
    all_labels = np.load(os.path.join(base_path, 'Labels.npy'))
    n_total = len(all_labels)
    label_map = load_label_map(os.path.join(base_path, 'label_map.npy'))
    global_iss_at_50 = np.zeros(n_total)
    global_predictions_clean = np.zeros(n_total, dtype=int)
    
    for k in range(1, 6):
        fold_dir = os.path.join(base_path, f"fold_{k}")
        cv_indices = np.load(os.path.join(base_path, 'cv_indices.npz'), allow_pickle=True)
        test_idx = cv_indices[f'test_index_{k}']
        test_rna = np.load(os.path.join(fold_dir, 'test_rna_2k.npy'))
        test_labels = np.load(os.path.join(fold_dir, 'test_labels.npy'))
        model_path = f"../result/PBMC_10k_Processed/models/tcsgn_PBMC_10k_Processed_full_model{k}.pth"
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
    
        loader = DataLoader(MyDataset(test_rna, test_labels), batch_size=256, shuffle=False)
        all_orig_conf = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                probs = F.softmax(model(x), dim=1)
                all_orig_conf.extend(probs[range(len(y)), y].cpu().numpy())
            
            all_orig_conf = np.array(all_orig_conf)
            all_decay_conf = []
            # Inject 50% Bernoulli dropout noise, calculating ratio of perturbed to unperturbed confidence as stability score (ISS)
            for x, y in loader:
                x = x.to(device)
                mask = torch.bernoulli(torch.full(x.shape, 0.5)).to(device)
                out = model(x * mask)
                probs = F.softmax(out, dim=1)
                all_decay_conf.extend(probs[range(len(y)), y].cpu().numpy())
            iss_scores = np.array(all_decay_conf) / (all_orig_conf + 1e-6)
            global_iss_at_50[test_idx] = iss_scores
            
    from scipy.stats import rankdata
    clean_iss = np.nan_to_num(global_iss_at_50, nan=0.0, posinf=0.0, neginf=0.0)
    clean_iss = np.clip(clean_iss, 0, 1)
    if clean_iss.max() > 0:
        iss_percentile = rankdata(clean_iss) / len(clean_iss)
    else:
        iss_percentile = clean_iss
    full_rna_path = os.path.join(base_path, 'RNA_matrix.npy')
    if os.path.exists(full_rna_path):
        full_rna = np.load(full_rna_path)
    else:
        raise FileNotFoundError(f"find error: {full_rna_path}")

    adata = sc.AnnData(X=full_rna.astype(np.float32))
    adata.obs['cell_type'] = [label_map[l] for l in all_labels]
    adata.obs['ISS_Percentile'] = iss_percentile 
    sc.pp.neighbors(adata, n_neighbors=15)
    sc.tl.umap(adata)
    full_rna = np.load(os.path.join(base_path, 'RNA_matrix.npy'))
    adata = sc.AnnData(X=full_rna.astype(np.float32))
    adata.obs['cell_type'] = [label_map[l] for l in all_labels]
    
    from scipy.stats import rankdata
    clean_iss = np.nan_to_num(global_iss_at_50, nan=0.0, posinf=0.0, neginf=0.0)
    clean_iss = np.clip(clean_iss, 0, 1)
    if clean_iss.max() > 0:
        iss_rank = rankdata(clean_iss) / len(clean_iss)
    else:
        iss_rank = clean_iss
    adata.obs['ISS_Stability_Rank'] = iss_rank
    print(f"mean ISS: {clean_iss.mean():.4f}")
    
    sc.pp.neighbors(adata, n_neighbors=15)
    sc.tl.umap(adata)
    plt.close('all') 
    # Dual-panel global UMAP: Left displays ground-truth cell types, right maps continuous ISS stability ranks (Figure 6E)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    sc.pl.umap(adata, color='cell_type', ax=ax1, show=False, title="")
    sc.pl.umap(adata, color='ISS_Stability_Rank', ax=ax2, show=False, 
            title="", 
            cmap='RdYlGn', vmin=0, vmax=1, colorbar_loc=None)
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(norm=norm, cmap='RdYlGn')
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, pad=0.03, aspect=25, shrink=0.75, drawedges=False)
    cbar.solids.set_rasterized(True)
    cbar.solids.set_edgecolor("face")
    plt.tight_layout()
    save_dual_figures("Figure6E_global_resilience_umap_finalPBMC")
    plt.show()
    # Aggregate and sort mean resilience scores across cell lineages to plot ranking bar chart (Figure 6D)
    type_resilience = adata.obs.groupby('cell_type')['ISS_Stability_Rank'].mean().sort_values()
    plt.figure(figsize=(10, 8))
    colors = plt.cm.RdYlGn(type_resilience.values)
    ax_bar = type_resilience.plot(kind='barh', color=colors, edgecolor='gray', alpha=0.8)
    
    for i, v in enumerate(type_resilience.values):
        ax_bar.text(v + 0.01, i, f"{v:.3f}", color='black', va='center', fontsize=14)
    #plt.title("Average Cellular Identity Resilience by Type\n(PBMC_10k)", fontsize=14)
    plt.xlabel("Mean IRS Stability Rank (Higher is more Resilient)", fontsize=14)
    plt.ylabel("Cell Type", fontsize=14)
    plt.yticks(fontsize=12) 
    plt.xticks(fontsize=12) 
    plt.xlim(0, 1.1)
    
    plt.tight_layout()
    save_dual_figures("Figure6D_resilience_by_cell_type_rankingPBMC")
    print("saved (resilience_by_cell_type_rankingPBMC.png)")
    
    from scipy.stats import mannwhitneyu
    cd4_lineage = ['CD4 Naive', 'CD4 TCM', 'CD4 TEM']
    b_lineage   = ['Naive B', 'Memory B', 'Plasma']
    sub_cd4 = adata.obs[adata.obs['cell_type'].isin(cd4_lineage)].copy()
    sub_b   = adata.obs[adata.obs['cell_type'].isin(b_lineage)].copy()
    palette_cd4 = ['#7C9D8E', '#68889E', '#D4A373'] 
    palette_b   = ['#8FA4B0', '#7C9D8E', '#B85B56'] 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.8), sharey=True)
    
    def get_p_text(p):
        if p < 0.001: return f"p = {p:.1e} (***)"
        elif p < 0.01: return f"p = {p:.2e} (**)"
        elif p < 0.05: return f"p = {p:.2e} (*)"
        else: return f"p = {p:.2f} (n.s.)"
    sns.violinplot(x='cell_type', y='ISS_Stability_Rank', data=sub_cd4, 
                   order=cd4_lineage, palette=palette_cd4, ax=ax1, 
                   
                   inner='quartile', cut=2, linewidth=0.8)
    for i, ct in enumerate(cd4_lineage):
        m_val = sub_cd4[sub_cd4['cell_type'] == ct]['ISS_Stability_Rank'].mean()
        ax1.text(i, m_val + 0.03, f"μ={m_val:.3f}", ha='center', va='bottom', 
                 fontsize=12.5, color='#1a1a1a')
    stat1, p1 = mannwhitneyu(sub_cd4[sub_cd4['cell_type'] == 'CD4 TCM']['ISS_Stability_Rank'],
                             sub_cd4[sub_cd4['cell_type'] == 'CD4 TEM']['ISS_Stability_Rank'])
    ax1.plot([1, 1, 2, 2], [0.91, 0.94, 0.94, 0.91], lw=1.0, c='k')
    ax1.text(1.5, 0.95, get_p_text(p1), ha='center', va='bottom', fontsize=12)
    #ax1.set_title("CD4+ T Cell Lineage Dynamics", fontsize=12,pad=12)
    ax1.set_xlabel("")
    ax1.set_ylabel("IRS Stability Rank", fontsize=13)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    sns.violinplot(x='cell_type', y='ISS_Stability_Rank', data=sub_b, 
                   order=b_lineage, palette=palette_b, ax=ax2, 
                   inner='quartile', cut=2, linewidth=0.8)    
    
    for i, ct in enumerate(b_lineage):
        m_val = sub_b[sub_b['cell_type'] == ct]['ISS_Stability_Rank'].mean()
        ax2.text(i, m_val + 0.03, f"μ={m_val:.3f}", ha='center', va='bottom', 
                 fontsize=12.5, color='#1a1a1a')

    stat2, p2 = mannwhitneyu(sub_b[sub_b['cell_type'] == 'Naive B']['ISS_Stability_Rank'],
                             sub_b[sub_b['cell_type'] == 'Memory B']['ISS_Stability_Rank'])
    ax2.plot([0, 0, 1, 1], [0.91, 0.94, 0.94, 0.91], lw=1.0, c='k')
    ax2.text(0.5, 0.95, get_p_text(p2), ha='center', va='bottom', fontsize=12)
    #ax2.set_title("B Cell Lineage Dynamics", fontsize=12, pad=12)
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.tick_params(axis="x", labelsize=11) 
        ax1.tick_params(axis="y", labelsize=11) 
        
    #plt.suptitle("Progressive Identity Fragility Along Lymphocyte Effector Differentiation", 
                 #fontsize=13.5, y=0.99)
    plt.tight_layout()
    save_dual_figures("Figure6F_Lymphocyte_Differentiation_Resilience_Dynamics")
    plt.show()
    print("saved: Lymphocyte_Differentiation_Resilience_Dynamics.png 和 .pdf")
    plot_topological_fingerprint(model, adata, "PBMC_10k", device)
    
    import gseapy as gp
    TARGET_CELL = "CD14 Mono"
    gene_file_path = os.path.join(base_path, 'gene_names.csv')
    if os.path.exists(gene_file_path):
        real_genes = np.loadtxt(gene_file_path, dtype=str)
        adata.var_names = real_genes
        adata.var_names_make_unique()
    # Stratify CD14 Monocytes into Fragile (Bottom 30%) vs. Resilient (Top 30%) subsets to identify DEGs via Wilcoxon test
    adata_sub = adata[adata.obs['cell_type'] == TARGET_CELL].copy()
    q_low = adata_sub.obs['ISS_Stability_Rank'].quantile(0.30)
    q_high = adata_sub.obs['ISS_Stability_Rank'].quantile(0.70)
    
    adata_sub.obs['Resilience_State'] = 'Intermediate'
    adata_sub.obs.loc[adata_sub.obs['ISS_Stability_Rank'] <= q_low, 'Resilience_State'] = 'Fragile (Bottom 30%)'
    adata_sub.obs.loc[adata_sub.obs['ISS_Stability_Rank'] >= q_high, 'Resilience_State'] = 'Resilient (Top 30%)'
    
    adata_contrast = adata_sub[adata_sub.obs['Resilience_State'].isin(['Fragile (Bottom 30%)', 'Resilient (Top 30%)'])].copy()
    sc.tl.rank_genes_groups(adata_contrast, groupby='Resilience_State', 
                            groups=['Fragile (Bottom 30%)'], reference='Resilient (Top 30%)', 
                            method='wilcoxon')
                            
    deg_df = sc.get.rank_genes_groups_df(adata_contrast, group='Fragile (Bottom 30%)')
    top_fragile_genes = deg_df.sort_values('scores', ascending=False).head(35)['names'].tolist()
    top_resilient_genes = deg_df.sort_values('scores', ascending=True).head(35)['names'].tolist()
    print(f" top_fragile_genes Top 5: {top_fragile_genes[:5]}")
    print(f" top_resilient_genes Top 5: {top_resilient_genes[:5]}")
    
    try:
        enr = gp.enrichr(
            gene_list=top_fragile_genes,
            gene_sets='GO_Biological_Process_2021',
            cutoff=0.05
        )
        if enr.results is not None and not enr.results.empty:
            top_terms = enr.results.sort_values('Adjusted P-value').head(7).copy()
            top_terms['-log10(Adj.P)'] = -np.log10(top_terms['Adjusted P-value'])
        else:
            top_terms = pd.DataFrame()
        plt.rcParams['font.family'] = 'DejaVu Sans'
        # Examine progressive loss of cellular resilience along CD4+ T and B cell effector differentiation trajectories (Figure 6F)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        plot_degs = deg_df.sort_values('scores', ascending=False).head(6).copy()
        plot_degs_res = deg_df.sort_values('scores', ascending=True).head(6).copy()
        combined_degs = pd.concat([plot_degs, plot_degs_res])
        
        deg_colors = ['#B85B56' if s > 0 else '#7C9D8E' for s in combined_degs['scores']]
        ax1.barh(range(len(combined_degs)), combined_degs['scores'], color=deg_colors, edgecolor='#2b2b2b', linewidth=0.6, alpha=0.88)
        # 在 ax1 每根柱子外侧标注 Z-score 具体数值
        for i, val in enumerate(combined_degs['scores']):
            offset = 0.2 if val >= 0 else -0.2
            ha_align = 'left' if val >= 0 else 'right'
            ax1.text(val + offset, i, f"{val:.2f}", va='center', ha=ha_align, 
                    fontsize=13, color='black', weight='normal')
        ax1.set_xlim(combined_degs['scores'].min() * 1.25, combined_degs['scores'].max() * 1.25)
        ax1.set_yticks(range(len(combined_degs)))
        ax1.set_yticklabels(combined_degs['names'], fontsize=14, )
        ax1.invert_yaxis()
        ax1.axvline(0, color='black', lw=0.8, linestyle='--')
        #ax1.set_title("Key Discriminative Genes (Fragile vs. Resilient)", fontsize=12, pad=12)
        ax1.set_xlabel("Z-score (Enriched in Fragile > 0 | Enriched in Resilient < 0)", fontsize=14)
        if not top_terms.empty:
            colors = plt.cm.YlOrRd(np.linspace(0.4, 0.85, len(top_terms)))
            bars = ax2.barh(range(len(top_terms)), top_terms['-log10(Adj.P)'], 
                            color=colors, edgecolor='#2b2b2b', linewidth=0.6, alpha=0.85)

            for i, val in enumerate(top_terms['-log10(Adj.P)']):
                ax2.text(val + 0.15, i, f"{val:.2f}", va='center', ha='left', 
                        fontsize=13, color='black', weight='normal')
            ax2.set_xlim(0, top_terms['-log10(Adj.P)'].max() * 1.18)
            formatted_terms = []
            for t in top_terms['Term']:
                if ' (GO:' in t:
                    parts = t.split(' (GO:')
                    name_part = textwrap.fill(parts[0].strip(), width=42)
                    go_id = f"(GO:{parts[1].strip()}"
                    formatted_terms.append(f"{name_part}\n{go_id}")
                else:
                    formatted_terms.append(textwrap.fill(t.strip(), width=45))

            ax2.set_yticks(range(len(top_terms)))
            ax2.set_yticklabels(formatted_terms, fontsize=13)
            #ax2.set_title("Functional Vulnerability Pathways (Fragile Cells)", 
                          #fontsize=12, pad=12)
            ax2.set_xlabel("-log10(Adjusted P-value)", fontsize=14)
            ax2.invert_yaxis()
        else:
            ax2.text(0.5, 0.5, "No significantly enriched GO terms found", ha='center', va='center')
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.8)
            ax.spines['bottom'].set_linewidth(0.8)
        #plt.suptitle(f"Molecular Drivers of Identity Destabilization within {TARGET_CELL}", 
                     #fontsize=13.5, y=0.98)
        plt.tight_layout()
        save_dual_figures("Figure8A_Fragility_Ridge_Mechanistic_Analysis")
        plt.show()
        print("saved: Fragility_Ridge_Mechanistic_Analysis.png and .pdf")
    except Exception as e:
        print(f"error: {e}")
        
    from scipy.stats import pearsonr
    adata_mono = adata[adata.obs['cell_type'] == TARGET_CELL].copy()
    deg_full = sc.get.rank_genes_groups_df(adata_contrast, group='Fragile (Bottom 30%)')
    deg_full['neg_log10_pval'] = -np.log10(deg_full['pvals_adj'] + 1e-300)
    # Dual-panel mechanistic plot: Left displays key driver gene Z-scores, right displays enriched vulnerability pathways (Figure 8B)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.2))
    logfc_thresh = 0.3
    pval_thresh = -np.log10(0.01)
    deg_full['color'] = 'Not Sig'
    deg_full.loc[(deg_full['logfoldchanges'] > logfc_thresh) & (deg_full['neg_log10_pval'] > pval_thresh), 'color'] = 'Enriched in Fragile'
    deg_full.loc[(deg_full['logfoldchanges'] < -logfc_thresh) & (deg_full['neg_log10_pval'] > pval_thresh), 'color'] = 'Enriched in Resilient'
    color_dict = {'Not Sig': '#e2e2e2', 'Enriched in Fragile': '#B85B56', 'Enriched in Resilient': '#7C9D8E'}
    
    for c_name, c_color in color_dict.items():
        sub = deg_full[deg_full['color'] == c_name]
        ax1.scatter(sub['logfoldchanges'], sub['neg_log10_pval'], c=c_color, s=22, alpha=0.75, label=c_name)
    ax1.set_xlim(-3.0, 3.0)
    ax1.set_ylim(-0.8, 19.5)
    
    plot_degs_fragile = deg_full.sort_values('scores', ascending=False).head(6)
    plot_degs_resilient = deg_full.sort_values('scores', ascending=True).head(6)
    
    red_sorted = plot_degs_fragile.sort_values('neg_log10_pval', ascending=False)
    green_sorted = plot_degs_resilient.sort_values('neg_log10_pval', ascending=False)
    left_y_slots = np.linspace(17.5, 6.0, 6)
    for i, (_, row) in enumerate(green_sorted.iterrows()):
        gene = row['names']
        px, py = row['logfoldchanges'], row['neg_log10_pval']
        ty = left_y_slots[i]
        ax1.annotate(
            gene, xy=(px, py), xytext=(-2.1, ty), textcoords='data',
            fontsize=11.5, color='#111111', ha='right', va='center',
            arrowprops=dict(arrowstyle='->', color='#555555', lw=0.8)
        )
    right_y_slots = np.linspace(17.5, 6.0, 6)
    for i, (_, row) in enumerate(red_sorted.iterrows()):
        gene = row['names']
        px, py = row['logfoldchanges'], row['neg_log10_pval']
        ty = right_y_slots[i]
        ax1.annotate(
            gene, xy=(px, py), xytext=(2.0, ty), textcoords='data',
            fontsize=11.5, color='#111111', ha='left', va='center',
            arrowprops=dict(arrowstyle='->', color='#555555', lw=0.8)
        )
    ax1.axvline(0, color='grey', linestyle='--', lw=0.8)
    ax1.axhline(pval_thresh, color='grey', linestyle='--', lw=0.8)
    #ax1.set_title("Transcriptomic Divergence (Volcano Plot)", fontsize=12, pad=36)
    ax1.set_xlabel("log2(Fold Change: Fragile / Resilient)", fontsize=14)
    ax1.set_ylabel("-log10(Adjusted P-value)", fontsize=14)
    ax1.legend(
        loc='lower left', 
        bbox_to_anchor=(0.0, 1.02), 
        ncol=3, 
        frameon=False, 
        fontsize=13,
        handletextpad=0.3,
        columnspacing=1.2
    )
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ifn_gene_set = ['IFI30', 'HLA-DRA', 'HLA-DPB1', 'HLA-DPA1', 'AIF1', 'STAT1', 'IRF1', 'B2M']
    valid_ifn_genes = [g for g in ifn_gene_set if g in adata_mono.var_names]
    sc.tl.score_genes(adata_mono, gene_list=valid_ifn_genes, score_name='IFN_gamma_Score')
    x_vals = adata_mono.obs['IFN_gamma_Score'].values
    y_vals = adata_mono.obs['ISS_Stability_Rank'].values
    ax2.scatter(x_vals, y_vals, color='#7A8B99', s=12, alpha=0.18, rasterized=True, label='Single Cells')
    sns.regplot(x=x_vals, y=y_vals, ax=ax2, scatter=False,
                color='#B85B56', line_kws={'linewidth': 2.6, 'zorder': 4})
    bins = np.linspace(-1.5, 2.0, 8)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_means, bin_sems = [], []
    for i in range(len(bins)-1):
        mask = (x_vals >= bins[i]) & (x_vals < bins[i+1])
        vals_in_bin = y_vals[mask]
        if len(vals_in_bin) > 0:
            bin_means.append(vals_in_bin.mean())
            bin_sems.append(vals_in_bin.std() / np.sqrt(len(vals_in_bin)))
        else:
            bin_means.append(np.nan)
            bin_sems.append(np.nan)
    ax2.errorbar(
        bin_centers, bin_means, yerr=bin_sems, fmt='o',
        markerfacecolor='white', markeredgecolor='#B85B56', markeredgewidth=2.0,
        ecolor='#B85B56', elinewidth=1.6, capsize=3.5, capthick=1.2,
        markersize=7.5, zorder=6, label=None)
    r_val, p_val = pearsonr(x_vals, y_vals)
    
    ax2.text(0.05, 0.08, f"Pearson R = {r_val:.3f}\nP = {p_val:.2e} (***)", 
             transform=ax2.transAxes, fontsize=14,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', edgecolor='#cccccc'))
    #ax2.set_title("Cellular Resilience vs. Inflammatory Stress", fontsize=12, pad=36)
    ax2.set_xlabel("Interferon-γ (IFN-γ) Activation Score", fontsize=14)
    ax2.set_ylabel("IRS Stability Rank", fontsize=14)
    ax2.set_ylim(0.0, 1.05)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(
        loc='lower right', 
        bbox_to_anchor=(1.0, 1.02), 
        ncol=1, 
        frameon=False, 
        fontsize=13,
        handletextpad=0.4,
        columnspacing=1.5
    ) 
    
    #plt.suptitle("Molecular Validation: Inflammatory Stress Progressively Destabilizes Monocyte Identity", 
                 #fontsize=13, y=0.99)
    plt.subplots_adjust(top=0.84, bottom=0.08, left=0.08, right=0.98)
    save_dual_figures("Figure8B_Mechanistic_DeepDive_Volcano_and_Regression")
    plt.show()
    
if __name__ == "__main__":
    run_global_analysis()
