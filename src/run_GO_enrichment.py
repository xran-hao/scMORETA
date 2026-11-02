"""
Script Function: Extracts gene modules from the hypergraph incidence matrix, performs GO Biological Process functional enrichment analysis, and exports summary tables and high-resolution bubble plots.
"""
import os
import textwrap
import gseapy as gp
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

plot_dir = "Figures_output"
os.makedirs(plot_dir, exist_ok=True)
data_dir = "../dataset/pre_data/PBMC_10k_Processed"
model_dir = "../result/PBMC_10k_Processed/models"
dataset_name = "PBMC_10k_Processed"
fold = 1
# Load the filtered gene feature names and the global hypergraph incidence matrix H saved during training
gene_file = os.path.join(data_dir, "gene_names.csv")
gene_names = np.loadtxt(gene_file, dtype=str)
print(f" Loaded gene list: {len(gene_names)}")

adj_file = os.path.join(model_dir, f"global_adj_{dataset_name}_full_{fold}.pt")

try:
    global_adj = torch.load(adj_file, weights_only=False).cpu().numpy()
except Exception as e:
    print(f"Load failed: {e}")
    exit(1)

# Traverse each hyperedge (module) and filter valid functional modules containing at least 10 genes
n_genes, n_modules = global_adj.shape
modules = {}

for i in range(n_modules):
    gene_indices = np.where(global_adj[:, i] > 0)[0]

    if len(gene_indices) >= 10:
        genes = gene_names[gene_indices].tolist()
        modules[f"Module_{i}"] = genes

print(
    f"Extraction completed! Total {len(modules)} valid modules (>=10 genes)."
)

for k, v in list(modules.items())[:3]:
    print(f"{k} ({len(v)} genes): {v[:5]} ...")
# Perform GO_Biological_Process_2021 enrichment analysis on modules sorted by gene count using GSEApy
gene_set_name = "GO_Biological_Process_2021"
results = []

print(f"Starting GO enrichment analysis (Database: {gene_set_name})...")
top_modules = sorted(
    modules.keys(), key=lambda k: len(modules[k]), reverse=True
)

print(f"Analyzing following modules: {top_modules}")
for mod_name in top_modules:
    genes = modules[mod_name]
    print(f" -> Analyzing {mod_name} ({len(genes)} genes)...")

    try:
        enr = gp.enrichr(
            gene_list=genes,
            gene_sets=gene_set_name,
            background=None,
            # organism='Mouse', # uncomment if using mouse dataset
            outdir=None,
            cutoff=0.05,
        )
        # Sort by Adjusted P-value ascendingly, retrieve the top leading term and collect all enriched terms
        df_res = enr.results
        if not df_res.empty:
            df_res = df_res.sort_values(by="Adjusted P-value", ascending=True)
            top_term = df_res.iloc[0]
            results.append(
                {
                    "Module": mod_name,
                    "Term": top_term["Term"],
                    "P-value": top_term["P-value"],
                    "Adjusted P-value": top_term["Adjusted P-value"],
                    "Genes": len(genes),
                    "Gene_List": genes,
                    "Overlap_Genes": top_term["Genes"],
                }
            )
            print(f"Top Term: {top_term['Term']}")
        else:
            print("No significant enrichment results found.")

    except Exception as e:
        print(f" Error: {e}")
# Compute enrichment score -log10(Adjusted P-value), remove redundant terms, and select the Top 10 leading functional terms
df_enrichment = pd.DataFrame(results)

if not df_enrichment.empty:
    df_enrichment["Score"] = -np.log10(
        df_enrichment["Adjusted P-value"] + 1e-300
    )

    unique_df = df_enrichment.sort_values(
        by="Score", ascending=False
    ).drop_duplicates(subset=["Term"])
    top10_df = unique_df.head(10).copy()

    top10_df["Module_ID"] = top10_df["Module"].apply(
        lambda x: x.split("_")[-1]
    )

    print("\n" + "=" * 60)
    print("Top 10 Enriched Terms and Gene Statistics:")
    all_top10_genes_union = set()

    for idx, row in top10_df.iterrows():
        mod = row["Module"]
        term = row["Term"]
        mod_gene_count = row["Genes"]
        overlap_str = row["Overlap_Genes"]
        overlap_count = (
            len(overlap_str.split(";")) if pd.notna(overlap_str) else 0
        )

        all_top10_genes_union.update(row["Gene_List"])

        print(f"\nTerm: {term}")
        print(f"  - Module: {mod}")
        print(f"  - Total Module Genes: {mod_gene_count}")
        print(f"  - Overlap Genes Count: {overlap_count}")
        print(f"  - Overlap Genes: {overlap_str}")

    print("\n" + "-" * 60)
    print(
        f"Total unique genes across Top 10 modules: {len(all_top10_genes_union)}"
    )
    print("=" * 60 + "\n")

    csv_rows = []
    for idx, row in top10_df.iterrows():
        overlap_str = (
            row["Overlap_Genes"] if pd.notna(row["Overlap_Genes"]) else ""
        )
        overlap_cnt = (
            len(overlap_str.split(";")) if len(overlap_str) > 0 else 0
        )

        csv_rows.append(
            {
                "Module_ID": row["Module_ID"],
                "Module_Name": row["Module"],
                "Enriched_Term": row["Term"],
                "Score_neg_log10_adj_p": round(row["Score"], 4),
                "Total_Gene_Count": row["Genes"],
                "All_Genes": ", ".join(row["Gene_List"]),
                "Overlap_Gene_Count": overlap_cnt,
                "Overlap_Genes": overlap_str,
            }
        )

    df_top10_export = pd.DataFrame(csv_rows)
    csv_output_path = os.path.join(
        plot_dir, "Top10_Modules_Genes_Summary.csv"
    )
    df_top10_export.to_csv(csv_output_path, index=False)
    print(f"Summary table successfully saved to: {csv_output_path}\n")

    def format_go_term_with_id(t):
        if "(GO:" in t:
            parts = t.split("(GO:")
            name = textwrap.fill(parts[0], width=55)
            go_id = f"(GO:{parts[1]}"
            return f"{name}\n{go_id}"
        return textwrap.fill(t, width=55)

    top10_df["Formatted_Term"] = top10_df["Term"].apply(
        format_go_term_with_id
    )

    fig, ax = plt.subplots(figsize=(13.5, 7.5))

    cmap = plt.cm.viridis
    norm = mcolors.Normalize(
        vmin=top10_df["Genes"].min(), vmax=top10_df["Genes"].max()
    )
    # Render bubble plot: X-axis is score, Y-axis is GO term, bubble size/color maps gene count, labeled with Module ID
    scatter = ax.scatter(
        x=top10_df["Score"],
        y=top10_df["Formatted_Term"],
        c=top10_df["Genes"],
        cmap=cmap,
        norm=norm,
        s=650,
        alpha=0.95,
        edgecolors="none",
        zorder=3,
    )

    for _, row in top10_df.iterrows():
        bg_rgba = cmap(norm(row["Genes"]))
        luminance = (
            0.299 * bg_rgba[0] + 0.587 * bg_rgba[1] + 0.114 * bg_rgba[2]
        )
        text_color = "black" if luminance > 0.55 else "white"

        ax.text(
            x=row["Score"],
            y=row["Formatted_Term"],
            s=row["Module_ID"],
            color=text_color,
            fontsize=10,
            weight="bold",
            ha="center",
            va="center",
            zorder=4,
        )

    ax.invert_yaxis()

    # Fully continuous colorbar with smooth rasterization
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=ax,
        pad=0.03,
        aspect=25,
        shrink=0.75,
        drawedges=False,
    )
    cbar.solids.set_rasterized(True)
    cbar.set_label("Gene Count", fontsize=12.5, labelpad=8)
    cbar.ax.tick_params(labelsize=10)

    plt.xlabel("-log10(Adjusted P-value)", fontsize=12.5)
    plt.ylabel("")
    plt.yticks(fontsize=11)

    x_min, x_max = top10_df["Score"].min(), top10_df["Score"].max()
    plt.xlim(x_min - 0.2, x_max + 0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, "Figure7C_Functional_Enrichment_Top10_Exact.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(plot_dir, "Figure7C_Functional_Enrichment_Top10_Exact.svg"),
        dpi=300,
        bbox_inches="tight",
    )
    print(
        "Plots successfully saved as 'Functional_Enrichment_Top10_Exact.png' and '.svg'"
    )
    plt.show()
else:
    print("Enrichment dataframe is empty, skipping plot generation.")