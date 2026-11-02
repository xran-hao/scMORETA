import os
import gc
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Reuse the verified data loader logic from the preprocessing script
from preprocess_all_datasets import load_data


def get_sparsity(matrix):
    """Memory-safe sparsity calculation function for both sparse and dense matrices."""
    total_elements = float(matrix.shape[0]) * float(matrix.shape[1])
    if total_elements == 0:
        return 0.0
    if sp.issparse(matrix):
        nnz = matrix.nnz
    else:
        nnz = np.count_nonzero(matrix)
    return float(1.0 - (nnz / total_elements))


def compute_metrics_for_dataset(name, config):
    print("\n" + "=" * 65)
    print(f"📊 Computing dataset metrics: {name}")
    print("=" * 65)

    # 1. Load aligned and pre-cleaned modal data
    adata_rna, adata_atac = load_data(config)

    # 2. Extract basic profile metrics
    n_cells = adata_rna.shape[0]
    n_genes = adata_rna.shape[1]
    n_peaks = adata_atac.shape[1]

    # 3. Cell type distribution statistics
    cell_counts = adata_rna.obs['cell_type'].value_counts()
    n_cell_types = len(cell_counts)

    # 4. Proportion of major type
    prop_major_type = float(cell_counts.max() / n_cells)

    # 5. Imbalance degree based on normalized Shannon entropy
    if n_cell_types > 1:
        p_i = cell_counts.values / n_cells
        entropy = -np.sum(p_i * np.log(p_i))
        max_entropy = np.log(n_cell_types)
        imbalance_degree = float(1.0 - (entropy / max_entropy))
    else:
        imbalance_degree = 0.0

    # 6. Memory-safe sparsity computation
    print("   ⏳ Calculating RNA and ATAC sparsity...")
    rna_sparsity = get_sparsity(adata_rna.X)
    atac_sparsity = get_sparsity(adata_atac.X)

    # 7. Print formatted terminal output
    print("-" * 50)
    print(f"{'No. of cells':<26}: {n_cells}")
    print(f"{'No. of genes':<26}: {n_genes}")
    print(f"{'No. of peaks':<26}: {n_peaks}")
    print(f"{'No. of cell types':<26}: {n_cell_types}")
    print(f"{'Proportion of major type':<26}: {prop_major_type:.3f}")
    print(f"{'Imbalance degree':<26}: {imbalance_degree:.3f}")
    print(f"{'RNA sparsity':<26}: {rna_sparsity:.3f}")
    print(f"{'ATAC sparsity':<26}: {atac_sparsity:.3f}")
    print("-" * 50)

    # Garbage collection
    del adata_rna, adata_atac
    gc.collect()

    return {
        'Dataset': name,
        'No. of cells': n_cells,
        'No. of genes': n_genes,
        'No. of peaks': n_peaks,
        'No. of cell types': n_cell_types,
        'Proportion of major type': round(prop_major_type, 3),
        'Imbalance degree': round(imbalance_degree, 3),
        'RNA sparsity': round(rna_sparsity, 3),
        'ATAC sparsity': round(atac_sparsity, 3)
    }


def main():
    datasets_config = {
        "PBMC_10k": {
            "type": "dual_h5ad",
            "data_dir": "../../data/PBMC_10k",
            "file_rna": "10x-Multiome-Pbmc10k-RNA.h5ad",
            "file_atac": "10x-Multiome-Pbmc10k-ATAC.h5ad",
        },
        "SNARE_seq_Cortex": {
            "type": "dual_h5ad",
            "data_dir": "../../data/SNARE_seq_Cortex",
            "file_rna": "Chen-2019-RNA.h5ad",
            "file_atac": "Chen-2019-ATAC.h5ad",
        },
        "SHARE_seq_SkinA": {
            "type": "dual_h5ad",
            "data_dir": "../../data/SHARE_seq_SkinA",
            "file_rna": "Ma-2020-RNA.h5ad",
            "file_atac": "Ma-2020-ATAC.h5ad",
        },
        "SHARE_seq_Brain": {
            "type": "share_seq_raw",
            "data_dir": "../../data/SHARE_seq_Brain",
            "file_rna": "GSM4156610_brain.rna.counts.txt.gz",
            "file_atac_mtx": "GSM4156599_brain.counts.txt.gz",
            "file_atac_barcodes": "GSM4156599_brain.barcodes.txt.gz",
            "file_atac_peaks": "GSM4156599_brain.peaks.bed.gz",
            "file_meta": "GSM4156599_brain_celltype.txt.gz",
        },
        "SHARE_seq_SkinB": {
            "type": "share_seq_raw",
            "data_dir": "../../data/SHARE_seq_SkinB",
            "file_rna": "GSM4156608_skin.late.anagen.rna.counts.txt.gz",
            "file_atac_mtx": "GSM4156597_skin.late.anagen.counts.txt.gz",
            "file_atac_barcodes": "GSM4156597_skin.late.anagen.barcodes.txt.gz",
            "file_atac_peaks": "GSM4156597_skin.late.anagen.peaks.bed.gz",
            "file_meta": "GSM4156597_skin_celltype.txt.gz",
        },
        "BMMC": {
            "type": "single_h5ad",
            "data_dir": "../../data/BMMC",
            "input_file": "GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad",
        },
        "Human_Fetal_Brain": {
            "type": "human_fetal_brain_raw",
            "data_dir": "../../data/Human_Fetal_Brain",
            "file_rna": "GSE162170_multiome_rna_counts.tsv.gz",
            "file_atac": "GSE162170_multiome_atac_counts.tsv.gz",
            "file_peaks": "GSE162170_multiome_atac_consensus_peaks.txt.gz",
            "file_meta": "GSE162170_multiome_cell_metadata.txt.gz",
            "file_map": "GSE162170_multiome_cluster_names.txt.gz",
        },
        "SNARE_seq_AdBC": {
            "type": "snare_seq_mtx",
            "data_dir": "../../data/SNARE_seq_AdBC",
            "file_rna_mtx": "GSE126074_AdBrainCortex_SNAREseq_cDNA.counts.mtx.gz",
            "file_rna_barcodes": "GSE126074_AdBrainCortex_SNAREseq_cDNA.barcodes.tsv.gz",
            "file_rna_genes": "GSE126074_AdBrainCortex_SNAREseq_cDNA.genes.tsv.gz",
            "file_atac_mtx": "GSE126074_AdBrainCortex_SNAREseq_chromatin.counts.mtx.gz",
            "file_atac_barcodes": "GSE126074_AdBrainCortex_SNAREseq_chromatin.barcodes.tsv.gz",
            "file_atac_peaks": "GSE126074_AdBrainCortex_SNAREseq_chromatin.peaks.tsv.gz",
            "file_meta": "GSE126074_AdBrainCortex_metadata.txt",
        },
        "SNARE_seq_P0BC": {
            "type": "snare_seq_mtx",
            "data_dir": "../../data/SNARE_seq_P0BC",
            "file_rna_mtx": "GSE126074_P0_BrainCortex_SNAREseq_cDNA.counts.mtx.gz",
            "file_rna_barcodes": "GSE126074_P0_BrainCortex_SNAREseq_cDNA.barcodes.tsv.gz",
            "file_rna_genes": "GSE126074_P0_BrainCortex_SNAREseq_cDNA.genes.tsv.gz",
            "file_atac_mtx": "GSE126074_P0_BrainCortex_SNAREseq_chromatin.counts.mtx.gz",
            "file_atac_barcodes": "GSE126074_P0_BrainCortex_SNAREseq_chromatin.barcodes.tsv.gz",
            "file_atac_peaks": "GSE126074_P0_BrainCortex_SNAREseq_chromatin.peaks.tsv.gz",
            "file_meta": "GSE126074_P0_BrainCortex_metadata.txt",
        },
        "MouseKidney_sciCAR": {
            "type": "scicar_raw",
            "data_dir": "../../data/MouseKidney_sciCAR",
            "file_rna_count": "GSM3271044_RNA_mouse_kidney_gene_count.txt.gz",
            "file_rna_cell": "GSM3271044_RNA_mouse_kidney_cell.txt.gz",
            "file_rna_gene": "GSM3271044_RNA_mouse_kidney_gene.txt.gz",
            "file_atac_count": "GSM3271045_ATAC_mouse_kidney_peak_count.txt.gz",
            "file_atac_cell": "GSM3271045_ATAC_mouse_kidney_cell.txt.gz",
            "file_atac_peak": "GSM3271045_ATAC_mouse_kidney_peak.txt.gz",
        }
    }

    parser = argparse.ArgumentParser(description="Compute single-cell multi-omics dataset statistics")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all"] + list(datasets_config.keys()),
                        help="Specify dataset name, default 'all'")
    parser.add_argument("--save_path", type=str, default="dataset_statistics_summary.csv",
                        help="Path to save summary CSV report")
    args = parser.parse_args()

    metrics_list = []
    if args.dataset == "all":
        for name, cfg in datasets_config.items():
            metrics = compute_metrics_for_dataset(name, cfg)
            metrics_list.append(metrics)
    else:
        metrics = compute_metrics_for_dataset(args.dataset, datasets_config[args.dataset])
        metrics_list.append(metrics)

    df_summary = pd.DataFrame(metrics_list)
    df_summary.to_csv(args.save_path, index=False)
    print("\n" + "=" * 80)
    print("📋 [Dataset Statistics Summary Table]:")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)
    print(f"Summary report successfully saved to: {os.path.abspath(args.save_path)}\n")


if __name__ == "__main__":
    main()