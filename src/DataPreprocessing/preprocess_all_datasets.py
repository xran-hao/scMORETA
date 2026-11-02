import os
import gc
import gzip
import bisect
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io
import scanpy as sc
from tqdm import tqdm

# GTF and Peak binary search index parsing
def parse_gtf(gtf_path):
    print(f"Parsing GTF: {gtf_path} ...")
    gene_dict = {}
    with gzip.open(gtf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 9 or parts[2] != 'gene':
                continue
            info = parts[8]
            if 'gene_name "' not in info:
                continue
            gene_name = info.split('gene_name "')[1].split('"')[0]
            chrom = parts[0] if parts[0].startswith('chr') else 'chr' + parts[0]
            gene_dict[gene_name] = (chrom, int(parts[3]), int(parts[4]), parts[6])
    print(f"Parsed {len(gene_dict)} gene coordinates")
    return gene_dict


def parse_peaks(var_names):
    chrom_peaks_dict = {}
    for idx, p in enumerate(var_names):
        try:
            if ':' in p:
                chrom_raw, pos = p.split(':')
            elif '-' in p:
                parts = p.split('-')
                chrom_raw = parts[0]
                pos = f"{parts[1]}-{parts[2]}"
            else:
                continue

            chrom = chrom_raw if chrom_raw.startswith('chr') else 'chr' + chrom_raw
            start, end = map(int, pos.split('-'))
            center = (start + end) // 2
            chrom_peaks_dict.setdefault(chrom, []).append((center, idx))
        except Exception:
            continue

    peaks_by_chrom = {}
    for chrom, p_list in chrom_peaks_dict.items():
        p_list.sort(key=lambda x: x[0])
        centers = [x[0] for x in p_list]
        indices = [x[1] for x in p_list]
        peaks_by_chrom[chrom] = (centers, indices)
    return peaks_by_chrom

# Memory-safe saving of full dense NPY matrix
def safe_save_full_matrix(adata, out_path, chunk_size=4000):
    n_cells, n_genes = adata.shape
    print(f"Safely saving full RNA matrix via streaming ({n_cells} x {n_genes})...")
    with open(out_path, 'wb') as f:
        np.lib.format.write_array_header_2_0(
            f, {'descr': np.dtype(np.float32).descr[0][1], 'fortran_order': False, 'shape': (n_cells, n_genes)}
        )
        for i in range(0, n_cells, chunk_size):
            end = min(i + chunk_size, n_cells)
            chunk = adata.X[i:end]
            if sp.issparse(chunk):
                chunk = chunk.toarray()
            f.write(chunk.astype(np.float32).tobytes())
    print(f"Full RNA matrix safely written to disk: {out_path}")


# 读取 MTX/三元组 (SNARE-seq 与 sci-CAR)
def load_snare_part(mtx_file, barcode_file, feature_file, modality):
    print(f"Reading {modality} MTX triplet format")
    mat = scipy.io.mmread(mtx_file).tocsr()
    barcodes = pd.read_csv(barcode_file, header=None)[0].values.astype(str)
    features = pd.read_csv(feature_file, header=None)[0].values.astype(str)

    if mat.shape[0] == len(features) and mat.shape[1] == len(barcodes):
        mat = mat.T
    elif mat.shape[0] == len(barcodes) and mat.shape[1] == len(features):
        pass
    else:
        raise ValueError(f"Dimension mismatch! Matrix: {mat.shape}, BC: {len(barcodes)}, Feat: {len(features)}")

    adata = sc.AnnData(X=mat)
    adata.obs_names = barcodes
    adata.var_names = features
    adata.var_names_make_unique()
    return adata


def load_scicar_matrix(count_file, cell_file, feature_file, modality):
    print(f"Reading sci-CAR {modality} ...")
    try:
        mat = scipy.io.mmread(count_file).tocsr()
    except Exception:
        df = pd.read_csv(count_file, sep=r'\s+', comment='%')
        if df.shape[1] == 3 and df.iloc[0, 2] > 1000:
            df = df.iloc[1:]
        mat = sp.coo_matrix((df.iloc[:, 2], (df.iloc[:, 0] - 1, df.iloc[:, 1] - 1))).tocsr()

    df_cell = pd.read_csv(cell_file, sep=',', header=0)
    df_feat = pd.read_csv(feature_file, sep=',', header=0)

    # Check dimensions and transpose
    if mat.shape[0] == len(df_feat) and mat.shape[1] == len(df_cell):
        mat = mat.T
    elif mat.shape[0] == len(df_cell) and mat.shape[1] == len(df_feat):
        pass
    else:
        print(f"Dimensions do not completely match, performing truncation adaptation")
        mat = mat.T
        n_c, n_g = min(mat.shape[0], len(df_cell)), min(mat.shape[1], len(df_feat))
        mat = mat[:n_c, :n_g]
        df_cell = df_cell.iloc[:n_c]
        df_feat = df_feat.iloc[:n_g]

    adata = sc.AnnData(X=mat)
    # Extract cell barcodes
    adata.obs_names = df_cell['sample'].astype(str) if 'sample' in df_cell.columns else df_cell.iloc[:, 0].astype(str)

    # Extract feature names
    if modality == 'RNA':
        col = 'gene_short_name' if 'gene_short_name' in df_feat.columns else 'gene_id'
        adata.var_names = df_feat[col].astype(str)
        if 'cell_name' in df_cell.columns:
            adata.obs['cell_type'] = df_cell['cell_name'].values
    else:
        adata.var_names = df_feat['peak'].astype(str) if 'peak' in df_feat.columns else df_feat.iloc[:, 0].astype(str)

    adata.var_names_make_unique()
    return adata

def load_data(config):
    data_type = config.get("type", "dual_h5ad")
    data_dir = config["data_dir"]

    # Mode A: Dual h5ad files
    if data_type == "dual_h5ad":
        print(f"Loading Mode: Dual h5ad files")
        adata_rna = sc.read_h5ad(os.path.join(data_dir, config["file_rna"]))
        adata_atac = sc.read_h5ad(os.path.join(data_dir, config["file_atac"]))

        if not np.array_equal(adata_rna.obs_names, adata_atac.obs_names):
            common_cells = adata_rna.obs_names.intersection(adata_atac.obs_names)
            adata_rna = adata_rna[common_cells].copy()
            adata_atac = adata_atac[common_cells].copy()

    # Mode B: Single h5ad modality split (BMMC)
    elif data_type == "single_h5ad":
        print(f"Loading Mode: Single h5ad modality split")
        input_file = os.path.join(data_dir, config["input_file"])
        adata_full = sc.read_h5ad(input_file)
        adata_rna = adata_full[:, adata_full.var['feature_types'] == 'GEX'].copy()
        adata_atac = adata_full[:, adata_full.var['feature_types'] == 'ATAC'].copy()
        del adata_full
        gc.collect()

    # Mode C: SHARE-seq raw multi-files (SkinB, Brain)
    elif data_type == "share_seq_raw":
        print(f"Loading Mode: SHARE-seq raw multi-files (streaming low-memory")
        file_rna = os.path.join(data_dir, config["file_rna"])
        print(f"Streaming and parsing RNA expression file: {os.path.basename(file_rna)} ...")

        row_list, col_list, data_list = [], [], []
        gene_names = []

        with gzip.open(file_rna, 'rt') as f:
            header_line = f.readline().rstrip('\n\r').split('\t')
            cell_names = [bc.replace(',', '.') for bc in header_line[1:]]
            n_cols = len(cell_names)

            for row_idx, line in enumerate(tqdm(f, desc="Streaming RNA genes")):
                parts = line.rstrip('\n\r').split('\t')
                gene_names.append(parts[0])

                vals = np.fromiter((float(x) for x in parts[1:]), dtype=np.float32, count=n_cols)
                nonzero_cols = np.nonzero(vals)[0]

                if len(nonzero_cols) > 0:
                    row_list.append(np.full(len(nonzero_cols), row_idx, dtype=np.int32))
                    col_list.append(nonzero_cols.astype(np.int32))
                    data_list.append(vals[nonzero_cols])

        mat_genes_cells = sp.csr_matrix(
            (np.concatenate(data_list), (np.concatenate(row_list), np.concatenate(col_list))),
            shape=(len(gene_names), n_cols),
            dtype=np.float32
        )
        del row_list, col_list, data_list
        gc.collect()

        adata_rna = sc.AnnData(X=mat_genes_cells.T.tocsr())
        adata_rna.obs_names = cell_names
        adata_rna.var_names = gene_names
        del mat_genes_cells
        gc.collect()

        # Read ATAC
        mat = scipy.io.mmread(os.path.join(data_dir, config["file_atac_mtx"])).tocsr()
        barcodes = pd.read_csv(os.path.join(data_dir, config["file_atac_barcodes"]), header=None)[0].values
        peaks_df = pd.read_csv(os.path.join(data_dir, config["file_atac_peaks"]), sep=None, engine='python', header=None)
        peak_names = peaks_df[0].astype(str) + ":" + peaks_df[1].astype(str) + "-" + peaks_df[2].astype(str)

        if mat.shape[0] == len(peak_names):
            mat = mat.T

        adata_atac = sc.AnnData(X=mat)
        adata_atac.obs_names = barcodes
        adata_atac.var_names = peak_names.values
        adata_atac.var_names_make_unique()
        del mat, peaks_df
        gc.collect()

        # Read Metadata and align
        file_meta = os.path.join(data_dir, config["file_meta"])
        try:
            meta = pd.read_csv(file_meta, sep='\t')
        except Exception:
            meta = pd.read_csv(file_meta, sep=r'\s+')
        meta.columns = meta.columns.str.strip()

        if 'rna.bc' in meta.columns and 'celltype' in meta.columns:
            type_map = dict(zip(meta['rna.bc'], meta['celltype']))
            common_cells = sorted(list(set(adata_rna.obs_names) & set(adata_atac.obs_names) & set(meta['rna.bc'])))
            if len(common_cells) == 0:
                raise ValueError("Cell intersection is 0, please check barcode replacement rules")

            adata_rna = adata_rna[common_cells].copy()
            adata_atac = adata_atac[common_cells].copy()
            adata_rna.obs['cell_type'] = adata_rna.obs_names.map(type_map)

            valid_mask = ~adata_rna.obs['cell_type'].isna()
            adata_rna = adata_rna[valid_mask].copy()
            adata_atac = adata_atac[valid_mask].copy()
        else:
            raise ValueError("Metadata must contain 'rna.bc' and 'celltype' columns")

# Mode D: Human Fetal Brain raw chunked TSV format 
    elif data_type == "human_fetal_brain_raw":
        print(f"Loading Mode: Human Fetal Brain native streaming fast load")
        
        file_map = os.path.join(data_dir, config["file_map"])
        file_meta = os.path.join(data_dir, config["file_meta"])
        df_map = pd.read_csv(file_map, sep='\t', header=0)
        cluster_dict = dict(zip(df_map['Cluster.ID'].astype(str), df_map['Cluster.Name']))

        df_meta = pd.read_csv(file_meta, sep='\t', index_col='Cell.ID')
        if 'seurat_clusters' in df_meta.columns:
            df_meta['cell_type'] = df_meta['seurat_clusters'].astype(str).map(cluster_dict)
        else:
            df_meta['cell_type'] = df_meta.iloc[:, 0].astype(str)

        file_rna = os.path.join(data_dir, config["file_rna"])
        print(f"Streaming accelerated reading of RNA: {os.path.basename(file_rna)} ...")
        
        gene_names = []
        r_list, c_list, d_list = [], [], []
        with gzip.open(file_rna, 'rt') as f:
            header = f.readline().rstrip('\n\r').split('\t')
            cell_names = header
            n_cells = len(cell_names)

            for g_idx, line in enumerate(tqdm(f, desc="Streaming RNA")):
                parts = line.rstrip('\n\r').split('\t')
                gene_names.append(parts[0])
                vals = np.fromiter((float(x) for x in parts[1:]), dtype=np.float32, count=n_cells)
                nz = np.nonzero(vals)[0]
                if len(nz) > 0:
                    r_list.append(np.full(len(nz), g_idx, dtype=np.int32))
                    c_list.append(nz.astype(np.int32))
                    d_list.append(vals[nz])

        mat_rna = sp.csr_matrix(
            (np.concatenate(d_list), (np.concatenate(r_list), np.concatenate(c_list))),
            shape=(len(gene_names), n_cells),
            dtype=np.float32
        )
        del r_list, c_list, d_list
        gc.collect()

        adata_rna = sc.AnnData(X=mat_rna.T.tocsr())
        adata_rna.obs_names = cell_names
        adata_rna.var_names = np.array(gene_names)
        del mat_rna
        gc.collect()
        print(f"RNA fast load completed:: {adata_rna.shape}")

        file_peaks = os.path.join(data_dir, config["file_peaks"])
        file_atac = os.path.join(data_dir, config["file_atac"])
        print(f"Reading ATAC Peaks and 460k-row Counts")
        
        df_peaks = pd.read_csv(file_peaks, sep='\t')
        peak_names = (df_peaks['seqnames'].astype(str) + ":" + 
                      df_peaks['start'].astype(str) + "-" + 
                      df_peaks['end'].astype(str)).values
        del df_peaks

        r_list, c_list, d_list = [], [], []
        with gzip.open(file_atac, 'rt') as f:
            for p_idx, line in enumerate(tqdm(f, desc="Streaming ATAC peaks (460k)")):
                parts = line.rstrip('\n\r').split('\t')
                if len(parts) == n_cells + 1:
                    vals = np.fromiter((float(x) for x in parts[1:]), dtype=np.float32, count=n_cells)
                else:
                    vals = np.fromiter((float(x) for x in parts), dtype=np.float32, count=n_cells)

                nz = np.nonzero(vals)[0]
                if len(nz) > 0:
                    r_list.append(np.full(len(nz), p_idx, dtype=np.int32))
                    c_list.append(nz.astype(np.int32))
                    d_list.append(vals[nz])

        mat_atac = sp.csr_matrix(
            (np.concatenate(d_list), (np.concatenate(r_list), np.concatenate(c_list))),
            shape=(len(peak_names), n_cells),
            dtype=np.float32
        )
        del r_list, c_list, d_list
        gc.collect()

        adata_atac = sc.AnnData(X=mat_atac.T.tocsr())
        adata_atac.obs_names = cell_names
        adata_atac.var_names = peak_names
        del mat_atac
        gc.collect()
        print(f"ATAC fast load completed: {adata_atac.shape}")

        # Align Metadata labels
        print("Aligning cells with metadata")
        common_cells = np.intersect1d(adata_rna.obs_names, df_meta.index)
        adata_rna = adata_rna[common_cells].copy()
        adata_atac = adata_atac[common_cells].copy()
        adata_rna.obs['cell_type'] = df_meta.loc[common_cells, 'cell_type'].values

        valid_mask = ~adata_rna.obs['cell_type'].isna()
        adata_rna = adata_rna[valid_mask].copy()
        adata_atac = adata_atac[valid_mask].copy()
        gc.collect()

    # Mode E: SNARE-seq MTX dual triplet format (AdBC, P0BC)
    elif data_type == "snare_seq_mtx":
        print(f"Loading Mode: SNARE-seq MTX triplet format")
        adata_rna = load_snare_part(
            os.path.join(data_dir, config["file_rna_mtx"]),
            os.path.join(data_dir, config["file_rna_barcodes"]),
            os.path.join(data_dir, config["file_rna_genes"]),
            "RNA"
        )
        adata_atac = load_snare_part(
            os.path.join(data_dir, config["file_atac_mtx"]),
            os.path.join(data_dir, config["file_atac_barcodes"]),
            os.path.join(data_dir, config["file_atac_peaks"]),
            "ATAC"
        )

        meta_path = os.path.join(data_dir, config["file_meta"])
        meta = pd.read_csv(meta_path, sep='\t', index_col='cellid')
        if 'celltype' not in meta.columns:
            raise ValueError("Metadata must contain 'celltype' column")

        common_cells = sorted(list(set(adata_rna.obs_names) & set(adata_atac.obs_names) & set(meta.index)))
        if len(common_cells) == 0:
            raise ValueError("Three-way intersection cell count is 0, please check cellid format")

        adata_rna = adata_rna[common_cells].copy()
        adata_atac = adata_atac[common_cells].copy()
        adata_rna.obs['cell_type'] = meta.loc[common_cells, 'celltype'].values

        valid_mask = ~adata_rna.obs['cell_type'].isna()
        adata_rna = adata_rna[valid_mask].copy()
        adata_atac = adata_atac[valid_mask].copy()

    # Mode F: sci-CAR matrix format (MouseKidney_sciCAR)
    elif data_type == "scicar_raw":
        print(f"Loading Mode: sci-CAR triplet format")
        adata_rna = load_scicar_matrix(
            os.path.join(data_dir, config["file_rna_count"]),
            os.path.join(data_dir, config["file_rna_cell"]),
            os.path.join(data_dir, config["file_rna_gene"]),
            "RNA"
        )
        adata_atac = load_scicar_matrix(
            os.path.join(data_dir, config["file_atac_count"]),
            os.path.join(data_dir, config["file_atac_cell"]),
            os.path.join(data_dir, config["file_atac_peak"]),
            "ATAC"
        )

        # Compute cell intersection
        common_cells = np.intersect1d(adata_rna.obs_names, adata_atac.obs_names)
        adata_rna = adata_rna[common_cells].copy()
        adata_atac = adata_atac[common_cells].copy()

        # Clean and remove invalid labels
        adata_rna.obs['cell_type'] = adata_rna.obs['cell_type'].replace(['NA', 'nan', 'N/A'], np.nan)
        valid_mask = ~adata_rna.obs['cell_type'].isna()
        adata_rna = adata_rna[valid_mask].copy()
        adata_atac = adata_atac[valid_mask].copy()

    # Standardize label columns
    possible_keys = ['cell_type', 'celltype', 'CellType', 'Cluster', 'subclass', 'Main_Cell_Type']
    for key in possible_keys:
        if key in adata_rna.obs.columns:
            if key != 'cell_type':
                adata_rna.obs['cell_type'] = adata_rna.obs[key]
            break
    
    # Filter out null/unknown/invalid labeled cells uniformly across all datasets
    if 'cell_type' in adata_rna.obs.columns:
        labels_series = adata_rna.obs['cell_type'].astype(str).str.strip().str.lower()
        global_valid_mask = ~adata_rna.obs['cell_type'].isna() & ~labels_series.isin(['nan', 'na', 'n/a', 'unknown', 'none', ''])
        
        if global_valid_mask.sum() < adata_rna.shape[0]:
            n_drop = adata_rna.shape[0] - global_valid_mask.sum()
            print(f"Successfully filtered and removed {n_drop} invalid/empty label cells")
            adata_rna = adata_rna[global_valid_mask].copy()
            adata_atac = adata_atac[global_valid_mask].copy()

    if not sp.isspmatrix_csr(adata_rna.X):
        adata_rna.X = sp.csr_matrix(adata_rna.X, dtype=np.float32)
    if not sp.isspmatrix_csr(adata_atac.X):
        adata_atac.X = sp.csr_matrix(adata_atac.X, dtype=np.float32)

    gc.collect()
    print(f"Successfully loaded: RNA {adata_rna.shape}, ATAC {adata_atac.shape}")
    return adata_rna, adata_atac

def process_dataset(name, config):
    print("\n" + "=" * 65)
    print(f"Starting to process dataset: {name}")
    print("=" * 65)

    output_dir = config["output_dir"]
    gtf_path = os.path.join(config["data_dir"], config["gtf_filename"])
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Step 1] Loading and adapting modalities")
    adata_rna, adata_atac = load_data(config)

    full_rna_path = os.path.join(output_dir, "RNA_matrix_full.npy")
    safe_save_full_matrix(adata_rna, full_rna_path, chunk_size=4000)
                          
    print(f"[Step 3] RNA normalization and log1p (retaining full {adata_rna.shape[1]} genes) ")
    adata_rna.layers['counts'] = adata_rna.X.copy()
    if adata_rna.X.max() > 20:
        sc.pp.normalize_total(adata_rna, target_sum=1e4)
        sc.pp.log1p(adata_rna)
    print(f"RNA normalization completed, full gene count: {adata_rna.shape[1]}")

    print(f"[Step 4] Parsing annotations and Peak coordinates")
    gene_dict = parse_gtf(gtf_path)
    peaks_by_chrom = parse_peaks(adata_atac.var_names)

    print(f"[Step 5] Sparse matrix accelerated computation of ATAC activity for [all genes]")
    target_genes = adata_rna.var_names
    n_peaks = adata_atac.shape[1]
    row_indices, col_indices = [], []

    for g_idx, gene in enumerate(tqdm(target_genes, desc=f"Mapping Peaks ({name})")):
        if gene not in gene_dict:
            continue
        chrom, start, end, strand = gene_dict[gene]
        if chrom not in peaks_by_chrom:
            continue

        if strand == '+':
            region_start = max(0, start - 2000)
            region_end = end
        else:
            region_start = start
            region_end = end + 2000

        centers, p_indices = peaks_by_chrom[chrom]
        left = bisect.bisect_left(centers, region_start)
        right = bisect.bisect_right(centers, region_end)

        for i in range(left, right):
            row_indices.append(p_indices[i])
            col_indices.append(g_idx)
            
    mapping_matrix = sp.csr_matrix(
        (np.ones(len(row_indices), dtype=np.float32), (row_indices, col_indices)),
        shape=(n_peaks, len(target_genes)),
        dtype=np.float32
    )
    gene_activity_csr = adata_atac.X.dot(mapping_matrix)

    gene_sums = np.array(gene_activity_csr.sum(axis=0)).flatten()
    valid_mask = gene_sums > 0
    valid_indices = np.where(valid_mask)[0]

    gene_activity_csr = gene_activity_csr[:, valid_indices]
    rna_aligned = adata_rna[:, valid_indices].copy()
    print(f"Candidate gene count after natural alignment: {rna_aligned.shape[1]}")
    print(f"[Step 6] Selecting Top 2000 highly variable genes directly from aligned matrix")
    target_final = min(2000, rna_aligned.shape[1])
    sc.pp.highly_variable_genes(rna_aligned, n_top_genes=target_final, subset=False)
    hvg_mask = rna_aligned.var['highly_variable'].values

    rna_final = rna_aligned[:, hvg_mask].copy()
    atac_final = gene_activity_csr[:, hvg_mask].toarray()
    atac_final = np.log1p(atac_final).astype(np.float32)
    print(f"Final retained gene dimension: {rna_final.shape[1]}")
    print(f"[Step 7] Saving final files to: {output_dir}")
    X_data = rna_final.X.toarray().astype(np.float32) if sp.issparse(rna_final.X) else rna_final.X.astype(np.float32)

    current_labels = rna_final.obs['cell_type'].values
    unique_labels = np.unique(current_labels)
    label_to_int = {label_name: i for i, label_name in enumerate(unique_labels)}
    labels_int = np.array([label_to_int[label_name] for label_name in current_labels], dtype=np.int64)

    np.save(os.path.join(output_dir, 'RNA_matrix.npy'), X_data)
    np.save(os.path.join(output_dir, 'ATAC_gene_activity.npy'), atac_final)
    np.save(os.path.join(output_dir, 'Labels.npy'), labels_int)
    np.save(os.path.join(output_dir, 'label_map.npy'), unique_labels)
    np.savetxt(os.path.join(output_dir, 'gene_names.csv'), rna_final.var_names, fmt='%s')

    print(f"Dataset {name} processing completed\n")
    gc.collect()

def main():
    datasets_config = {
        # 1. PBMC_10k (Human GRCh38)
        "PBMC_10k": {
            "type": "dual_h5ad",
            "data_dir": "../../data/PBMC_10k",
            "file_rna": "10x-Multiome-Pbmc10k-RNA.h5ad",
            "file_atac": "10x-Multiome-Pbmc10k-ATAC.h5ad",
            "output_dir": "../../dataset/pre_data/PBMC_10k_Processed",
            "gtf_filename": "gencode.v41.annotation.gtf.gz",
        },
        # 2. SNARE_seq_Cortex (Mouse mm10)
        "SNARE_seq_Cortex": {
            "type": "dual_h5ad",
            "data_dir": "../../data/SNARE_seq_Cortex",
            "file_rna": "Chen-2019-RNA.h5ad",
            "file_atac": "Chen-2019-ATAC.h5ad",
            "output_dir": "../../dataset/pre_data/SNARE_seq_Cortex_Processed",
            "gtf_filename": "gencode.vM23.annotation.gtf.gz",
        },
        # 3. SHARE_seq_SkinA (Mouse mm10)
        "SHARE_seq_SkinA": {
            "type": "dual_h5ad",
            "data_dir": "../../data/SHARE_seq_SkinA",
            "file_rna": "Ma-2020-RNA.h5ad",
            "file_atac": "Ma-2020-ATAC.h5ad",
            "output_dir": "../../dataset/pre_data/SHARE_seq_SkinA_Processed",
            "gtf_filename": "gencode.vM23.annotation.gtf.gz",
        },
        # 4. SHARE_seq_Brain (Mouse mm10 - Raw format)
        "SHARE_seq_Brain": {
            "type": "share_seq_raw",
            "data_dir": "../../data/SHARE_seq_Brain",
            "file_rna": "GSM4156610_brain.rna.counts.txt.gz",
            "file_atac_mtx": "GSM4156599_brain.counts.txt.gz",
            "file_atac_barcodes": "GSM4156599_brain.barcodes.txt.gz",
            "file_atac_peaks": "GSM4156599_brain.peaks.bed.gz",
            "file_meta": "GSM4156599_brain_celltype.txt.gz",
            "output_dir": "../../dataset/pre_data/SHARE_seq_Brain_Processed",
            "gtf_filename": "gencode.vM23.annotation.gtf.gz",
        },
        # 5. SHARE_seq_SkinB (Mouse mm10 - Raw format, 34k large sample streaming)
        "SHARE_seq_SkinB": {
            "type": "share_seq_raw",
            "data_dir": "../../data/SHARE_seq_SkinB",
            "file_rna": "GSM4156608_skin.late.anagen.rna.counts.txt.gz",
            "file_atac_mtx": "GSM4156597_skin.late.anagen.counts.txt.gz",
            "file_atac_barcodes": "GSM4156597_skin.late.anagen.barcodes.txt.gz",
            "file_atac_peaks": "GSM4156597_skin.late.anagen.peaks.bed.gz",
            "file_meta": "GSM4156597_skin_celltype.txt.gz",
            "output_dir": "../../dataset/pre_data/SHARE_seq_SkinB_Processed",
            "gtf_filename": "gencode.vM23.annotation.gtf.gz",
        },
        # 6. BMMC (Human GRCh38 - Single h5ad split)
        "BMMC": {
            "type": "single_h5ad",
            "data_dir": "../../data/BMMC",
            "input_file": "GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad",
            "output_dir": "../../dataset/pre_data/BMMC_Processed",
            "gtf_filename": "gencode.v41.annotation.gtf.gz",
        },
        # 7. Human_Fetal_Brain (Human GRCh38 - Chunked TSV raw multi-files)
        "Human_Fetal_Brain": {
            "type": "human_fetal_brain_raw",
            "data_dir": "../../data/Human_Fetal_Brain",
            "file_rna": "GSE162170_multiome_rna_counts.tsv.gz",
            "file_atac": "GSE162170_multiome_atac_counts.tsv.gz",
            "file_peaks": "GSE162170_multiome_atac_consensus_peaks.txt.gz",
            "file_meta": "GSE162170_multiome_cell_metadata.txt.gz",
            "file_map": "GSE162170_multiome_cluster_names.txt.gz",
            "output_dir": "../../dataset/pre_data/Human_Fetal_Brain_Processed",
            "gtf_filename": "gencode.v41.annotation.gtf.gz",
        },
        # 8. SNARE_seq_AdBC (Mouse mm10 - MTX triplets)
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
            "output_dir": "../../dataset/pre_data/SNARE_seq_AdBC_Processed",
            "gtf_filename": "gencode.vM23.annotation.gtf.gz",
        },
        # 9. SNARE_seq_P0BC (Mouse mm10 - MTX triplets)
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
            "output_dir": "../../dataset/pre_data/SNARE_seq_P0BC_Processed",
            "gtf_filename": "gencode.vM23.annotation.gtf.gz",
        },
        # 10. MouseKidney_sciCAR (Mouse mm10 - sci-CAR format)
        "MouseKidney_sciCAR": {
            "type": "scicar_raw",
            "data_dir": "../../data/MouseKidney_sciCAR",
            "file_rna_count": "GSM3271044_RNA_mouse_kidney_gene_count.txt.gz",
            "file_rna_cell": "GSM3271044_RNA_mouse_kidney_cell.txt.gz",
            "file_rna_gene": "GSM3271044_RNA_mouse_kidney_gene.txt.gz",
            "file_atac_count": "GSM3271045_ATAC_mouse_kidney_peak_count.txt.gz",
            "file_atac_cell": "GSM3271045_ATAC_mouse_kidney_cell.txt.gz",
            "file_atac_peak": "GSM3271045_ATAC_mouse_kidney_peak.txt.gz",
            "output_dir": "../../dataset/pre_data/MouseKidney_sciCAR_Processed",
            "gtf_filename": "gencode.vM23.annotation.gtf.gz",
        }
    }

    parser = argparse.ArgumentParser(description="Multi-omics unified preprocessing pipeline (Supports all 10 datasets)")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all"] + list(datasets_config.keys()),
                        help="Specify the name of the dataset to process, default 'all'")
    args = parser.parse_args()

    if args.dataset == "all":
        for name, config in datasets_config.items():
            process_dataset(name, config)
    else:
        process_dataset(args.dataset, datasets_config[args.dataset])


if __name__ == "__main__":
    main()