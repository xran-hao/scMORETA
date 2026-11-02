import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import LabelEncoder


def main():
    raw_h5ad = "../../data/BMMC/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad"
    bmmc_dir = "../../dataset/pre_data/BMMC_Processed"
    out_dir = os.path.join(bmmc_dir, "InterDonor_BMMC")
    os.makedirs(out_dir, exist_ok=True)

    source_donor_id = "15078"
    target_donor_id = "19593"
    num_clusters = 200  

    print("=" * 65)
    print("Slicing and extracting inter-donor data and generating global_adj.pt")
    print("=" * 65)

    print("[1/4] Reading cell metadata (DonorID and cell_type)")
    adata_meta = sc.read_h5ad(raw_h5ad, backed='r')
    obs = adata_meta.obs.copy()

    for col in ['cell_type', 'celltype', 'CellType', 'Cluster', 'subclass']:
        if col in obs.columns:
            obs['cell_type'] = obs[col]
            break
    obs['DonorID'] = obs['DonorID'].astype(str)

    source_mask_all = (obs['DonorID'] == source_donor_id)
    target_mask_all = (obs['DonorID'] == target_donor_id)

    common_types = sorted(list(
        set(obs.loc[source_mask_all, 'cell_type']).intersection(
            set(obs.loc[target_mask_all, 'cell_type'])
        )
    ))
    print(f"Common cell types between donor {source_donor_id} and {target_donor_id}: {len(common_types)}")

    le = LabelEncoder()
    le.fit(common_types)

    print("[2/4] Using mmap zero-memory slicing on existing matrices")
    rna_path = os.path.join(bmmc_dir, "RNA_matrix.npy")
    atac_path = os.path.join(bmmc_dir, "ATAC_gene_activity.npy")

    rna_mmap = np.load(rna_path, mmap_mode='r')
    atac_mmap = np.load(atac_path, mmap_mode='r')
    n_genes = rna_mmap.shape[1] 

    def save_donor(donor_id, folder_name):
        mask = (obs['DonorID'] == donor_id) & (obs['cell_type'].isin(common_types))
        indices = np.where(mask.values)[0]

        donor_dir = os.path.join(out_dir, folder_name)
        os.makedirs(donor_dir, exist_ok=True)

        sub_rna = np.array(rna_mmap[indices], dtype=np.float32)
        sub_atac = np.array(atac_mmap[indices], dtype=np.float32)
        sub_labels_str = obs['cell_type'].iloc[indices].values
        sub_labels_num = le.transform(sub_labels_str)

        np.save(os.path.join(donor_dir, "RNA_matrix.npy"), sub_rna)
        np.save(os.path.join(donor_dir, "ATAC_gene_activity.npy"), sub_atac)
        np.save(os.path.join(donor_dir, "Labels.npy"), sub_labels_num)
        np.save(os.path.join(donor_dir, "Labels_str.npy"), sub_labels_str)
        print(f"  -> [{folder_name}] Saved successfully: ({sub_rna.shape[0]}, {sub_rna.shape[1]})")

    save_donor(source_donor_id, f"Source_Train_{source_donor_id}")
    save_donor(target_donor_id, f"Target_Test_{target_donor_id}")

    print("[3/4] Generating graph structure scaffold global_adj.pt for training set")
    source_dir = os.path.join(out_dir, f"Source_Train_{source_donor_id}")
    
    global_adj = torch.randn(n_genes, num_clusters, dtype=torch.float32)
    torch.save(global_adj, os.path.join(source_dir, "global_adj.pt"))
    print(f"Saved to Source directory: global_adj.pt, Dimensions: {tuple(global_adj.shape)}")

    print("[4/4] Saving shared feature_names and label_map")
    gene_csv = os.path.join(bmmc_dir, "gene_names.csv")
    feat_txt = os.path.join(out_dir, f"feature_names_{n_genes}.txt")
    if os.path.exists(gene_csv):
        genes = pd.read_csv(gene_csv, header=None)[0].tolist()
        with open(feat_txt, 'w') as f:
            for g in genes:
                f.write(f"{g}\n")

    label_map = {i: name for i, name in enumerate(le.classes_)}
    np.save(os.path.join(out_dir, "label_map.npy"), label_map)

    print("\n" + "=" * 65)
    print(f"All generation completed：\n{out_dir}")
    print(f"├── Source_Train_{source_donor_id}/")
    print(f"│     ├── RNA_matrix.npy")
    print(f"│     ├── ATAC_gene_activity.npy")
    print(f"│     ├── Labels.npy")
    print(f"│     ├── Labels_str.npy")
    print(f"│     └── global_adj.pt ")
    print(f"├── Target_Test_{target_donor_id}/")
    print(f"│     ├── RNA_matrix.npy")
    print(f"│     ├── ATAC_gene_activity.npy")
    print(f"│     ├── Labels.npy")
    print(f"│     └── Labels_str.npy")
    print(f"├── feature_names_{n_genes}.txt")
    print(f"└── label_map.npy")
    print("=" * 65)


if __name__ == "__main__":
    main()