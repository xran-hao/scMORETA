# Processed Dataset Guide & Reproduction

> [!NOTE]
> **Pre-packaged Benchmark Dataset (PBMC 10k)**:
> For convenience, the processed **`PBMC_10k_Processed`** dataset is already provided within this repository under `dataset/pre_data/`. 
> Large files (such as `RNA_matrix.npy` and `ATAC_gene_activity.npy`) exceed GitHub's raw file limits and are compressed into individual `.zip` files. 
> **Please extract them before running downstream scripts:**

Due to GitHub's file storage constraints, the remaining 9 benchmark datasets in `dataset/pre_data/` are not fully hosted. You can readily reproduce all 10 standardized datasets from raw files.

### 1. Raw Data Preparation
Ensure that raw multimodal sequencing files are downloaded and organized under `data/` according to **Supplementary Table S2**:

```
data/
├── BMMC/                          # NeurIPS 2021 Multiome BMMC
├── PBMC_10k/                      # 10x Multiome PBMC
├── MouseKidney_sciCAR/            # sci-CAR triplet format
├── SHARE_seq_Brain/               # SHARE-seq raw files
├── SHARE_seq_SkinA/               # Dual h5ad format
├── SHARE_seq_SkinB/               # Large-scale streaming files
├── SNARE_seq_Cortex/              # Dual h5ad format
├── SNARE_seq_AdBC/                # Adult brain cortex MTX triplets
├── SNARE_seq_P0BC/                # P0 brain cortex MTX triplets
└── Human_Fetal_Brain/             # Chunked TSV multi-files
```

### 2. One-Click Reproduction Command
From the project root directory, execute the unified preprocessing pipeline to generate any missing datasets:

> python src/DataPreprocessing/preprocess_all_datasets.py

### 3.  Generated Directory Structure
Upon extraction or execution, the dataset/pre_data/ directory will adhere to the following standardized hierarchy:

```
dataset/
└── pre_data/
    ├── PBMC_10k_Processed/                # Ready upon unzipping
    │   ├── fold_1/ ~ fold_5/              # Synchronized 5-fold cross-validation splits
    │   ├── RNA_matrix.npy                 # Top 2,000 HVG normalized RNA matrix
    │   ├── RNA_matrix_full.npy            # Unzipped full-scale RNA matrix
    │   ├── ATAC_gene_activity.npy         # Promoter activity matrix
    │   ├── Labels.npy                     # Integer cell-type labels
    │   ├── label_map.npy                  # Label dictionary
    │   └── gene_names.csv                 # Gene list
    ├── BMMC_Processed/
    │   ├── fold_1/ ~ fold_5/              # Synchronized 5-fold cross-validation splits
    │   ├── InterDonor_BMMC/               # Subsets for cross-donor evaluation
    │   │   ├── Source_Train_15078/        # Training donor (Donor 15078)
    │   │   └── Target_Test_19593/         # Unseen target donor (Donor 19593)
    │   └── ...
    └── <Other_8_Datasets>_Processed/      # Identical structure across all benchmark 
```