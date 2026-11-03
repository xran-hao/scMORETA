# scMORETA: Topological Hypergraph Learning for Cellular Identity Resolution and Resilience Profiling in Single-Cell Multi-Omics

> **Nomenclature Note**: The proposed method was previously designated as **T-CSGN** (Topological Cell-Specific Graph Network) during internal development. In the associated manuscript, it has been officially renamed to **scMORETA**. Please note that identifiers such as `TCSGN`, `tcsgn`, `CSGNet`, and related model checkpoint names in this repository refer to **scMORETA**.

## 1. Project Overview
We propose **scMORETA**, a topological deep learning framework designed for robust single-cell multi-omics (scRNA-seq and scATAC-seq) cell-type annotation and molecular resilience profiling. Rooted in the biological principle of epigenetic priming, scMORETA leverages Mapper-based Topological Data Analysis (TDA) to construct high-order chromatin accessibility hypergraphs, while dynamically specializing context-specific regulatory subnetworks via individual gene expression. Integrated with supervised contrastive learning and cell-specific hypergraph convolutions, scMORETA consistently achieves superior classification accuracy, Macro-F1, and Cohen's Kappa across 10 diverse single-cell multi-omics datasets. Furthermore, it establishes an in silico transcriptomic stress-testing paradigm and introduces the Identity Resilience Score (IRS), enabling quantitative profiling of cellular identity destabilization and mechanistic decoding of functional regulatory modules governing cellular robustness under severe transcriptomic perturbations.

---

### 1.1 Project Directory Structure

```
├── data/                                      # Raw multi-omics datasets (downloaded separately)
│   ├── PBMC_10k/                              # 10x Multiome PBMC
│   ├── BMMC/                                  # NeurIPS 2021 Multiome BMMC
│   ├── Human_Fetal_Brain/                     # Chunked TSV multi-files
│   ├── MouseKidney_sciCAR/                    # sci-CAR triplet format
│   ├── SHARE_seq_Brain/                       # SHARE-seq raw files
│   ├── SHARE_seq_SkinA/                       # Dual h5ad format
│   ├── SHARE_seq_SkinB/                       # Large-scale streaming files
│   ├── SNARE_seq_AdBC/                        # Adult brain cortex MTX triplets
│   ├── SNARE_seq_Cortex/                      # Dual h5ad format
│   └── SNARE_seq_P0BC/                        # P0 brain cortex MTX triplets
│
├── dataset/                                   # Processed experimental data
│   └── pre_data/                              # Standardized datasets
│       ├── BMMC_Processed/                    
│       │   ├── fold_1/ ~ fold_5/              # Synchronized 5-fold data splits
│       │   ├── InterDonor_BMMC/               # Cross-donor evaluation subsets
│       │   │   ├── Source_Train_15078/        # Training donor (Donor 15078)
│       │   │   └── Target_Test_19593/         # Unseen target donor (Donor 19593)
│       │   ├── RNA_matrix.npy                 # Top 2,000 HVG normalized RNA matrix
│       │   ├── RNA_matrix_full.npy            # Full-scale RNA expression matrix
│       │   ├── ATAC_gene_activity.npy         # GTF-aligned ATAC gene activity matrix
│       │   ├── Labels.npy                     # Numerical cell labels
│       │   ├── label_map.npy                  # Label name mapping dictionary
│       │   ├── gene_names.csv                 # Synchronized gene list
│       │   └── cv_indices.npz                 # 5-fold cross-validation split indices
│       ├── Human_Fetal_Brain_Processed/       
│       ├── MouseKidney_sciCAR_Processed/      
│       ├── PBMC_10k_Processed/                
│       ├── SHARE_seq_Brain_Processed/         
│       ├── SHARE_seq_SkinA_Processed/         
│       ├── SHARE_seq_SkinB_Processed/         
│       ├── SNARE_seq_AdBC_Processed/          
│       ├── SNARE_seq_Cortex_Processed/        
│       └── SNARE_seq_P0BC_Processed/          
│ 
├── src/                                       # Core source code of scMORETA
│   ├── DataPreprocessing/                     # Data wrangling and metric calculation
│   │   ├── preprocess_all_datasets.py         # Unified multi-omics preprocessing pipeline
│   │   ├── compute_dataset_metrics.py         # Profile and complexity statistics calculator
│   │   ├── extract_inter_donor_data.py        # BMMC cross-donor data extraction utility
│   │   └── dataset_statistics_summary.csv     # Output summary table (Table S1)
│   ├── Cell_Confusion_Matrices_Top14/         # Top ≤14 populations comparative confusion matrices (Figure 5 & S4)
│   ├── Figures_output/                        # High-resolution (300 DPI) publication-ready PNG & SVG figures
│   │   ├── Figure2A_Confusion                 # Row-normalized out-of-fold classification confusion matrices
│   │   ├── Figure2B_Umap                      # Latent space UMAP projections
│   │   ├── Figure2C_Sankey                    # Alluvial Sankey mapping diagrams
│   │   └── Figure6A-Figure8B
│   ├── Plot                                   # Figure3-Figure4
│   ├── model.py                               # CSGNet model architecture & SupConLoss definition
│   ├── gene_mapper.py                         # Mapper topological hypergraph construction algorithm
│   ├── datasets.py                            # PyTorch dataset loaders
│   ├── plot_utils.py                          # Plotting utilities for UMAP, ROC, and skeleton graphs
│   ├── tcsgn_classify_train.py                # 5-fold cross-validation training engine
│   ├── tcsgn_classify_test.py                 # Model inference, OOF evaluation & stress testing
│   ├── train_50_50.py                         # 50/50 Train/Test split training engine (Figure 4)
│   ├── test_50_50_plot.py                     # 50/50 representation evaluation & UMAP plotting (Figure 4)
│   ├── tcsgn_donor_train.py                   # BMMC cross-donor zero-shot generalization training (Figure S5)
│   ├── donor_global_adj.py                    # Cross-donor topological hypergraph generator
│   ├── tcsgn_rarecell_analysis.py             # Rare cell (<3%) identification & evaluation
│   ├── plot_rare_confusion_matrices.py        # Top ≤14 comparative confusion matrix generator (Figure 5 & S4)
│   ├── tcsgn_resilience_analysis.py           # IRS scoring, lineage dynamics & mechanisms (Figure 6D-F, 7A, 8A-B)
│   ├── tcsgn_insilico_pertubation.py          # In silico gene knockdown perturbation (Figure 6C, Table S3)
│   ├── plot_paper_resilience_figure.py        # Multi-model stress degradation curves (Figure 6A-B)
│   ├── tcsgn_sankey_compensation.py           # Hyperedge regulatory compensation flow Sankey (Figure 7B)
│   ├── run_GO_enrichment.py                   # Hyperedge regulatory module GO functional enrichment (Figure 7C)
│   ├── calc_SNR.py                            # Modular activation specificity and SNR rankings (Table S4)
│   ├── Figure2C&supp3_sankey_10datasets.py    # Alluvial Sankey mapping generator for 10 datasets (Figure 2C & S3)
│   ├── run_scjoint_benchmark.py               # Benchmark runner for scJoint
│   ├── run_benchmark_mogonet.py               # Benchmark runner for MOGONET
│   ├── run_benchmark_multikano.py             # Benchmark runner for MultiKano
│   ├── run_baseline_donor.py                  # scJoint & MultiKano cross-donor baseline evaluation
│   └── run_baseline_mogonet_donor.py          # MOGONET cross-donor baseline evaluation
│
├── requirements.txt                           # Python environment specifications
└── README.md                                  # Project documentation
```

## 2. Installation & Environment Setup
Hardware & OS Requirements
Operating System: Ubuntu 24.04.4 LTS
GPU: NVIDIA TITAN V (12GB)
CUDA Version: 12.8
Environment Setup
To reproduce scMORETA, we suggest creating an isolated Conda environment:
  ### Create and activate a clean conda environment
    conda create -n scmoreta python=3.9 -y
    conda activate scmoreta

  ### Install required packages
    pip install -r requirements.txt

## 3. Workflow & Usage

### 3.1 Data Collection and Preprocessing

#### 3.1.1 Download Single-Cell Multi-Omics Datasets
Save raw downloaded files into the corresponding `data/<Dataset>/` directory according to the file list in Table S2. Each directory requires its corresponding reference genome annotation file:
* **Human datasets (GRCh38):** `gencode.v41.annotation.gtf.gz` ([Download](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/gencode.v41.annotation.gtf.gz))
* **Mouse datasets (mm10):** `gencode.vM23.annotation.gtf.gz` ([Download](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M23/gencode.vM23.annotation.gtf.gz))

#### 3.1.2 Multi-Omics Preprocessing and Metric Calculation

* **Unified Data Preprocessing Pipeline**

Preprocesses raw multi-omics files (supporting Dual `.h5ad`, Single `.h5ad` split, raw SHARE-seq/Human Fetal Brain TSV streams, and SNARE-seq/sci-CAR MTX triplets). Aligns barcodes, cleans invalid labels, streams full RNA expression matrices, calculates ATAC gene activity via GTF binary search, and selects the top 2,000 highly variable genes (HVGs).

> **Execution Commands:**
> * Run single dataset (e.g., PBMC_10k): `python src/DataPreprocessing/preprocess_all_datasets.py --dataset PBMC_10k`
> * Run all 10 datasets in batch: `python src/DataPreprocessing/preprocess_all_datasets.py --dataset all`

> **Outputs Saved to:** `dataset/pre_data/<Dataset>_Processed/` (contains `RNA_matrix.npy`, `RNA_matrix_full.npy`, `ATAC_gene_activity.npy`, `Labels.npy`, `label_map.npy`, and `gene_names.csv`).

* **Dataset Profile & Complexity Statistics**

Calculates core dataset metrics (cell count, gene count, peak count, cell types, major type proportion, Shannon entropy imbalance degree, and RNA/ATAC sparsity).

> **Execution Commands:**
> * Run single dataset (e.g., PBMC_10k): `python src/DataPreprocessing/compute_dataset_metrics.py --dataset PBMC_10k`
> * Run all 10 datasets in batch: `python src/DataPreprocessing/compute_dataset_metrics.py --dataset all --save_path dataset_statistics_summary.csv`

> **Outputs Saved to:** `./dataset_statistics_summary.csv`.

### 3.2 Training and Prediction using scMORETA

#### 3.2.1 Model Training and Cross-Validation

Constructs the global topological hypergraph (`global_adj_*.pt`) via Mapper, synchronizes 5-fold cross-validation splits, and trains CSGNet with Supervised Contrastive Loss (`SupConLoss`).

> **Execution Commands:**
> * Run training (e.g., PBMC_10k): `python src/tcsgn_classify_train.py -expr ../dataset/pre_data/PBMC_10k_Processed`

> **Outputs Saved to:** `result/<Dataset>_Processed/models/` (contains `tcsgn_*_model[1-5].pth` and `global_adj_*.pt`), and training loss logs in `result/<Dataset>_Processed/train_data/`.

#### 3.2.2 Model Testing and Multi-Omics Evaluation

Evaluates trained models across all 5 folds, computes Accuracy, Macro-F1, and Cohen's Kappa, and supports multi-gradient transcriptomic dropout stress testing.

> **Execution Commands:**
> * Standard 5-fold testing (e.g., PBMC_10k): `python src/tcsgn_classify_test.py -expr ../dataset/pre_data/PBMC_10k_Processed`
> * Testing with dropout stress testing (0% to 90%): `python src/tcsgn_classify_test.py -expr ../dataset/pre_data/PBMC_10k_Processed --stress_test`

> **Outputs Saved to:** `result/<Dataset>_Processed/plot_data/` (contains `final_metrics_summary.csv`, `tcsgn_stress_results_summary.csv`, `predictions_<Dataset>.csv`).

---

### 3.3 Benchmarking against Supervised Baseline Methods

Runs 5-fold cross-validation benchmarks across all 10 datasets for baseline models (scJoint, MOGONET, and MultiKano). Please clone each official repository and place our corresponding benchmark script into the designated directory before running:

> **scJoint Benchmark:** (Clone from: https://github.com/SydneyBioX/scJoint, place `run_scjoint_benchmark.py` in `scJoint-main/`)
> * Run single dataset (e.g., PBMC_10k): `python run_scjoint_benchmark.py -expr ../scMORETA-main/dataset/pre_data/PBMC_10k_Processed`
> * Run all 10 datasets in batch: `python run_scjoint_benchmark.py -all`

> **MOGONET Benchmark:** (Clone from: https://github.com/txWang/MOGONET, place `run_benchmark_mogonet.py` in `MOGONET-main/`)
> * Run single dataset (e.g., PBMC_10k): `python run_benchmark_mogonet.py -expr ../scMORETA-main/dataset/pre_data/PBMC_10k_Processed`
> * Run all 10 datasets in batch: `python run_benchmark_mogonet.py -all`

> **MultiKano Benchmark:** (Clone from: https://github.com/BioX-NKU/MultiKano, place `run_benchmark_multikano.py` in `MultiKano-main/MultiKano/`)
> * Run single dataset (e.g., PBMC_10k): `python run_benchmark_multikano.py -expr ../../scMORETA-main/dataset/pre_data/PBMC_10k_Processed`
> * Run all 10 datasets in batch: `python run_benchmark_multikano.py -all`

> **Outputs Saved to:** `result/<Dataset>_Processed/plot_data/*_final_metrics_summary.csv` and out-of-fold prediction arrays in `result/<Dataset>_Processed/predictions/`.

---

### 3.4 Cellular Resilience & Downstream Mechanistic Analyses

#### 3.4.1 Identity Resilience Score (IRS) & Lineage Dynamics

Computes the cell-specific Identity Resilience Score (IRS) under severe transcriptomic stress (50% dropout), tracks lineage dynamics along lymphocyte differentiation, extracts topological fingerprints, and decodes molecular drivers.

> **Execution Command:** `python src/tcsgn_resilience_analysis.py`

> **Outputs Saved to:** `src/Figures_output/` (exports 300 DPI PNG and vector SVG formats for Figure 6D-F, Figure 7A, and Figure 8A-B).

#### 3.4.2 In Silico Virtual Gene Perturbation

Performs targeted in silico gene knockdown on top abundant marker genes in target cell types to quantify cellular identity retention and prediction probability shifts.

> **Execution Command:** `python src/tcsgn_insilico_pertubation.py`

> **Outputs Saved to:** `src/Figures_output/Figure6C_Combined_with_Metrics.png/.svg` and `src/Figures_output/Figure6C_Single_Top*.png/.svg`.

#### 3.4.3 Multi-Model Stress Degradation Curve Analysis

Evaluates the robustness of scMORETA compared to baseline models under continuous transcriptomic dropout stress (0% to 70%) and calculates performance retention at critical 50% stress.

> **Execution Command:** `python src/plot_paper_resilience_figure.py`

> **Outputs Saved to:** `src/Figures_output/Figure6AB_Resilience_Benchmark_Clean.png` and `.svg`

#### 3.4.4 Hyperedge Regulatory Compensation Flow Analysis

Traces topological compensatory flows starting from key target genes (e.g., *LYZ*) through top-activated dynamic hyperedges down to downstream co-regulatory partner genes.

> **Execution Command:** `python src/tcsgn_sankey_compensation.py`

> **Outputs Saved to:** `src/Figures_output/Figure7B_Sankey_Compensation_Flow.png` and `.svg`

#### 3.4.5 Functional GO Enrichment of Top Regulatory Modules

Extracts partner genes associated with core topological hyperedges and performs Gene Ontology (GO Biological Process) functional enrichment analysis using Enrichr.

> **Execution Command:** `python src/run_GO_enrichment.py`

> **Outputs Saved to:** `src/Figures_output/Figure7C_Functional_Enrichment_Top10_Exact.png` and `.svg` 

#### 3.4.6 Modular Activation Specificity and Signal-to-Noise Ratio (SNR)

Quantifies the dynamic hyperedge activation intensities across all cell lineages, computing the modular specificity (Peak, Median) and Signal-to-Noise Ratio (SNR) rankings.

> **Execution Command:** `python src/calc_SNR.py`

> **Outputs Saved to:** Terminal outputs and metrics corresponding to Table S4.

---

## 4. Figure Reproduction Instructions

### 4.1 Main Figures

#### Figure 2: Classification Performance and Manifold Representation
* **Figure 2(A)**: Row-normalized out-of-fold confusion matrices for representative datasets (left: PBMC_10k; right: SHARE_seq_Brain) generated via `src/tcsgn_classify_test.py`, saved to `src/Figures_output/Figure2A_Confusion/OOF_WholeData_Confusion_Matrix_*.png/.svg`.
* **Figure 2(B)**: Latent space UMAP projections generated via `src/tcsgn_classify_test.py` (aligned via Procrustes alignment), saved to `src/Figures_output/Figure2B_Umap/BestModel_WholeData_UMAP_*.png/.svg`.
* **Figure 2(C)**: Alluvial Sankey mapping diagrams generated via `src/Figure2C&supp3_sankey_10datasets.py` using `predictions_<Dataset>.csv`, saved to `src/Figures_output/Figure2C_Sankey/Sankey_OOF_*.png/.svg`.

#### Figure 3: Benchmarking against Supervised Baseline Methods
* **Figure 3(A-C)**: Comparative bar charts illustrating Accuracy (A), Macro-F1 (B), and Cohen's Kappa (C) across all 10 datasets against scJoint, MOGONET, and MultiKano.
  * Generated using summary metrics from `src/tcsgn_classify_test.py`, `src/run_scjoint_benchmark.py`, `src/run_benchmark_mogonet.py`, and `src/run_benchmark_multikano.py`.
  * Visualized and plotted using `src/plot/figure3.py`, which reads the compiled benchmark results and exports `figure3.png` and `figure3.svg`

#### Figure 4: Unsupervised Benchmark Evaluation on PBMC 10k
* **Figure 4(A)**: Biological conservation scores (AvgBIO, NMI, ARI, ASW_cell) comparing 50% Train vs. 50% unseen Test sets against unsupervised baselines (scGPT, scGLUE, Seurat v4).
  * Generated using `src/test_50_50_plot.py` (after training via `src/train_50_50.py`), saved to `result/PBMC_10k_Processed_50_50/eval_plots/tcsgn_50_50_evaluation_metrics.csv`.
  * Visualized and plotted using `src/plot/figure4A.py`, exporting the multi-model comparison bar chart to `figure4A.png` and `figure4A.svg`
* **Figure 4(B)**: Train Set (50%) latent space 2D UMAP projection comparing Ground Truth annotations with optimal Louvain cluster assignments generated using `src/test_50_50_plot.py`, saved to `result/PBMC_10k_Processed_50_50/eval_plots/umap_train_50pct_comparison.*`
* **Figure 4(C)**: Independent unseen Test Set (50%) latent space 2D UMAP projection generated using `src/test_50_50_plot.py`, saved to `result/PBMC_10k_Processed_50_50/eval_plots/umap_test_50pct_unseen_comparison.*`

#### Figure 5: Comparative Confusion Matrices across Four Methods
* **Figure 5**: 2×2 comparative confusion matrices across all 14 cell types in MouseKidney_sciCAR (arranged ascendingly by relative abundance) comparing scMORETA (Ours), scJoint, MOGONET, and MultiKano.
  * Generated using `src/plot_rare_confusion_matrices.py --dataset MouseKidney_sciCAR`.
  * Outputs saved to `Cell_Confusion_Matrices_Top14/Figure5_Confusion_Matrices_Top14_MouseKidney_sciCAR.png` and `.svg`

#### Figure 6: Transcriptomic Resilience under Perturbation Stress
* **Figure 6(A-B)**: Macro-F1 decay curves (0% to 70%) and retention rates (0% vs. 50%) generated using `src/plot_paper_resilience_figure.py`, saved to `src/Figures_output/Figure6AB_Resilience_Benchmark_Clean.*`.
* **Figure 6(C)**: In silico knockdown of top abundant marker genes in Naive B cells generated using `src/tcsgn_insilico_pertubation.py`, saved to `src/Figures_output/Figure6C_Combined_with_Metrics.*`.
* **Figure 6(D-F)**: Average IRS stability ranking by cell type (D), global resilience UMAP (E), and lymphocyte lineage differentiation violin plots (F) generated using `src/tcsgn_resilience_analysis.py`, saved to `src/Figures_output/Figure6D_resilience_by_cell_type_rankingPBMC.*`, `src/Figures_output/Figure6E_global_resilience_umap_finalPBMC.*`, and `src/Figures_output/Figure6F_Lymphocyte_Differentiation_Resilience_Dynamics.*`.

#### Figure 7: Topological Fingerprints and Module Mechanics
* **Figure 7(A)**: Dynamic regulatory module activation heatmap generated using `src/tcsgn_resilience_analysis.py`, saved to `src/Figures_output/Figure7A_topological_fingerprint_PBMC_10k.*`.
* **Figure 7(B)**: Hierarchical Sankey diagram of regulatory compensation flows generated using `src/tcsgn_sankey_compensation.py`, saved to `src/Figures_output/Figure7B_Sankey_Compensation_Flow.*`.
* **Figure 7(C)**: GO biological process functional enrichment bubble plot of top regulatory modules generated using `src/run_GO_enrichment.py`, saved to `src/Figures_output/Figure7C_Functional_Enrichment_Top10_Exact.*`.

#### Figure 8: Molecular Drivers of Identity Destabilization
* **Figure 8(A)**: Discriminative key marker genes and GO vulnerability pathways within CD14+ Monocytes.
* **Figure 8(B)**: Transcriptomic divergence volcano plot and regression correlation between IFN-γ inflammatory stress and IRS stability rank.
  * Generated using `src/tcsgn_resilience_analysis.py`, saved to`src/Figures_output/Figure8A_Fragility_Ridge_Mechanistic_Analysis.*`, and `src/Figures_output/Figure8B_Mechanistic_DeepDive_Volcano_and_Regression.*`.

---

### 4.2 Supplementary Figures Reproduction Instructions

### Figure S1: Out-of-Fold Confusion Matrices across Eight Cohorts
* **Figure S1**: Row-normalized out-of-fold classification confusion matrices summarizing scMORETA across the remaining eight benchmark datasets (BMMC, SHARE_seq_SkinA, SHARE_seq_SkinB, SNARE_seq_Cortex, SNARE_seq_P0BC, SNARE_seq_AdBC, Human_Fetal_Brain, and MouseKidney_sciCAR).
  * Generated using `src/tcsgn_classify_test.py`, saved to `src/Figures_output/Figure2A_Confusion/OOF_WholeData_Confusion_Matrix_*.png/.svg`.

### Figure S2: Latent Space UMAP Projections across Eight Cohorts
* **Figure S2**: 2D UMAP projections of the 64-dimensional latent representations learned by scMORETA across the remaining eight cohorts.
  * Generated using `src/tcsgn_classify_test.py` (with Procrustes alignment), saved to `src/Figures_output/Figure2B_Umap/BestModel_WholeData_UMAP_*.png/.svg`.

### Figure S3: Alluvial Sankey Diagrams across Eight Cohorts
* **Figure S3**: Alluvial mapping diagrams illustrating assignment fidelity between predicted labels and ground-truth annotations across the remaining eight cohorts.
  * Generated using `src/Figure2C&supp3_sankey_10datasets.py`, saved to `src/Figures_output/Figure2C_Sankey/Sankey_OOF_*.png/.svg`.

### Figure S4: Comparative Confusion Matrices across Nine Cohorts (Top ≤14 Populations)
* **Figure S4**: Row-normalized confusion matrices comparing scMORETA (Ours), scJoint, MOGONET, and MultiKano across the remaining nine cohorts (A to I sequentially: PBMC_10k, BMMC, SHARE_seq_Brain, SHARE_seq_SkinA, SHARE_seq_SkinB, SNARE_seq_Cortex, SNARE_seq_AdBC, SNARE_seq_P0BC, and Human_Fetal_Brain).
  * For datasets containing up to 14 cell types, all populations are presented; for datasets exceeding 14 types, the top 14 populations ranked by ascending relative abundance are displayed.
  * Generated using `src/plot_rare_confusion_matrices.py --dataset all`, saved to `Cell_Confusion_Matrices_Top14/Figure5_Confusion_Matrices_Top14_<Dataset>.png/.svg`.

### Figure S5: Cross-Donor Zero-Shot Generalization on BMMC
* **Figure S5**: Comparative analysis of cross-donor zero-shot prediction performance on the BMMC dataset (trained exclusively on Donor 15078 and deployed onto completely unseen Donor 19593 across 19 shared cell types).
  * Generated using `src/tcsgn_donor_train.py` (alongside baseline cross-donor runners `src/run_baseline_donor.py` and `src/run_baseline_mogonet_donor.py`), saving the evaluation metrics to `result/BMMC_Processed/plot_data/tcsgn_inter_donor_results.csv`.

---

### 4.3 Supplementary Tables Reproduction Instructions

### Table S1
* **Table S1**: Generated using `src/DataPreprocessing/compute_dataset_metrics.py`. Run `python src/DataPreprocessing/compute_dataset_metrics.py --dataset all --save_path dataset_statistics_summary.csv` to calculate cell/gene/peak statistics, imbalance degrees, and sparsity across all 10 datasets.

### Table S2
* **Table S2**: Curated benchmark summary of data availability, accession IDs, and raw filenames across the 10 multimodal single-cell datasets (sourced from GEO and scGLUE database).

### Table S3
* **Table S3**: Generated using `src/calc_SNR.py`. Run `python src/calc_SNR.py` to extract dynamic hyperedge activation intensities (`model.current_dynamic_H`), calculating the quantitative modular activation specificity (Peak, Median) and Signal-to-Noise Ratio (SNR) rankings across all 19 cell lineages in the PBMC 10k dataset.

### Table S4
> **Table S4**: Generated using `src/run_GO_enrichment.py`. Run `python src/run_GO_enrichment.py` to extract candidate gene modules from the global hypergraph incidence matrix, perform Gene Ontology (GO) Biological Process functional enrichment via Enrichr, and export statistical significance (`-log10(Adjusted P-value)`), module member genes, and overlapping pathway-specific genes for the top-ranked regulatory modules (saved as `src/Figures_output/Top10_Modules_Genes_Summary.csv`).
