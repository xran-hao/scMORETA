"""
Script Function: Constructs the source domain global consensus topological hypergraph (global_adj.pt) using ATAC/RNA data via the Mapper algorithm for cross-donor transfer tasks.
"""
import os
import torch
import numpy as np
from gene_mapper import get_global_adj
# Load source donor training data, apply TDA Mapper algorithm, and save the hypergraph incidence matrix as a PyTorch tensor
def generate_true_global_adj():
    source_dir = "../dataset/pre_data/BMMC_Processed/InterDonor_BMMC/Source_Train_15078"
    atac_path = os.path.join(source_dir, "ATAC_gene_activity.npy")
    rna_path = os.path.join(source_dir, "RNA_matrix.npy")
    save_path = os.path.join(source_dir, "global_adj.pt")

    print("=" * 65)

    if os.path.exists(atac_path):
        print("Loading ATAC gene activity matrix as scaffold input")
        X_skeleton = np.load(atac_path)
    else:
        print("ATAC not found, falling back to RNA expression matrix")
        X_skeleton = np.load(rna_path)

    print(f"Input feature dimensions: {X_skeleton.shape} (Cells × Genes)")
    # Execute Mapper with high resolution (35) and overlap (0.5) to capture gene co-regulatory hypergraph backbone
    print("  -> Executing get_global_adj ")
    global_adj = get_global_adj(
        X_skeleton,
        resolution=35, 
        overlap=0.5,
        eps=0.8
    )

    if not isinstance(global_adj, torch.Tensor):
        global_adj = torch.tensor(global_adj, dtype=torch.float32)
    else:
        global_adj = global_adj.float()

    torch.save(global_adj, save_path)

    print(f"True global_adj.pt generated successfully")
    print(f"Save path: {save_path}")
    print(f"Hypergraph matrix dimensions: {tuple(global_adj.shape)}")
    print("=" * 65)

if __name__ == "__main__":
    generate_true_global_adj()