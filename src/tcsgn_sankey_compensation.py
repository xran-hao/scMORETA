"""
Script Function: Generates a 3-tier Sankey diagram (Figure 7B) visualizing topological signal compensation: from a target gene (LYZ) through regulatory hyperedges (Modules) to co-regulated downstream partner genes.
"""
import os
import re
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import plotly.graph_objects as go
from model import CSGNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_path = "/home/horizon/scMORETA-main/dataset/pre_data/PBMC_10k_Processed"
raw_h5ad_path = "/home/horizon/scMORETA-main/data/PBMC_10k/10x-Multiome-Pbmc10k-RNA.h5ad"
model_path = "../result/PBMC_10k_Processed/models/tcsgn_PBMC_10k_Processed_full_model1.pth"
SAVE_DIR = "Figures_output"
os.makedirs(SAVE_DIR, exist_ok=True)

TARGET_CELL_TYPE = "CD14 Mono"
TARGET_GENE_NAME = "LYZ" 

def load_data():
    gene_names = pd.read_csv(os.path.join(base_path, 'gene_names.csv'), header=None)[0].values
    labels = np.load(os.path.join(base_path, 'Labels.npy'))
    raw_rna = np.load(os.path.join(base_path, 'RNA_matrix.npy'))
    
    label_map_raw = np.load(os.path.join(base_path, 'label_map.npy'), allow_pickle=True)
    if isinstance(label_map_raw, np.ndarray) and label_map_raw.size > 1:
        label_map = {i: name for i, name in enumerate(label_map_raw)}
    elif hasattr(label_map_raw, 'item'):
        try:
            label_map = label_map_raw.item()
        except ValueError:
            label_map = {i: name for i, name in enumerate(label_map_raw)}
    else:
        label_map = label_map_raw
        
    barcodes = None
    if os.path.exists(raw_h5ad_path):
        try:
            adata_obs = sc.read_h5ad(raw_h5ad_path, backed='r')
            barcodes = adata_obs.obs_names.values
        except Exception as e:
            print(f"[!] Warning reading raw h5ad barcodes: {e}")
            
    return gene_names, labels, raw_rna, label_map, barcodes

def generate_sankey():
    gene_names, labels, raw_rna, label_map, barcodes = load_data()
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"error: {model_path}")

    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    type_to_id = {v: k for k, v in label_map.items()}
    target_id = type_to_id[TARGET_CELL_TYPE]
    gene_idx = list(gene_names).index(TARGET_GENE_NAME)
    # Locate target lineage (CD14 Mono) and select the representative single cell with peak expression of the target gene (LYZ)
    type_mask = (labels == target_id)
    type_indices = np.where(type_mask)[0]
    cell_idx = type_indices[np.argmax(raw_rna[type_mask, gene_idx])]
    
    cell_barcode_str = "Unknown"
    if barcodes is not None and len(barcodes) > cell_idx:
        cell_barcode_str = str(barcodes[cell_idx])
        
    print("=" * 65)
    print(f"[*] Representative CD14+ Monocyte:")
    print(f"  - Global Cell Index: {cell_idx}")
    print(f"  - Cell Barcode: {cell_barcode_str}")
    print(f"  - LYZ Peak Expression: {raw_rna[cell_idx, gene_idx]:.2f}")
    print("=" * 65)
    
    cell_data = torch.tensor(raw_rna[cell_idx:cell_idx+1], dtype=torch.float32).to(device)
    with torch.no_grad():
        _ = model(cell_data)
        H_dynamic = model.current_dynamic_H.squeeze().cpu().numpy()
    # Extract cell-specific dynamic hypergraph matrix H_dynamic, filtering top connected hyperedges and their co-regulated partner genes    
    candidate_edges = np.argsort(H_dynamic[gene_idx, :])[::-1]
    selected_modules = []
    module_to_valid_partners = {}

    for e_idx in candidate_edges:
        lyz_to_mod_weight = float(H_dynamic[gene_idx, e_idx])
        if lyz_to_mod_weight <= 1e-4:
            break

        edge_weights = H_dynamic[:, e_idx]
        sorted_genes = np.argsort(edge_weights)[::-1]

        valid_genes_for_this_mod = []
        for p_idx in sorted_genes:
            if p_idx == gene_idx:
                continue
            p_name = str(gene_names[p_idx]).strip()
            if re.search(r"module|cluster", p_name, re.IGNORECASE):
                continue

            if edge_weights[p_idx] > 1e-4:
                valid_genes_for_this_mod.append(
                    (p_name, float(edge_weights[p_idx]))
                )

            if len(valid_genes_for_this_mod) >= 4:
                break

        if len(valid_genes_for_this_mod) >= 2:
            selected_modules.append(e_idx)
            module_to_valid_partners[e_idx] = valid_genes_for_this_mod
    selected_modules = [m for m in [5, 6, 13, 14] if m in selected_modules]
    nodes = [TARGET_GENE_NAME]
    sources = []
    targets = []
    values = []

    mod_id_to_node_idx = {}
    for i, e_idx in enumerate(selected_modules):
        mod_name = f"Module_{e_idx}"
        nodes.append(mod_name)
        node_idx = 1 + i
        mod_id_to_node_idx[e_idx] = node_idx
        
        raw_val = float(H_dynamic[gene_idx, e_idx])
        sources.append(0)
        targets.append(node_idx)
        values.append(raw_val)

    gene_to_node_idx = {}
    current_node_count = len(nodes)

    for e_idx in selected_modules:
        mod_node_idx = mod_id_to_node_idx[e_idx]
        partners = module_to_valid_partners[e_idx]
        
        for g_name, g_val in partners:
            if g_name not in gene_to_node_idx:
                nodes.append(g_name)
                gene_to_node_idx[g_name] = current_node_count
                current_node_count += 1
            
            sources.append(mod_node_idx)
            targets.append(gene_to_node_idx[g_name])
            values.append(g_val)

    num_total_nodes = len(nodes)
    node_x = [0.0] * num_total_nodes
    node_y = [0.0] * num_total_nodes
    # Explicitly assign coordinates for the 3-tier Sankey nodes (Left: target gene -> Mid: modules -> Right: partner genes)
    # 1. Target gene node (Left)
    node_x[0] = 0.02
    node_y[0] = 0.50

    # 2. Module nodes 
    n_mods = len(selected_modules)
    mod_y = np.linspace(0.2, 0.75, n_mods)
    for i in range(n_mods):
        node_x[1 + i] = 0.4
        node_y[1 + i] = mod_y[i]

    # 3. Target partner gene nodes 
    n_genes = num_total_nodes - (1 + n_mods)
    gene_y = np.linspace(-0.02, 1.06, n_genes)
    for j in range(n_genes):
        g_idx = 1 + n_mods + j
        node_x[g_idx] = 0.9
        node_y[g_idx] = gene_y[j]

    fig = go.Figure(data=[go.Sankey(
        arrangement = "snap",
        node = dict(
            pad = 50,
            thickness = 16,
            line = dict(color = "#1a1a1a", width = 0.8),
            label = nodes,
            x = node_x,
            y = node_y,
            color = "rgba(46, 204, 113, 0.85)"
        ),
        link = dict(
            source = sources,
            target = targets,
            value = values,
            color = "rgba(52, 152, 219, 0.30)"
        )
    )])

    fig.update_layout(
        title_text = None,
        font = dict(size=26, family="Arial, sans-serif", color="#1a1a1a"),
        width = 1600,
        height = 600,
        margin = dict(l=50, r=50, t=35, b=55),
        paper_bgcolor = 'white',
        plot_bgcolor = 'white'
    )
    
    svg_out = os.path.join(SAVE_DIR, "Figure7B_Sankey_Compensation_Flow.svg")
    png_out = os.path.join(SAVE_DIR, "Figure7B_Sankey_Compensation_Flow.png")
    fig.write_image(svg_out)
    fig.write_image(png_out, scale=4)
    
    print(f"Generated successfully:\n  - {svg_out}\n  - {png_out}")

if __name__ == "__main__":
    generate_sankey()