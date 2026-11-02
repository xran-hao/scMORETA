"""
Script Function: Extracts dynamic hyperedge activation intensities across cell types and computes the Signal-to-Noise Ratio (SNR: peak / median) to quantify lineage-specific regulatory module specificity.
"""
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from model import CSGNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_path = "/home/horizon/scMORETA-main/dataset/pre_data/PBMC_10k_Processed"

model_path = "../result/PBMC_10k_Processed/models/tcsgn_PBMC_10k_Processed_full_model1.pth"
if not os.path.exists(model_path):
    model_path = "../result/PBMC_10k_Processed/models/tcsgn_PBMC_10k_Processed_model1.pth"

def load_label_map(path):
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.size > 1:
        return {i: name for i, name in enumerate(loaded)}
    elif hasattr(loaded, 'item'):
        return loaded.item()
    return loaded

def calculate_contrast():
    print("Loading full-scale expression profiles and labels")
    full_rna = np.load(os.path.join(base_path, 'RNA_matrix.npy'))
    all_labels = np.load(os.path.join(base_path, 'Labels.npy'))
    label_map = load_label_map(os.path.join(base_path, 'label_map.npy'))
    cell_types = [label_map[l] for l in all_labels]

    print(f"Loading model: {model_path}")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    print("Extracting dynamic hyperedge activation intensities for all cells")
    data_tensor = torch.tensor(full_rna, dtype=torch.float32)
    loader = DataLoader(data_tensor, batch_size=256, shuffle=False)
    # Iterate batches and extract mean absolute value of current_dynamic_H from model as activation intensity
    all_h_activation = []
    with torch.no_grad():
        for batch_x in loader:
            _ = model(batch_x.to(device))
            h_act = model.current_dynamic_H.abs().mean(dim=1).cpu().numpy()
            all_h_activation.append(h_act)
            
    full_h_activation = np.vstack(all_h_activation)

    df_h = pd.DataFrame(full_h_activation)
    df_h['cell_type'] = cell_types
    type_fingerprint = df_h.groupby('cell_type').mean()
    # For each lineage fingerprint, calculate the ratio of maximal activation peak to baseline median as SNR
    peak_val = type_fingerprint.max(axis=1)
    median_val = type_fingerprint.median(axis=1)
    snr_ratio = peak_val / (median_val + 1e-12)

    df_summary = pd.DataFrame({
        'Peak': peak_val,
        'Median': median_val,
        'SNR_Fold': snr_ratio
    }).sort_values(by='SNR_Fold', ascending=False)

    print("\n" + "="*75)
    print(f"{'rank':<4} | {'Cell Lineage':<18} | {'Peak':<12} | {'Median':<15} | {'SNR':<12}")
    print("="*75)
    
    for rank, (ctype, row) in enumerate(df_summary.iterrows(), 1):
        print(f"No.{rank:2d} | {ctype:<18} | {row['Peak']:.6f}     | {row['Median']:.6f}        | {row['SNR_Fold']:6.2f} 倍")

    top_lineage = df_summary.index[0]
    top_snr = df_summary.iloc[0]['SNR_Fold']
    top_peak = df_summary.iloc[0]['Peak']
    top_median = df_summary.iloc[0]['Median']

    print("="*75)
    print(f"Final Conclusion: The lineage with the highest activation SNR is: 【{top_lineage}】")
    print(f"Exact data: Peak = {top_peak:.6f}, Median = {top_median:.6f}, SNR = {top_snr:.2f} x")
    print("="*75)

calculate_contrast()