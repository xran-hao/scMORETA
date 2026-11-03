"""
Script Function: Compares predictions from benchmark methods (scMORETA, scJoint, MOGONET, MultiKano), filters the 14 least abundant rare cell subpopulations, and plots a 2x2 normalized recall confusion matrix (Figure 5).
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader, Dataset

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DATA_DIR = os.path.expanduser("~/scMORETA/dataset/pre_data")
BASE_RESULT_DIR = os.path.expanduser("~/scMORETA/result")
OUTPUT_DIR = "Cell_Confusion_Matrices_Top14"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_NAME_MAPPING = {
    "SNARE_seq_Cortex": "SNARE_seq_Cortex",
    "SNARE_seq_Cortex_Processed": "SNARE_seq_Cortex",
    "SNARE_seq_AdBC": "SNARE_seq_AdBC",
    "SNARE_seq_AdBC_Processed": "SNARE_seq_AdBC",
    "SNARE_seq_P0BC": "SNARE_seq_P0BC",
    "SNARE_seq_P0BC_Processed": "SNARE_seq_P0BC",
    "SHARE_seq_Brain": "SHARE_seq_Brain",
    "SHARE_seq_Brain_Processed": "SHARE_seq_Brain",
    "SHARE_seq_SkinA": "SHARE_seq_SkinA",
    "SHARE_seq_SkinA_Processed": "SHARE_seq_SkinA",
    "SHARE_seq_SkinB": "SHARE_seq_SkinB",
    "SHARE_seq_SkinB_Processed": "SHARE_seq_SkinB",
    "MouseKidney_sciCAR": "MouseKidney_sciCAR",
    "MouseKidney_sciCAR_Processed": "MouseKidney_sciCAR",
    "PBMC_10k": "PBMC_10k",
    "PBMC_10k_Processed": "PBMC_10k",
    "Human_Fetal_Brain": "Human_Fetal_Brain",
    "Human_Fetal_Brain_Processed": "Human_Fetal_Brain",
    "BMMC": "BMMC",
    "BMMC_Processed": "BMMC"
}

ALL_DATASETS = [
    "BMMC_Processed",
    "PBMC_10k_Processed",
    "MouseKidney_sciCAR_Processed",
    "SHARE_seq_Brain_Processed",
    "SHARE_seq_SkinA_Processed",
    "SHARE_seq_SkinB_Processed",
    "SNARE_seq_Cortex_Processed",
    "SNARE_seq_AdBC_Processed",
    "SNARE_seq_P0BC_Processed",
    "Human_Fetal_Brain_Processed"
]

MAX_CELL_TYPES = 14

class SimpleDataset(Dataset):
    def __init__(self, rna):
        self.rna = torch.tensor(rna, dtype=torch.float32)
    def __len__(self): 
        return len(self.rna)
    def __getitem__(self, idx): 
        return self.rna[idx]
# Load 5-fold trained model checkpoints iteratively and infer on test splits to obtain unbiased predictions
def get_scmoreta_predictions(data_path, dir_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models_dir = os.path.join(BASE_RESULT_DIR, dir_name, "models")
    
    y_true_all = []
    y_pred_all = []

    for k in range(1, 6):
        fold_dir = os.path.join(data_path, f"fold_{k}")
        test_rna = np.load(os.path.join(fold_dir, "test_rna_2k.npy"))
        test_labels = np.load(os.path.join(fold_dir, "test_labels.npy"))

        candidate_models = [
            os.path.join(models_dir, f"tcsgn_{dir_name}_full_model{k}.pth"),
            os.path.join(models_dir, f"tcsgn_{dir_name}_model{k}.pth")
        ]
        model_path = next((m for m in candidate_models if os.path.exists(m)), None)
        if model_path is None:
            raise FileNotFoundError(f"Model checkpoint not found in {models_dir}")

        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()

        loader = DataLoader(SimpleDataset(test_rna), batch_size=512, shuffle=False)
        with torch.no_grad():
            for x in loader:
                out = model(x.to(device))
                y_pred_all.extend(out.argmax(1).cpu().numpy())
        y_true_all.extend(test_labels)

    return np.array(y_true_all), np.array(y_pred_all)

def split_into_two_lines(name, max_len=14):
    if len(name) <= max_len:
        return name
    spaces = [i for i, c in enumerate(name) if c == ' ']
    if not spaces:
        return name
    mid = len(name) / 2.0
    best_space = min(spaces, key=lambda s: abs(s - mid))
    return name[:best_space] + '\n' + name[best_space+1:]

def plot_four_methods_confusion_matrices(raw_input_name):
    raw_input_name = raw_input_name.replace('.py', '')

    if os.path.exists(os.path.join(BASE_DATA_DIR, raw_input_name)):
        dir_name = raw_input_name
    elif os.path.exists(os.path.join(BASE_DATA_DIR, f"{raw_input_name}_Processed")):
        dir_name = f"{raw_input_name}_Processed"
    else:
        dir_name = raw_input_name

    display_name = DATASET_NAME_MAPPING.get(raw_input_name, DATASET_NAME_MAPPING.get(dir_name, raw_input_name))

    data_path = os.path.join(BASE_DATA_DIR, dir_name)
    res_path = os.path.join(BASE_RESULT_DIR, dir_name)
    preds_dir = os.path.join(res_path, "predictions")

    all_labels = np.load(os.path.join(data_path, "Labels.npy"))
    load_data = np.load(os.path.join(data_path, "label_map.npy"), allow_pickle=True)
    if isinstance(load_data, np.ndarray) and load_data.size > 1:
        label_map = {i: name for i, name in enumerate(load_data)}
    elif hasattr(load_data, 'item'):
        label_map = load_data.item()
    else:
        label_map = load_data

    unique, counts = np.unique(all_labels, return_counts=True)
    total_cells = len(all_labels)

    sorted_indices = np.argsort(counts)
    sorted_unique = unique[sorted_indices]
    sorted_counts = counts[sorted_indices]

    selected_ids = sorted_unique[:MAX_CELL_TYPES].tolist()
    selected_counts = sorted_counts[:MAX_CELL_TYPES].tolist()
    n_selected = len(selected_ids)

    selected_names_with_pct = []
    for uid, cnt in zip(selected_ids, selected_counts):
        pct = (cnt / total_cells) * 100.0
        raw_name = label_map[uid]
        selected_names_with_pct.append(f"{raw_name} ({pct:.2f}%)")

    formatted_labels = [split_into_two_lines(n, max_len=13) for n in selected_names_with_pct]
    # Label harmonization: Parse native predictions/ground-truth for each baseline and map to standard textual cell types using a unified Label Map to prevent indexing misalignment
    method_data = {}

    # 1. scMORETA 
    pred_csv = os.path.join(res_path, "plot_data", f"predictions_{dir_name}.csv")
    if not os.path.exists(pred_csv):
        pred_csv = os.path.join(res_path, f"predictions_{dir_name}.csv")
    
    if os.path.exists(pred_csv):
        df_oof = pd.read_csv(pred_csv)
        method_data['scMORETA (Ours)'] = {
            'y_true': df_oof['True'].values.astype(str),
            'y_pred': df_oof['Predicted'].values.astype(str)
        }

    # 2. scJoint 
    scj_tr_file = os.path.join(preds_dir, "scjoint_all_oof_trues.npy")
    scj_pr_file = os.path.join(preds_dir, "scjoint_all_oof_preds.npy")
    if os.path.exists(scj_pr_file):
        scj_tr = np.load(scj_tr_file)
        scj_pr = np.load(scj_pr_file)
        method_data['scJoint'] = {
            'y_true': np.array([label_map[i] for i in scj_tr]),
            'y_pred': np.array([label_map[i] for i in scj_pr])
        }

    # 3. MOGONET 
    mogo_tr_file = os.path.join(preds_dir, "mogonet_all_oof_trues.npy")
    mogo_pr_file = os.path.join(preds_dir, "mogonet_all_oof_preds.npy")
    if os.path.exists(mogo_pr_file):
        mogo_tr = np.load(mogo_tr_file)
        mogo_pr = np.load(mogo_pr_file)
        unique_classes_sorted = sorted(list(label_map.keys()))
        label_to_clean_str = {c: f"Type_{i:03d}" for i, c in enumerate(unique_classes_sorted)}
        clean_str_to_real_name = {f"Type_{i:03d}": label_map[c] for i, c in enumerate(unique_classes_sorted)}
        
        method_data['MOGONET'] = {
            'y_true': np.array([clean_str_to_real_name[f"Type_{i:03d}"] for i in mogo_tr]),
            'y_pred': np.array([clean_str_to_real_name[f"Type_{i:03d}"] for i in mogo_pr])
        }

    # 4. MultiKano 
    mkano_tr_file = os.path.join(preds_dir, "multikano_all_oof_trues.npy")
    mkano_pr_file = os.path.join(preds_dir, "multikano_all_oof_preds.npy")
    if os.path.exists(mkano_pr_file):
        mkano_tr = np.load(mkano_tr_file)
        mkano_pr = np.load(mkano_pr_file)
        unique_classes_sorted = sorted(list(label_map.keys()))
        clean_str_to_real_name = {f"Type_{i:03d}": label_map[c] for i, c in enumerate(unique_classes_sorted)}
        
        method_data['MultiKano'] = {
            'y_true': np.array([clean_str_to_real_name[f"Type_{i:03d}"] for i in mkano_tr]),
            'y_pred': np.array([clean_str_to_real_name[f"Type_{i:03d}"] for i in mkano_pr])
        }

    all_raw_trues = method_data['scMORETA (Ours)']['y_true']
    unique_names, counts = np.unique(all_raw_trues, return_counts=True)
    total_cells = len(all_raw_trues)

    sorted_indices = np.argsort(counts)
    sorted_names = unique_names[sorted_indices]
    sorted_counts = counts[sorted_indices]

    selected_raw_names = sorted_names[:MAX_CELL_TYPES].tolist()
    selected_counts = sorted_counts[:MAX_CELL_TYPES].tolist()
    n_selected = len(selected_raw_names)

    selected_names_with_pct = []
    for rname, cnt in zip(selected_raw_names, selected_counts):
        pct = (cnt / total_cells) * 100.0
        selected_names_with_pct.append(f"{rname} ({pct:.2f}%)")

    formatted_labels = [split_into_two_lines(n, max_len=13) for n in selected_names_with_pct]

    fig_height = max(15.5, n_selected * 1.25)
    fig, axes = plt.subplots(2, 2, figsize=(20.5, fig_height))
    axes = axes.flatten()

    cmap = sns.color_palette("GnBu", as_cmap=True)
    all_known_names = list(unique_names)

    method_order = ['scMORETA (Ours)', 'scJoint', 'MOGONET', 'MultiKano']
    plot_methods = [m for m in method_order if m in method_data]

    for idx, m_name in enumerate(plot_methods):
        ax = axes[idx]
        cur_y_true = method_data[m_name]['y_true']
        cur_y_pred = method_data[m_name]['y_pred']

        full_cm = confusion_matrix(cur_y_true, cur_y_pred, labels=all_known_names)
        full_row_sums = full_cm.sum(axis=1, keepdims=True)
        full_cm_norm = np.divide(full_cm.astype('float'), full_row_sums, 
                                 out=np.zeros_like(full_cm, dtype=float), where=full_row_sums != 0)

        sub_indices = [all_known_names.index(n) for n in selected_raw_names]
        cm_norm = full_cm_norm[np.ix_(sub_indices, sub_indices)]

        sns.heatmap(
            cm_norm, ax=ax, cmap=cmap, vmin=0.0, vmax=1.0,
            cbar=True,
            cbar_kws={
                'label': 'Recall',
                'fraction': 0.046,
                'pad': 0.03
            },
            linewidths=0.5, linecolor='#f5f5f5'
        )

        for r in range(n_selected):
            for c in range(n_selected):
                val = cm_norm[r, c]
                if val >= 0.005:
                    txt_color = "white" if val > 0.60 else "#222222"
                    ax.text(c + 0.5, r + 0.5, f"{val:.2f}",
                            ha="center", va="center", fontsize=12,
                            fontweight="bold" if r == c else "normal",
                            color=txt_color)

        ax.set_title(m_name, fontsize=13.5, fontweight='bold', pad=12)

        ax.set_xticks(np.arange(n_selected) + 0.5)
        ax.set_xticklabels(formatted_labels, rotation=40, ha='right', fontsize=8.5)
        ax.set_yticks(np.arange(n_selected) + 0.5)
        ax.set_yticklabels(formatted_labels, rotation=0, ha='right', fontsize=8.5)

        ax.set_ylabel("True Cell Types", fontsize=11, labelpad=10)
        ax.set_xlabel("Predicted Cell Types", fontsize=11, labelpad=10)

    for empty_idx in range(len(plot_methods), 4):
        fig.delaxes(axes[empty_idx])

    plt.suptitle(f"Confusion Matrices of Cell Populations - {display_name}", 
                 fontsize=16.0, x=0.5, y=0.02, ha='center', fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.11, left=0.10, right=0.95, hspace=0.45, wspace=0.48)

    out_base = os.path.join(OUTPUT_DIR, f"Figure5_Confusion_Matrices_Top14_{display_name}")
    plt.savefig(f"{out_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{out_base}.svg", format='svg', bbox_inches='tight')
    plt.close()

    print(f"Exported: {out_base}.png and {out_base}.svg")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all', help="Dataset name or 'all'")
    args = parser.parse_args()

    if args.dataset == 'all':
        for d in ALL_DATASETS:
            try:
                plot_four_methods_confusion_matrices(d)
            except Exception as e:
                print(f"Skipping {d} due to error: {e}")
    else:
        plot_four_methods_confusion_matrices(args.dataset)
