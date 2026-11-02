"""
Script Function: Dedicated benchmark of T-CSGN on rare cell types (abundance < 3%). Evaluates 5-fold Out-of-Fold (OOF) unbiased predictions to quantify Precision, Recall, and F1-Score for each rare subpopulation.
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

import matplotlib
matplotlib.use('Agg')

DATASETS = [
    "BMMC_Processed",
    "Human_Fetal_Brain_Processed",
    "MouseKidney_sciCAR_Processed",
    "PBMC_10k_Processed",
    "SHARE_seq_Brain_Processed",
    "SHARE_seq_SkinA_Processed",
    "SHARE_seq_SkinB_Processed",
    "SNARE_seq_AdBC_Processed",
    "SNARE_seq_Cortex_Processed",
    "SNARE_seq_P0BC_Processed",
]

RARE_THRESHOLD = 0.03  

class MyDataset(Dataset):
    def __init__(self, rna, labels):
        self.rna = torch.tensor(rna, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.rna[idx], self.labels[idx]
# Profile cell-type frequencies, identify rare subpopulations (< 3%), and evaluate concatenated 5-fold model predictions
def analyze_single_dataset(dataset_name, data_root, result_root, device):
    print("\n" + "=" * 75)
    print(f"Analyzing dataset: 【{dataset_name}】")
    print("=" * 75)

    base_path = os.path.join(data_root, dataset_name)
    dataset_res_dir = os.path.join(result_root, dataset_name)
    models_dir = os.path.join(dataset_res_dir, "models")
    os.makedirs(dataset_res_dir, exist_ok=True)

    labels_file = os.path.join(base_path, "Labels.npy")
    label_map_file = os.path.join(base_path, "label_map.npy")
    if not os.path.exists(labels_file) or not os.path.exists(label_map_file):
        print(f"Skip: Label file not found for {dataset_name}.")
        return None

    all_labels = np.load(labels_file)
    load_data = np.load(label_map_file, allow_pickle=True)

    if isinstance(load_data, np.ndarray) and load_data.size > 1:
        label_map = {i: name for i, name in enumerate(load_data)}
    elif hasattr(load_data, 'item'):
        label_map = load_data.item()
    else:
        label_map = load_data

    unique, counts = np.unique(all_labels, return_counts=True)
    df_dist = pd.DataFrame({
        'id': unique,
        'name': [label_map.get(i, f"Class_{i}") for i in unique],
        'count': counts,
        'pct': counts / len(all_labels)
    })

    rare_df = df_dist[df_dist['pct'] < RARE_THRESHOLD]
    rare_ids = rare_df['id'].tolist()

    print(f"Total cells: {len(all_labels)}, Classes: {len(unique)}")
    if len(rare_ids) == 0:
        print(f"No rare cells with proportion < {RARE_THRESHOLD*100}% found in this dataset, skipping analysis")
        return None

    print(f"Identified {len(rare_ids)} rare cell types:")
    for _, row in rare_df.iterrows():
        print(f"{row['name']}: {row['count']} cells ({row['pct']*100:.2f}%)")

    y_true_all = []
    y_pred_all = []
    # Loop over 5 folds, load trained model checkpoints, and collect unbiased predictions on out-of-fold test sets
    for k in range(1, 6):
        fold_dir = os.path.join(base_path, f"fold_{k}")
        test_rna_path = os.path.join(fold_dir, "test_rna_2k.npy")
        test_labels_path = os.path.join(fold_dir, "test_labels.npy")

        candidate_models = [
            os.path.join(models_dir, f"tcsgn_{dataset_name}_model{k}.pth"),
            os.path.join(models_dir, f"tcsgn_{dataset_name}_full_model{k}.pth"),
            os.path.join(models_dir, f"tcsgn_{dataset_name}_model_{k}.pth")
        ]
        model_path = next((m for m in candidate_models if os.path.exists(m)), None)

        if not os.path.exists(test_rna_path) or model_path is None:
            print(f"error")
            continue

        test_rna = np.load(test_rna_path)
        test_labels = np.load(test_labels_path)
        test_loader = DataLoader(MyDataset(test_rna, test_labels), batch_size=256, shuffle=False)

        model = torch.load(model_path, map_location=device, weights_only=False)
        model.ablation_mode = 'full'
        model.to(device)
        model.eval()

        with torch.no_grad():
            for x, y in test_loader:
                out = model(x.to(device))
                y_pred_all.extend(out.argmax(1).cpu().numpy())
                y_true_all.extend(y.numpy())

    if len(y_true_all) == 0:
        print(f"Dataset {dataset_name} failed to complete prediction for any fold, skipping")
        return None

    report = classification_report(y_true_all, y_pred_all, output_dict=True, zero_division=0)
    rare_metrics = []
    for rid in rare_ids:
        key = str(rid)
        if key in report:
            metrics = report[key]
            rare_metrics.append({
                'Dataset': dataset_name,
                'CellType': label_map.get(rid, f"Class_{rid}"),
                'Precision': round(metrics['precision'], 4),
                'Recall': round(metrics['recall'], 4),
                'F1-Score': round(metrics['f1-score'], 4),
                'Support': int(metrics['support']),
                'Ratio(%)': round(metrics['support'] / len(y_true_all) * 100, 2)
            })

    df_res = pd.DataFrame(rare_metrics)
    if df_res.empty:
        return None

    csv_save_path = os.path.join(dataset_res_dir, "tcsgn_rare_cell_results.csv")
    df_res.to_csv(csv_save_path, index=False)
    print(f"Rare cell CSV results saved to: {csv_save_path}")

    plt.figure(figsize=(max(8, len(df_res) * 0.9), 6))
    ax = sns.barplot(data=df_res, x='CellType', y='F1-Score', palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title(f"TCSGN Performance on Rare Cell Types (<{int(RARE_THRESHOLD*100)}%) - {dataset_name}")
    plt.ylim(0, 1.15)

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.3f}",
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    xytext=(0, 4), textcoords='offset points', fontsize=9)

    plt.tight_layout()
    fig_save_path = os.path.join(dataset_res_dir, "tcsgn_rare_cell_results.png")
    plt.savefig(fig_save_path, dpi=300)
    plt.close('all')
    print(f"High-resolution barplot saved to: {fig_save_path}")

    return df_res

def main():
    parser = argparse.ArgumentParser(description="TCSGN Automated Rare Cell Evaluation on 10 Single-Cell Datasets")
    parser.add_argument('--dataset', type=str, default='all', choices=['all'] + DATASETS,
                        help="Specify dataset name to evaluate, default 'all' traverses all 10 datasets")
    parser.add_argument('--data_root', type=str, default="../dataset/pre_data", help="Preprocessed data root directory")
    parser.add_argument('--result_root', type=str, default="../result", help="Results save root directory")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Current compute device: {device}")

    target_datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    all_summary_list = []

    for name in target_datasets:
        res_df = analyze_single_dataset(name, args.data_root, args.result_root, device)
        if res_df is not None:
            all_summary_list.append(res_df)

    if len(all_summary_list) > 1:
        total_summary_df = pd.concat(all_summary_list, ignore_index=True)
        summary_csv_path = os.path.join(args.result_root, "all_datasets_rare_cell_summary.csv")
        total_summary_df.to_csv(summary_csv_path, index=False)


if __name__ == "__main__":
    main()