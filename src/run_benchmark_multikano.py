"""
Script Function: Standard 5-fold cross-validation runner for the MultiKano benchmark across 10 datasets, including early stopping, dropout resilience stress testing, and rare subpopulation evaluation.
"""
import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import time
import argparse
import random
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import anndata as ad
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from model import MultiOmicsNet, CustomLoss, data_processing
print("Successfully imported official MultiOmicsNet network and CustomLoss via native import")

pathjoin = os.path.join

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
seed_everything(42)

BASE_DATA_DIR = os.path.expanduser("~/scMORETA/dataset/pre_data")
BASE_RESULT_DIR = os.path.expanduser("~/scMORETA/result")

DATASETS = [
    pathjoin(BASE_DATA_DIR, "BMMC_Processed"),
    pathjoin(BASE_DATA_DIR, "PBMC_10k_Processed"),
    pathjoin(BASE_DATA_DIR, "MouseKidney_sciCAR_Processed"),
    pathjoin(BASE_DATA_DIR, "SHARE_seq_Brain_Processed"),
    pathjoin(BASE_DATA_DIR, "SHARE_seq_SkinA_Processed"),
    pathjoin(BASE_DATA_DIR, "SHARE_seq_SkinB_Processed"),
    pathjoin(BASE_DATA_DIR, "SNARE_seq_Cortex_Processed"),
    pathjoin(BASE_DATA_DIR, "SNARE_seq_AdBC_Processed"),
    pathjoin(BASE_DATA_DIR, "SNARE_seq_P0BC_Processed"),
    pathjoin(BASE_DATA_DIR, "Human_Fetal_Brain_Processed")
]

def to_adata_optimized(rna, atac, labels):
    if not sp.issparse(rna):
        rna_sparse = sp.csr_matrix(rna.astype(np.float32))
    else:
        rna_sparse = rna.tocsr().astype(np.float32)

    if not sp.issparse(atac):
        atac_sparse = sp.csr_matrix(atac.astype(np.float32))
    else:
        atac_sparse = atac.tocsr().astype(np.float32)

    n_samples = rna_sparse.shape[0]
    unique_names = [f"cell_{i}" for i in range(n_samples)]

    adata_r = ad.AnnData(X=rna_sparse)
    adata_r.obs_names = unique_names
    adata_r.obs['cell_type'] = labels
    
    adata_a = ad.AnnData(X=atac_sparse)
    adata_a.obs_names = unique_names
    return adata_r, adata_a

def execute_multikano_batch_training(train_r_ad, train_a_ad, test_r_ad, test_a_ad, batch_size=256, epochs=100, stress_test=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_features_tensor, labels_train_tensor, test_features_tensor, test_labels_tensor, \
    first_round_features_tensor, second_round_features_tensor, num_classes = data_processing(
        train_r_ad, test_r_ad, train_a_ad, test_a_ad
    )

    input_size = train_features_tensor.shape[1]
    hidden_size = 256
    initial_lr = 0.001
    T_max = 50
    model = MultiOmicsNet(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)
    criterion = CustomLoss(cross_entropy_weight=1.0, cosine_similarity_weight=0.1)

    train_dataset = TensorDataset(train_features_tensor, labels_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 20

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)

            optimizer.zero_grad()
            outputs = model(bx)
            
            mid_point = len(by) // 2
            loss = criterion(outputs, by, bx[:mid_point], bx[:mid_point])
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'    Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            print(f" Early stopping at epoch {epoch + 1}")
            break

    model.eval()
    test_dataset = TensorDataset(test_features_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    with torch.no_grad():
        for (bx,) in test_loader:
            bx = bx.to(device)
            outputs = model(bx)
            pred_batch = torch.max(outputs, 1)[1]
            all_preds.extend(pred_batch.cpu().numpy())
    # Stress test: Apply Bernoulli dropout masks exclusively to the RNA feature slice of the fused representation to evaluate robustness
    stress_records = []
    if stress_test:
        print(f"Executing Dropout continuous testing")
        dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        rna_feat_dim = first_round_features_tensor.shape[1]

        for gamma in dropout_rates:
            f_pert = test_features_tensor.clone()
            np.random.seed(42)
            mask = torch.tensor(np.random.binomial(n=1, p=(1.0 - gamma), size=(len(test_features_tensor), rna_feat_dim)).astype(np.float32))
            f_pert[:, :rna_feat_dim] *= mask

            pert_loader = DataLoader(TensorDataset(f_pert), batch_size=batch_size, shuffle=False)
            preds_g = []
            with torch.no_grad():
                for (bx,) in pert_loader:
                    preds_g.extend(model(bx.to(device)).argmax(dim=1).cpu().numpy())
            
            f1_g = f1_score(test_labels_tensor.cpu().numpy(), preds_g, average='macro')
            acc_g = accuracy_score(test_labels_tensor.cpu().numpy(), preds_g)
            kappa_g = cohen_kappa_score(test_labels_tensor.cpu().numpy(), preds_g)
            stress_records.append({'Dropout': gamma * 100, 'Macro_F1': f1_g, 'Acc': acc_g, 'Kappa': kappa_g})
            print(f"      Dropout {int(gamma*100):02d}% -> F1: {f1_g:.4f} | Acc: {acc_g:.4f}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.array(all_preds), stress_records

def run_multikano_5fold_eval(data_path, out_root=BASE_RESULT_DIR, stress_test=False):
    if not os.path.exists(data_path):
        base_folder_name = os.path.basename(os.path.normpath(data_path))
        data_path = pathjoin(BASE_DATA_DIR, base_folder_name)

    base_name = os.path.basename(os.path.normpath(data_path))

    print(f"\n" + "=" * 65)
    print(f"Starting full evaluation: {base_name}")
    print(f"Data path: {data_path}")
    print("-" * 65)

    test_metrics = {'Acc': [], 'F1': [], 'Kappa': []}

    preds_out_dir = pathjoin(out_root, base_name, "predictions")
    os.makedirs(preds_out_dir, exist_ok=True)
    stress_all_folds_collector = []  
    all_oof_trues = []
    all_oof_preds = []
    #Iterate through folds, initialize DataLoaders, and train scJoint with early stopping regularization
    for k in range(1, 6):
        seed_everything(42)
        k_fold = k
        print(f"\n-------------------- Fold {k_fold} --------------------")

        fold_dir = pathjoin(data_path, f'fold_{k_fold}')
        if not os.path.exists(fold_dir):
            raise FileNotFoundError(f"Cannot find {fold_dir}, please make sure 5-fold slices are generated for this dataset")

        train_rna = np.load(pathjoin(fold_dir, 'train_rna.npy'))
        train_labels = np.load(pathjoin(fold_dir, 'train_labels.npy'))
        test_rna = np.load(pathjoin(fold_dir, 'test_rna.npy'))
        test_labels = np.load(pathjoin(fold_dir, 'test_labels.npy'))

        if os.path.exists(pathjoin(fold_dir, 'train_atac.npy')):
            train_atac = np.load(pathjoin(fold_dir, 'train_atac.npy'))
            test_atac = np.load(pathjoin(fold_dir, 'test_atac.npy'))
        else:
            train_atac = np.load(pathjoin(fold_dir, 'train_peaks.npy'))
            test_atac = np.load(pathjoin(fold_dir, 'test_peaks.npy'))

        print(f"Sample size: Train set {train_rna.shape[0]} cells | Test set")

        all_lbls = np.concatenate([train_labels, test_labels])
        unique_classes = sorted(list(np.unique(all_lbls)))
        label_to_clean_str = {c: f"Type_{i:03d}" for i, c in enumerate(unique_classes)}
        tr_str_labels = [label_to_clean_str[c] for c in train_labels]
        te_str_labels = [label_to_clean_str[c] for c in test_labels]

        train_r_ad, train_a_ad = to_adata_optimized(train_rna, train_atac, tr_str_labels)
        test_r_ad, test_a_ad = to_adata_optimized(test_rna, test_atac, te_str_labels)

        del train_rna, train_atac, test_rna, test_atac
        gc.collect()

        print(f"Executing MultiKano Fold {k_fold} training and prediction")
        t0 = time.time()
        y_pred, stress_k = execute_multikano_batch_training(train_r_ad, train_a_ad, test_r_ad, test_a_ad, batch_size=256, epochs=100, stress_test=stress_test)
        if stress_k:
            for r in stress_k: r['Fold'] = k_fold; stress_all_folds_collector.append(r)
        cost = time.time() - t0

        le = LabelEncoder()
        le.fit(sorted(list(label_to_clean_str.values())))
        y_true = le.transform(te_str_labels)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        kappa = cohen_kappa_score(y_true, y_pred)

        test_metrics['Acc'].append(acc)
        test_metrics['F1'].append(f1)
        test_metrics['Kappa'].append(kappa)

        all_oof_preds.extend(y_pred)
        all_oof_trues.extend(y_true)

        print(f"Fold {k_fold} completed! Time: {cost:.1f}s | Acc: {acc:.4f} | F1: {f1:.4f} | Kappa: {kappa:.4f}")

        del train_r_ad, train_a_ad, test_r_ad, test_a_ad, y_pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    oof_preds_file = pathjoin(preds_out_dir, "multikano_all_oof_preds.npy")
    oof_trues_file = pathjoin(preds_out_dir, "multikano_all_oof_trues.npy")
    np.save(oof_preds_file, np.array(all_oof_preds))
    np.save(oof_trues_file, np.array(all_oof_trues))
    print(f"MultiKano full-sample 5-fold unbiased predicted labels saved to: {preds_out_dir}")

    final_res = {k: np.mean(v) for k, v in test_metrics.items()}
    print(f" [{base_name}] MultiKano 5-Fold Final Performance Report:")
    for metric_name, val in final_res.items():
        print(f"  • {metric_name}: {val:.4f}")

    out_dir = pathjoin(out_root, base_name, "plot_data")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = pathjoin(out_dir, "multikano_final_metrics_summary.csv")

    with open(summary_path, 'w') as f:
        header = "Dataset," + ",".join(final_res.keys())
        values = f"{base_name}," + ",".join([f"{v:.4f}" for v in final_res.values()])
        f.write(header + "\n")
        f.write(values + "\n")

    print(f"MultiKano complete metrics saved to: {summary_path}")
    if stress_test and len(stress_all_folds_collector) > 0:
        total_stress_df = pd.DataFrame(stress_all_folds_collector)
        summary_stress = total_stress_df.groupby('Dropout').agg({
            'Macro_F1': ['mean', 'std'],
            'Acc': ['mean', 'std'],
            'Kappa': ['mean', 'std']
        }).reset_index()

        stress_summary_path = pathjoin(out_dir, "multikano_stress_results_summary.csv")
        summary_stress.to_csv(stress_summary_path, index=False)
        print("\n" + "=" * 65)
        print("MultiKano Overall 5-Fold Stress Summary Completed")
        print(f"Overall mean and standard deviation saved to: {stress_summary_path}")
        print("=" * 65)
def get_parser():
    parser = argparse.ArgumentParser(description="MultiKano 5-Fold Benchmark Runner (Zero-Modification on model.py)")
    parser.add_argument('-all', action='store_true', help='Enable this to automatically run all 10 datasets sequentially')
    parser.add_argument('-expr', type=str, default=pathjoin(BASE_DATA_DIR, "BMMC_Processed"), help='Path to a single dataset')
    parser.add_argument('-outdir', type=str, default=BASE_RESULT_DIR, help='Output root directory')
    parser.add_argument('--stress_test', action='store_true', help='Enable multi-gradient transcriptome dropout stress test for resilience')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if args.all:
        print("=" * 70)
        print(f"Starting MultiKano automatic batch evaluation for {len(DATASETS)} datasets in total")
        print("=" * 70)
        for i, d in enumerate(DATASETS, 1):
            if os.path.exists(d):
                try:
                    run_multikano_5fold_eval(d, out_root=args.outdir, stress_test=args.stress_test)
                    time.sleep(3)
                except Exception as e:
                    print(f"Dataset {d} failed with error: {e}")
                    continue
            else:
                print(f"Skipping non-existent dataset: {d}")
        print("MultiKano evaluation for all datasets completed successfully")
    else:
        run_multikano_5fold_eval(args.expr, out_root=args.outdir, stress_test=args.stress_test)