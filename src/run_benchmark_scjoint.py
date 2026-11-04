import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report
import random

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

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from util.model_regress import Net_encoder, Net_cell
print("Successfully imported official Net_encoder and Net_cell via native import")

DATASETS = [
    "../scMORETA/dataset/pre_data/BMMC_Processed",
    "../scMORETA/dataset/pre_data/PBMC_10k_Processed",
    "../scMORETA/dataset/pre_data/MouseKidney_sciCAR_Processed",
    "../scMORETA/dataset/pre_data/SHARE_seq_Brain_Processed",
    "../scMORETA/dataset/pre_data/SHARE_seq_SkinA_Processed",
    "../scMORETA/dataset/pre_data/SHARE_seq_SkinB_Processed",
    "../scMORETA/dataset/pre_data/SNARE_seq_Cortex_Processed",
    "../scMORETA/dataset/pre_data/SNARE_seq_AdBC_Processed",
    "../scMORETA/dataset/pre_data/SNARE_seq_P0BC_Processed",
    "../scMORETA/dataset/pre_data/Human_Fetal_Brain_Processed"
]

def run_scjoint_5fold_eval(data_path, out_root="../scMORETA/result", epochs=100, batch_size=32, lr=1e-4, stress_test=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_name = os.path.basename(os.path.normpath(data_path))

    print(f"\n" + "=" * 65)
    print(f"[scJoint Official Model] Starting evaluation:: {base_name}")
    print(f"Data path: {data_path}")
    print("-" * 65)

    test_metrics = {'Acc': [], 'F1': [], 'Kappa': []}

    preds_out_dir = pathjoin(out_root, base_name, "predictions")
    os.makedirs(preds_out_dir, exist_ok=True)

    all_oof_trues = []
    all_oof_preds = []
    stress_all_folds_collector = [] 

    for k in range(1, 6):
        seed_everything(42)  
        k_fold = k
        print(f"\n--- Start Fold {k_fold} ---")

        fold_dir = pathjoin(data_path, f'fold_{k_fold}')
        if not os.path.exists(fold_dir):
            raise FileNotFoundError(f"Cannot find {fold_dir}, please make sure 5-fold slices are generated in scMORETA")

        if os.path.exists(pathjoin(fold_dir, 'train_rna.npy')):
            X_train = np.load(pathjoin(fold_dir, 'train_rna.npy')).astype(np.float32)
            y_train = np.load(pathjoin(fold_dir, 'train_labels.npy')).astype(np.int64)
            X_test = np.load(pathjoin(fold_dir, 'test_rna.npy')).astype(np.float32)
            y_test = np.load(pathjoin(fold_dir, 'test_labels.npy')).astype(np.int64)
        else:
            X_train = sp.load_npz(pathjoin(fold_dir, 'scjoint_tr_rna.npz')).toarray().astype(np.float32)
            y_train = np.loadtxt(pathjoin(fold_dir, 'scjoint_tr_lbl.txt'), dtype=int)
            X_test = sp.load_npz(pathjoin(fold_dir, 'scjoint_te_rna.npz')).toarray().astype(np.float32)
            y_test = np.loadtxt(pathjoin(fold_dir, 'scjoint_te_lbl.txt'), dtype=int)

        num_classes = len(np.unique(np.concatenate([y_train, y_test])))
        input_dim = X_train.shape[1]

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
            batch_size=batch_size, shuffle=True, drop_last=True
        )
        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
            batch_size=batch_size, shuffle=False
        )

        encoder = Net_encoder(input_size=input_dim).to(device)
        classifier = Net_cell(num_of_class=num_classes).to(device)
        optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        best_loss = float('inf')
        patience_counter = 0
        patience_limit = 20

        for epoch in range(1, epochs + 1):
            encoder.train()
            classifier.train()
            loss_all = 0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                embedding = encoder(bx)
                logits = classifier(embedding)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                loss_all += loss.item()

            avg_loss = loss_all / len(train_loader)
            if epoch % 20 == 0 or epoch == 1:
                print(f"  Epoch [{epoch:03d}/{epochs}]: Loss = {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        encoder.eval()
        classifier.eval()

        if stress_test:
            print(f"[scJoint Stress Test Mode Activated] Starting to evaluate Dropout gradients")
            dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            stress_fold_k = []
            
            for gamma in dropout_rates:
                np.random.seed(42 + k_fold)
                mask = np.random.binomial(n=1, p=(1.0 - gamma), size=X_test.shape).astype(np.float32)
                X_test_pert = X_test * mask
                pert_loader = DataLoader(TensorDataset(torch.tensor(X_test_pert), torch.tensor(y_test)), batch_size=batch_size, shuffle=False)
                
                preds_g = []
                with torch.no_grad():
                    for bx, by in pert_loader:
                        preds_g.extend(classifier(encoder(bx.to(device))).argmax(dim=-1).cpu().numpy())
                
                f1_g = f1_score(y_test, preds_g, average='macro')
                acc_g = accuracy_score(y_test, preds_g)
                kappa_g = cohen_kappa_score(y_test, preds_g)
                
                record = {'Fold': k_fold, 'Dropout': gamma * 100, 'Macro_F1': f1_g, 'Acc': acc_g, 'Kappa': kappa_g}
                stress_fold_k.append(record)
                stress_all_folds_collector.append(record)
                print(f"    Dropout {int(gamma*100):02d}% -> F1: {f1_g:.4f} | Acc: {acc_g:.4f}")
            
            preds = []
            with torch.no_grad():
                for bx, by in test_loader:
                    preds.extend(classifier(encoder(bx.to(device))).argmax(dim=-1).cpu().numpy())
        else:
            preds = []
            with torch.no_grad():
                for bx, by in test_loader:
                    bx = bx.to(device)
                    embedding = encoder(bx)
                    logits = classifier(embedding)
                    pred = logits.argmax(dim=-1).cpu().numpy()
                    preds.extend(pred)

        preds = np.array(preds)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro')
        kappa = cohen_kappa_score(y_test, preds)

        test_metrics['Acc'].append(acc)
        test_metrics['F1'].append(f1)
        test_metrics['Kappa'].append(kappa)
        print(f"Fold {k_fold} Result: Acc = {acc:.4f}, F1 = {f1:.4f}, Kappa = {kappa:.4f}")

        all_oof_preds.extend(preds)
        all_oof_trues.extend(y_test)

    np.save(pathjoin(preds_out_dir, "scjoint_all_oof_preds.npy"), np.array(all_oof_preds))
    np.save(pathjoin(preds_out_dir, "scjoint_all_oof_trues.npy"), np.array(all_oof_trues))
    print(f"scJoint full-sample predicted labels saved to:\n {preds_out_dir}")

    final_res = {k: np.mean(v) for k, v in test_metrics.items()}
    print(f"\n--- Final Average Results ---")
    for k, v in final_res.items():
        print(f"  {k}: {v:.4f}")

    out_dir = pathjoin(out_root, base_name, "plot_data")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = pathjoin(out_dir, "scjoint_final_metrics_summary.csv")

    with open(summary_path, 'w') as f:
        header = "Dataset," + ",".join(final_res.keys())
        values = f"{base_name}," + ",".join([f"{v:.4f}" for v in final_res.values()])
        f.write(header + "\n")
        f.write(values + "\n")

    print(f"scJoint results saved to: {summary_path}")

    if stress_test and len(stress_all_folds_collector) > 0:
        total_stress_df = pd.DataFrame(stress_all_folds_collector)
        summary_stress = total_stress_df.groupby('Dropout').agg({
            'Macro_F1': ['mean', 'std'],
            'Acc': ['mean', 'std'],
            'Kappa': ['mean', 'std']
        }).reset_index()

        stress_summary_path = pathjoin(out_dir, "scjoint_stress_results_summary.csv")
        summary_stress.to_csv(stress_summary_path, index=False)
        print("\n" + "=" * 65)
        print("[scJoint Overall 5-Fold Stress Summary Completed]")
        print(f"Overall mean and standard deviation saved to: {stress_summary_path}")
        print("=" * 65)

def get_parser():
    parser = argparse.ArgumentParser(description="Official scJoint 5-Fold Benchmark Runner")
    parser.add_argument('-all', action='store_true', help='Enable this to automatically run all 10 datasets')
    parser.add_argument('-expr', type=str, default='../scMORETA/dataset/pre_data/PBMC_10k_Processed', help='Path to a single dataset')
    parser.add_argument('-outdir', type=str, default='../scMORETA/result', help='Output root directory')
    parser.add_argument('-epoch', type=int, default=100, help='Number of training epochs')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--stress_test', action='store_true', help='Enable multi-gradient transcriptome dropout stress test for resilience')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if args.all:
        print("-" * 70)
        print(f"Starting automatic batch evaluation for {len(DATASETS)} datasets in total")
        print("-" * 70)
        for i, d in enumerate(DATASETS, 1):
            if os.path.exists(d):
                try:
                    run_scjoint_5fold_eval(d, out_root=args.outdir, epochs=args.epoch, batch_size=args.batch_size, stress_test=args.stress_test)
                    time.sleep(2)
                except Exception as e:
                    print(f"Dataset {d} failed with error: {e}")
                    continue
            else:
                print(f"Skipping non-existent dataset: {d}")
        print("\nOfficial scJoint benchmark evaluation for all 10 datasets completed successfully")
    else:
        run_scjoint_5fold_eval(args.expr, out_root=args.outdir, epochs=args.epoch, batch_size=args.batch_size, stress_test=args.stress_test)
        
        