import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import copy
import random
import gc
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from model import CSGNet, SupConLoss
from datasets import MyDataset
from gene_mapper import get_global_adj
from plot_utils import plot_learning_curve, plot_gene_skeleton

pathjoin = os.path.join

print("Is GPU available:", torch.cuda.is_available())
print("Current device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

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

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    #parser.add_argument('-expr', type=str, default='../dataset/pre_data/BMMC_Processed', help='Input directory')
    #parser.add_argument('-expr', type=str, default='../dataset/pre_data/PBMC_10k_Processed', help='Input directory')
    parser.add_argument('-expr', type=str, default='../dataset/pre_data/MouseKidney_sciCAR_Processed', help='Input directory')
    #parser.add_argument('-expr', type=str, default='../dataset/pre_data/SHARE_seq_Brain_Processed', help='Input directory')
    #parser.add_argument('-expr', type=str, default='../dataset/pre_data/SHARE_seq_SkinA_Processed', help='Input directory')
    #parser.add_argument('-expr', type=str, default='../dataset/pre_data/SHARE_seq_SkinB_Processed', help='Input directory')
    #parser.add_argument('-expr', type=str, default='../dataset/pre_data/SNARE_seq_Cortex_Processed', help='Input directory')
    #parser.add_argument('-expr', type=str, default='../dataset/pre_data/SNARE_seq_AdBC_Processed', help='Input directory')
    #parser.add_argument('-expr', type=str, default='../dataset/pre_data/SNARE_seq_P0BC_Processed', help='Input directory')
    #parser.add_argument('-expr', type=str, default='../dataset/pre_data/Human_Fetal_Brain_Processed', help='Input directory')
    
    parser.add_argument('-outdir', type=str, default='../result', help='Output root directory')

    parser.add_argument('-cuda', '--cuda', type=bool, default=True)
    parser.add_argument('-bs', '--batch_size', type=int, default=1024)
    parser.add_argument('-epoch', '--epoch', type=int, default=100)
    parser.add_argument('-hvgs', '--high_var_genes', type=int, default=2000)
    parser.add_argument('-ca', '--csn_alpha', type=float, default=0.01)
    parser.add_argument('-addname', type=str, default="")
    parser.add_argument('-nsp', '--n_splits', type=int, default=5)

    parser.add_argument('-res', '--resolution', type=int, default=35)
    parser.add_argument('-ov', '--overlap', type=float, default=0.5)
    parser.add_argument('-eps', '--eps', type=float, default=0.8)

    parser.add_argument('-clist', '--channel_list', nargs='+', type=int, default=[256, 64])
    parser.add_argument('-mc', '--mid_channel', type=int, default=16)
    parser.add_argument('-gcdim1', '--global_conv1_dim', type=int, default=12)
    parser.add_argument('-gcdim2', '--global_conv2_dim', type=int, default=4)
    parser.add_argument('-dr', '--dropout_ratio', type=float, default=0.2)

    return parser
# Runs one training epoch, jointly optimizing Cross-Entropy and Supervised Contrastive losses.
def train(model, optimizer, train_loader, device, loss_fn, supcon_fn=None):
    model.train()
    loss_all = 0
    iters = len(train_loader)
    y_pred = []
    y_true = []

    for idx, (data_x, data_y) in enumerate(train_loader):
        data_x = data_x.to(device)
        data_y = data_y.to(device)

        optimizer.zero_grad()
        latent, output = model(data_x, get_latent_variable=True)
        
        loss_ce = loss_fn(output, data_y)
        if supcon_fn is not None:
            loss_con = supcon_fn(latent, data_y)
            loss = loss_ce + 0.04 * loss_con
        else:
            loss = loss_ce

        loss.backward()
        optimizer.step()
        loss_all += loss.item()
        with torch.no_grad():
            pred = output.argmax(dim=-1).detach().cpu().numpy()
            y = data_y.detach().cpu().numpy()
            y_pred.extend(pred)
            y_true.extend(y)

    avg_loss = loss_all / iters if iters > 0 else 0
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return avg_loss, acc, f1, kappa

def classify_train(args):
    expr_path = args.expr
    base_name = os.path.basename(os.path.normpath(expr_path)).replace('.npz', '')

    dataset_outdir = pathjoin(args.outdir, base_name)

    models_folder = pathjoin(dataset_outdir, 'models')
    os.makedirs(models_folder, exist_ok=True)
    train_dir = pathjoin(dataset_outdir, 'train_data')
    os.makedirs(train_dir, exist_ok=True)

    args.outdir = dataset_outdir
    print(f" [Train] : {base_name}")
    print(f"Output directory: {dataset_outdir}")

    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    if os.path.isdir(expr_path):
        print("Loading multi-omics 2,000 gene data and labels")
        try:
            raw_rna = np.load(pathjoin(expr_path, 'RNA_matrix.npy'))
            label = np.load(pathjoin(expr_path, 'Labels.npy'))

            atac_path = pathjoin(expr_path, 'ATAC_gene_activity.npy')
            if os.path.exists(atac_path):
                print("Found ATAC gene activity data (2k), will be used to construct scaffold.")
                raw_atac = np.load(atac_path)
            else:
                print("ATAC data not found, falling back to using only RNA for scaffold construction.")
                raw_atac = None

            gene_df = pd.read_csv(pathjoin(expr_path, 'gene_names.csv'), header=None)
            feature_names = gene_df.iloc[:, 0].values.astype(str)

            map_path = pathjoin(expr_path, 'label_map.npy')
            if os.path.exists(map_path):
                str_labels_list = np.load(map_path, allow_pickle=True)
                str_labels = str_labels_list[label]
            else:
                str_labels = label.astype(str)

            seq_dict = {'label': label, 'str_labels': str_labels, 'feature_name': feature_names}

            index_save_path = pathjoin(expr_path, 'cv_indices.npz')
            if os.path.exists(index_save_path):
                print(f"Existing index file found, loading: {index_save_path}")
                seq_dict = dict(np.load(index_save_path, allow_pickle=True))
            else:
                print("Index file not found, generating 5-Fold stratified split on the fly")
                # Generate 5-fold stratified cross-validation splits to ensure balanced cell-type distributions
                skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
                for k, (tr_idx, te_idx) in enumerate(skf.split(raw_rna, label)):
                    seq_dict[f'train_index_{k+1}'] = tr_idx
                    seq_dict[f'test_index_{k+1}'] = te_idx
                np.savez(index_save_path, **seq_dict)

            if not os.path.exists(pathjoin(expr_path, 'fold_1')):
                print("Performing synchronized slicing for 5-fold cross-validation")
                full_rna_path = pathjoin(expr_path, 'RNA_matrix_full.npy')
                has_full_rna = os.path.exists(full_rna_path)

                if has_full_rna:
                    print("Full-scale data successfully detected, MultiKano will allocate full features")
                    raw_rna_full = np.load(full_rna_path, mmap_mode='r')
                else:
                    print("Full-scale file not detected, MultiKano will use 2k features.")

                for k in range(1, args.n_splits + 1):
                    tr_idx = seq_dict[f'train_index_{k}']
                    te_idx = seq_dict[f'test_index_{k}']
                    fold_dir = pathjoin(expr_path, f'fold_{k}')
                    os.makedirs(fold_dir, exist_ok=True)

                    tr_lbl = label[tr_idx]
                    te_lbl = label[te_idx]
                    np.save(pathjoin(fold_dir, 'train_labels.npy'), tr_lbl)
                    np.save(pathjoin(fold_dir, 'test_labels.npy'), te_lbl)

                    tr_rna_2k = raw_rna[tr_idx]
                    te_rna_2k = raw_rna[te_idx]
                    tr_atac_2k = raw_atac[tr_idx] if raw_atac is not None else tr_rna_2k
                    te_atac_2k = raw_atac[te_idx] if raw_atac is not None else te_rna_2k

                    np.save(pathjoin(fold_dir, 'train_rna_2k.npy'), tr_rna_2k)
                    np.save(pathjoin(fold_dir, 'test_rna_2k.npy'), te_rna_2k)
                    np.save(pathjoin(fold_dir, 'train_atac.npy'), tr_atac_2k)
                    np.save(pathjoin(fold_dir, 'test_atac.npy'), te_atac_2k)

                    sp.save_npz(pathjoin(fold_dir, 'scjoint_tr_rna.npz'), sp.csr_matrix(tr_rna_2k))
                    sp.save_npz(pathjoin(fold_dir, 'scjoint_te_rna.npz'), sp.csr_matrix(te_rna_2k))
                    sp.save_npz(pathjoin(fold_dir, 'scjoint_tr_atac.npz'), sp.csr_matrix(tr_atac_2k))
                    sp.save_npz(pathjoin(fold_dir, 'scjoint_te_atac.npz'), sp.csr_matrix(te_atac_2k))
                    np.savetxt(pathjoin(fold_dir, 'scjoint_tr_lbl.txt'), tr_lbl, fmt='%d')
                    np.savetxt(pathjoin(fold_dir, 'scjoint_te_lbl.txt'), te_lbl, fmt='%d')

                    np.save(pathjoin(fold_dir, 'train_rna.npy'), np.array(raw_rna_full[tr_idx]))
                    np.save(pathjoin(fold_dir, 'test_rna.npy'), np.array(raw_rna_full[te_idx]))
                    np.save(pathjoin(fold_dir, 'train_atac.npy'), tr_atac_2k)
                    np.save(pathjoin(fold_dir, 'test_atac.npy'), te_atac_2k)

                    print(f" fold {k} synchronized slicing completed  ")

                if has_full_rna:
                    del raw_rna_full
                    gc.collect()

                print("All 5-fold data synchronized generation completed")
            else:
                print("Existing 5-fold sliced data detected, loading directly for use")

        except Exception as e:
            raise RuntimeError(f"error: {e}")

    if raw_rna.dtype != np.float32: 
        raw_rna = raw_rna.astype(np.float32)
    if raw_atac is not None and raw_atac.dtype != np.float32:
        raw_atac = raw_atac.astype(np.float32)

    current_supcon = SupConLoss(temperature=0.01).to(device)

    class_num = len(np.unique(label))
    gene_num = raw_rna.shape[1]
    print(f"data is ready: {raw_rna.shape[0]} cells, {gene_num} genes, {class_num} celltypes")

    counts = np.array([np.sum(label == x) for x in np.unique(label)])
    weights = np.clip(np.max(counts) / (counts + 1e-6), 1, 50)
    weights = weights / np.sum(weights)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights).float().to(device))

    for k in range(args.n_splits):
        k_fold = k + 1
        print(f"\n==================== Start Fold {k_fold} ====================")
        train_idx = seq_dict[f'train_index_{k_fold}']
        
        X_train_rna = raw_rna[train_idx]
        y_train = label[train_idx]
        if raw_atac is not None:
            X_train_skeleton = raw_atac[train_idx]
        else:
            X_train_skeleton = X_train_rna

        global_adj = get_global_adj(
            X_train_skeleton,
            resolution=args.resolution,
            overlap=args.overlap,
            eps=args.eps
        )

        adj_save_path = pathjoin(models_folder, f'global_adj_{base_name}_full_{k_fold}.pt')
        torch.save(global_adj, adj_save_path)
        global_adj = global_adj.to(device)

        sk_save_path = pathjoin(train_dir, f'Gene_Skeleton_full_fold{k_fold}.png')
        try:
            plot_gene_skeleton(global_adj.cpu(), feature_names, sk_save_path, top_k=1000)
            plt.close('all')
        except Exception:
            plt.close('all')

        train_dataset = MyDataset(X_train_rna, y_train)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            drop_last=True,
            num_workers=0,
            pin_memory=False
        )

        model = CSGNet(
            in_channel=1,
            mid_channel=args.mid_channel,
            num_nodes=gene_num,
            out_channel=class_num,
            dropout_ratio=args.dropout_ratio,
            channel_list=copy.deepcopy(args.channel_list),
            global_conv1_dim=args.global_conv1_dim,
            global_conv2_dim=args.global_conv2_dim,
            global_adj=global_adj
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_kappa': []}
        best_loss = float('inf')
        patience_counter = 0
        patience_limit = 20 
        
        for epoch in range(1, args.epoch + 1):
            t_loss, t_acc, t_f1, t_kappa = train(
                model, optimizer, train_loader, device, loss_fn, supcon_fn=current_supcon
            )
            scheduler.step()

            history['epoch'].append(epoch)
            history['train_loss'].append(t_loss)
            history['train_acc'].append(t_acc)
            history['train_f1'].append(t_f1)
            history['train_kappa'].append(t_kappa)

            if epoch % 5 == 0 or epoch == 1:
                print(f"    Epoch {epoch:3d}: Loss={t_loss:.4f}, Acc={t_acc:.4f}, F1={t_f1:.4f}, Kappa={t_kappa:.4f}")
            if t_loss < best_loss:
                best_loss = t_loss
                patience_counter = 0
                
                save_model_name = f'tcsgn_{base_name}_full_model{args.addname}{k_fold}.pth'
                torch.save(model, pathjoin(models_folder, save_model_name))
            else:
                patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping at epoch {epoch}")
                break

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(pathjoin(train_dir, f'training_log_full_fold{k_fold}.csv'), index=False)
        try:
            plot_learning_curve(hist_df, pathjoin(train_dir, f'Learning_Curve_full_fold{k_fold}.png'))
            plt.close('all')
        except Exception:
            plt.close('all')

        print(f"Fold {k_fold} is over")

if __name__ == '__main__':
    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()
    print('Args:', args)
    classify_train(args)
    end_time = time.time()