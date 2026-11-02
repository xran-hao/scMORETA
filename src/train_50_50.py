"""
Script Function: Model training pipeline for the 50% train / 50% test extreme generalization experiment. Constructs the topological hypergraph strictly on 50% training cells, jointly optimizes CE and SupCon losses, and saves the best model checkpoint and adjacency matrix.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import copy
import random
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

from model import CSGNet, SupConLoss
from datasets import MyDataset
from gene_mapper import get_global_adj

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
base_data_dir = os.path.expanduser("~/scMORETA-main/dataset/pre_data/PBMC_10k_Processed")
output_dir = os.path.expanduser("~/scMORETA-main/result/PBMC_10k_Processed_50_50")
models_dir = pathjoin(output_dir, 'models')
os.makedirs(models_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, optimizer, train_loader, loss_fn, supcon_fn):
    model.train()
    loss_all = 0.0
    y_pred, y_true = [], []
    for data_x, data_y in train_loader:
        data_x, data_y = data_x.to(device), data_y.to(device)
        optimizer.zero_grad()
        latent, output = model(data_x, get_latent_variable=True)
        loss = loss_fn(output, data_y) + 0.04 * supcon_fn(latent, data_y)
        loss.backward()
        optimizer.step()
        loss_all += loss.item()
        
        with torch.no_grad():
            y_pred.extend(output.argmax(dim=-1).cpu().numpy())
            y_true.extend(data_y.cpu().numpy())
            
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return loss_all / len(train_loader), acc, f1

def main():
    print("=" * 60)
    raw_rna = np.load(pathjoin(base_data_dir, 'RNA_matrix.npy')).astype(np.float32)
    labels = np.load(pathjoin(base_data_dir, 'Labels.npy')).astype(np.int64)
    atac_path = pathjoin(base_data_dir, 'ATAC_gene_activity.npy')
    raw_atac = np.load(atac_path).astype(np.float32) if os.path.exists(atac_path) else None
    # Perform Stratified Shuffle Split to partition data into 50% train and 50% test sets, saving partition indices
    split_file = pathjoin(output_dir, 'split_indices_50_50.npz')
    if os.path.exists(split_file):
        split_data = np.load(split_file)
        tr_idx, te_idx = split_data['train_idx'], split_data['test_idx']
    else:
        print("(Stratified 50% Train / 50% Test)...")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        tr_idx, te_idx = next(sss.split(raw_rna, labels))
        np.savez(split_file, train_idx=tr_idx, test_idx=te_idx)
        print(f"saved: {split_file}")

    print(f"data sum {len(labels)}, train {len(tr_idx)}, test {len(te_idx)}")

    X_train_rna = raw_rna[tr_idx]
    y_train = labels[tr_idx]
    X_train_skeleton = raw_atac[tr_idx] if raw_atac is not None else X_train_rna
    global_adj = get_global_adj(X_train_skeleton, resolution=35, overlap=0.5, eps=0.8)
    
    adj_save_path = pathjoin(models_dir, 'global_adj_50_50.pt')
    torch.save(global_adj, adj_save_path)
    global_adj = global_adj.to(device)
    train_dataset = MyDataset(X_train_rna, y_train)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, drop_last=True)
    class_num = len(np.unique(labels))
    gene_num = raw_rna.shape[1]
    counts = np.array([np.sum(y_train == x) for x in np.unique(y_train)])
    weights = np.clip(np.max(counts) / (counts + 1e-6), 1, 50)
    weights = weights / np.sum(weights)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights).float().to(device))
    supcon_fn = SupConLoss(temperature=0.01).to(device)

    model = CSGNet(
        in_channel=1,
        mid_channel=16,
        num_nodes=gene_num,
        out_channel=class_num,
        dropout_ratio=0.2,
        channel_list=[256, 64],
        global_conv1_dim=12,
        global_conv2_dim=4,
        global_adj=global_adj,
        ablation_mode='full'
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    best_loss = float('inf')
    patience, patience_limit = 0, 20
    for epoch in range(1, 101):
        t_loss, t_acc, t_f1 = train_epoch(model, optimizer, train_loader, loss_fn, supcon_fn)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"   Epoch {epoch:3d}: Loss={t_loss:.4f}, Acc={t_acc:.4f}, Macro-F1={t_f1:.4f}")

        if t_loss < best_loss:
            best_loss = t_loss
            patience = 0
            torch.save(model, pathjoin(models_dir, 'tcsgn_50_50_best_model.pth'))
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"⏹️ Early stopping at epoch {epoch}")
                break

    print(f"best model: {pathjoin(models_dir, 'tcsgn_50_50_best_model.pth')}")

if __name__ == '__main__':
    main()