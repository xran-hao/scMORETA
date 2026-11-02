"""
Script Function: Cross-donor transfer training for T-CSGN. Trains on source donor (Donor 15078) guided by the pre-computed Mapper hypergraph (global_adj.pt) using CE + SupCon loss, and evaluates zero-shot transfer on target donor (Donor 19593).
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
import random
from model import CSGNet, SupConLoss

source_path = "../dataset/pre_data/BMMC_Processed/InterDonor_BMMC/Source_Train_15078"
target_path = "../dataset/pre_data/BMMC_Processed/InterDonor_BMMC/Target_Test_19593"
model1_path = "../result/BMMC_Processed/models"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        
class MyDataset(Dataset):
    def __init__(self, rna, labels):
        self.rna = torch.tensor(rna, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.rna[idx], self.labels[idx]

def train_one_epoch(model, loader, optimizer, criterion_ce, criterion_con):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        latent, out = model(x, get_latent_variable=True)
        loss = criterion_ce(out, y) + 0.04 * criterion_con(latent, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_true.extend(y.numpy())
    return (
        accuracy_score(all_true, all_preds),
        f1_score(all_true, all_preds, average='macro'),
        cohen_kappa_score(all_true, all_preds)
    )

def main():
    seed_everything(42)
    X_raw = np.load(os.path.join(source_path, 'RNA_matrix.npy'))
    y_raw = np.load(os.path.join(source_path, 'Labels.npy'))
    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y_raw, test_size=0.1, stratify=y_raw, random_state=42
    )

    train_loader = DataLoader(MyDataset(X_train, y_train), batch_size=1024, shuffle=True)
    val_loader = DataLoader(MyDataset(X_val, y_val), batch_size=1024, shuffle=False)
    gene_num = X_raw.shape[1]
    class_num = len(np.unique(y_raw))
    # Load the deterministic consensus hypergraph global_adj.pt built from the source donor to initialize CSGNet
    global_adj = torch.load(os.path.join(source_path, 'global_adj.pt')).to(device)

    model = CSGNet(
        in_channel=1,
        mid_channel=16,  
        num_nodes=gene_num,
        out_channel=class_num,
        global_adj=global_adj,
        ablation_mode='full'  
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion_ce = nn.CrossEntropyLoss().to(device)
    criterion_con = SupConLoss(temperature=0.01).to(device)
    best_val_acc = 0
    
    for epoch in range(1, 151):
        loss = train_one_epoch(model, train_loader, optimizer, criterion_ce, criterion_con)
        val_acc, _, _ = evaluate(model, val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_inter_donor_model.pth")

        if epoch % 20 == 0:
            print(f"    Epoch {epoch}: Train Loss = {loss:.4f}, Val Acc = {val_acc:.4f}")
    # Load the best checkpoint based on validation accuracy and evaluate zero-shot transfer on the unseen Target Donor 
    model.load_state_dict(torch.load("best_inter_donor_model.pth"))

    X_test = np.load(os.path.join(target_path, 'RNA_matrix.npy'))
    y_test = np.load(os.path.join(target_path, 'Labels.npy'))
    test_loader = DataLoader(MyDataset(X_test, y_test), batch_size=1024, shuffle=False)

    t_acc, t_f1, t_kappa = evaluate(model, test_loader)

    print("-" * 50)
    print(f"    Accuracy : {t_acc:.4f}")
    print(f"    Macro-F1 : {t_f1:.4f}")
    print(f"    Kappa    : {t_kappa:.4f}")
    print("-" * 50)

    out_dir = "../result/BMMC_Processed/plot_data"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "tcsgn_inter_donor_results.csv")

    df_res = pd.DataFrame([{
        'Model': 'TCSGN',
        'Accuracy': round(t_acc, 4),
        'Macro_F1': round(t_f1, 4),
        'Kappa': round(t_kappa, 4)
    }])
    
    df_res.to_csv(csv_path, index=False)
    print(f"saved: {csv_path}")

if __name__ == "__main__":
    main()