"""
Script Function: Cross-donor transfer benchmark evaluation, training MultiKano and scJoint on the source donor (Donor 15078) and evaluating zero-shot generalization on the target donor (Donor 19593).
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import gc
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

pathjoin = os.path.join

source_path = os.path.expanduser("~/scMORETA-main/dataset/pre_data/BMMC_Processed/InterDonor_BMMC/Source_Train_15078")
target_path = os.path.expanduser("~/scMORETA-main/dataset/pre_data/BMMC_Processed/InterDonor_BMMC/Target_Test_19593")
out_dir = os.path.expanduser("~/scMORETA-main/result/BMMC_Processed/plot_data")
os.makedirs(out_dir, exist_ok=True)

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

def evaluate_pytorch_model(encoder, classifier, loader, device):
    encoder.eval()
    classifier.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            embedding = encoder(bx)
            logits = classifier(embedding)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(by.numpy())
    return all_true, all_preds

def execute_multikano_batch_training(train_r_ad, train_a_ad, test_r_ad, test_a_ad, batch_size=32, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mk_dir = os.path.expanduser("~/MultiKano-main/MultiKano")
    if mk_dir not in sys.path:
        sys.path.insert(0, mk_dir)
    from model import MultiOmicsNet, CustomLoss, data_processing

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
            print(f'Epoch [{(epoch + 1):3d}/{epochs}], Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            print(f"Early stopping at epoch {epoch + 1}")
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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.array(all_preds)

def main():
    print("=" * 70)
    src_rna = np.load(pathjoin(source_path, 'RNA_matrix.npy')).astype(np.float32)
    src_atac = np.load(pathjoin(source_path, 'ATAC_gene_activity.npy')).astype(np.float32)
    y_train = np.load(pathjoin(source_path, 'Labels.npy')).astype(np.int64)

    tgt_rna = np.load(pathjoin(target_path, 'RNA_matrix.npy')).astype(np.float32)
    tgt_atac = np.load(pathjoin(target_path, 'ATAC_gene_activity.npy')).astype(np.float32)
    y_test = np.load(pathjoin(target_path, 'Labels.npy')).astype(np.int64)

    print(f"Data ready: Train donor {src_rna.shape[0]} cells, Test donor {tgt_rna.shape[0]} cells\n")

    results = []
    
    # MultiKano 
    try:
        all_lbls = np.concatenate([y_train, y_test])
        unique_classes = sorted(list(np.unique(all_lbls)))
        label_to_clean_str = {c: f"Type_{i:03d}" for i, c in enumerate(unique_classes)}

        tr_str_labels = [label_to_clean_str[c] for c in y_train]
        te_str_labels = [label_to_clean_str[c] for c in y_test]

        train_r_ad, train_a_ad = to_adata_optimized(src_rna, src_atac, tr_str_labels)
        test_r_ad, test_a_ad = to_adata_optimized(tgt_rna, tgt_atac, te_str_labels)

        t0 = time.time()
        y_pred_multi = execute_multikano_batch_training(train_r_ad, train_a_ad, test_r_ad, test_a_ad, batch_size=32, epochs=100)
        cost_m = time.time() - t0

        le = LabelEncoder().fit(sorted(list(label_to_clean_str.values())))
        y_true_multi = le.transform(te_str_labels)

        acc_m = accuracy_score(y_true_multi, y_pred_multi)
        f1_m = f1_score(y_true_multi, y_pred_multi, average='macro')
        kappa_m = cohen_kappa_score(y_true_multi, y_pred_multi)

        print(f" {'Multikano':10s} -> Acc: {acc_m:.4f}, Macro-F1: {f1_m:.4f}, Kappa: {kappa_m:.4f} ({cost_m:.1f}s)")
        results.append({'Model': 'Multikano', 'Accuracy': round(acc_m, 4), 'Macro_F1': round(f1_m, 4), 'Kappa': round(kappa_m, 4)})
        
        del train_r_ad, train_a_ad, test_r_ad, test_a_ad
        gc.collect()
    except Exception as e:
        print(f"MultiKano inter-donor evaluation skipped or failed: {e}")

    # scJoint 
    try:
        scj_dir = os.path.expanduser("~/scJoint-main")
        if scj_dir not in sys.path:
            sys.path.insert(0, scj_dir)
            
        try:
            from util.model_regress import Net_encoder, Net_cell
        except ImportError:
            from util.model_regress import Net_rna as Net_encoder, Classifier as Net_cell

        input_dim = src_rna.shape[1]
        num_classes = len(np.unique(np.concatenate([y_train, y_test])))
        device_sj = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_loader = DataLoader(TensorDataset(torch.tensor(src_rna), torch.tensor(y_train)), batch_size=32, shuffle=True, drop_last=True)
        test_loader = DataLoader(TensorDataset(torch.tensor(tgt_rna), torch.tensor(y_test)), batch_size=32, shuffle=False)

        encoder = Net_encoder(input_size=input_dim).to(device_sj)
        classifier = Net_cell(num_of_class=num_classes).to(device_sj)

        optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        encoder.train()
        classifier.train()
        t0 = time.time()
        for epoch in range(1, 101):
            total_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device_sj), by.to(device_sj)
                optimizer.zero_grad()
                embedding = encoder(bx)
                logits = classifier(embedding)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 10 == 0 or epoch == 1:
                print(f"      scJoint Epoch [{epoch}/100], Loss: {total_loss/len(train_loader):.4f}")
        cost_s = time.time() - t0

        all_true_sj, all_preds_sj = evaluate_pytorch_model(encoder, classifier, test_loader, device_sj)
        acc_s = accuracy_score(all_true_sj, all_preds_sj)
        f1_s = f1_score(all_true_sj, all_preds_sj, average='macro')
        kappa_s = cohen_kappa_score(all_true_sj, all_preds_sj)

        print(f"{'scJoint':10s} -> Acc: {acc_s:.4f}, Macro-F1: {f1_s:.4f}, Kappa: {kappa_s:.4f} ({cost_s:.1f}s)")
        results.append({'Model': 'scJoint', 'Accuracy': round(acc_s, 4), 'Macro_F1': round(f1_s, 4), 'Kappa': round(kappa_s, 4)})
    except Exception as e:
        print(f"scJoint inter-donor evaluation skipped or failed: {e}")

    df_res = pd.DataFrame(results)
    print("\n" + "=" * 65)
    print("Inter-donor baseline methods final metrics summary:")
    print("=" * 65)
    print(df_res.to_string(index=False))
    print("=" * 65)

    save_path = pathjoin(out_dir, "inter_donor_baselines_summary.csv")
    df_res.to_csv(save_path, index=False)
    print(f"Results saved to: {save_path}")

if __name__ == '__main__':
    main()