"""
Script Function: Standard 5-fold cross-validation runner for MOGONET across 10 datasets, supporting transcriptomic dropout resilience stress testing and rare cell subpopulation evaluations.
"""
import os
import sys
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import time
import argparse
import random
import gc
import copy
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

if not hasattr(np, 'asscalar'):
    np.asscalar = lambda a: a.item()

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from models import init_model_dict, init_optim
    from train_test import train_epoch, test_epoch
    from utils import one_hot_tensor
    print("Successfully loaded official MOGONET models, train_test, utils core components via native import")
except ImportError as e:
    raise ImportError(f"Import failed! Please ensure the script is located in the MOGONET-main root directory! Error: {e}")

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

BASE_DATA_DIR = os.path.expanduser("~/scMORETA/dataset/pre_data")
BASE_RESULT_DIR = os.path.expanduser("~/scMORETA/result")

DATASETS = [
    pathjoin(BASE_DATA_DIR, "MouseKidney_sciCAR_Processed"),
    pathjoin(BASE_DATA_DIR, "SHARE_seq_Brain_Processed"),
    pathjoin(BASE_DATA_DIR, "SHARE_seq_SkinA_Processed"),
    pathjoin(BASE_DATA_DIR, "SHARE_seq_SkinB_Processed"),
    pathjoin(BASE_DATA_DIR, "SNARE_seq_Cortex_Processed"),
    pathjoin(BASE_DATA_DIR, "SNARE_seq_AdBC_Processed"),
    pathjoin(BASE_DATA_DIR, "SNARE_seq_P0BC_Processed"),
    pathjoin(BASE_DATA_DIR, "Human_Fetal_Brain_Processed"),
    pathjoin(BASE_DATA_DIR, "BMMC_Processed"),
    pathjoin(BASE_DATA_DIR, "PBMC_10k_Processed")
    
]

def preprocess_single_cell_omics(train_rna, test_rna, train_atac, test_atac, n_components=64):
    n_tr = train_atac.shape[0]

    if sp.issparse(train_rna):
        r_tr = train_rna.copy()
        r_te = test_rna.copy()
        r_tr.data = np.log1p(np.maximum(r_tr.data, 0))
        r_te.data = np.log1p(np.maximum(r_te.data, 0))
    else:
        r_tr = np.log1p(np.maximum(train_rna, 0))
        r_te = np.log1p(np.maximum(test_rna, 0))

    svd_r = TruncatedSVD(n_components=n_components, random_state=42)
    rna_tr_emb = svd_r.fit_transform(r_tr)
    rna_te_emb = svd_r.transform(r_te)

    if sp.issparse(train_atac):
        atac_all = sp.vstack([train_atac, test_atac]).tocsr()
        atac_binary = (atac_all > 0).astype(np.float32)
    else:
        atac_all = np.vstack([train_atac, test_atac])
        atac_binary = sp.csr_matrix(np.where(atac_all > 0, 1.0, 0.0).astype(np.float32))

    tfidf = TfidfTransformer(norm='l2', sublinear_tf=True)
    atac_tfidf = tfidf.fit_transform(atac_binary)

    svd_a = TruncatedSVD(n_components=n_components + 1, random_state=42)
    atac_emb_all = svd_a.fit_transform(atac_tfidf)
    atac_clean = atac_emb_all[:, 1:(n_components + 1)]

    atac_tr_emb = atac_clean[:n_tr]
    atac_te_emb = atac_clean[n_tr:]

    norm_r_tr = rna_tr_emb / (np.linalg.norm(rna_tr_emb, axis=1, keepdims=True) + 1e-8)
    norm_r_te = rna_te_emb / (np.linalg.norm(rna_te_emb, axis=1, keepdims=True) + 1e-8)
    norm_a_tr = atac_tr_emb / (np.linalg.norm(atac_tr_emb, axis=1, keepdims=True) + 1e-8)
    norm_a_te = atac_te_emb / (np.linalg.norm(atac_te_emb, axis=1, keepdims=True) + 1e-8)

    scaler_r = StandardScaler()
    scaler_a = StandardScaler()
    rna_tr_norm = scaler_r.fit_transform(norm_r_tr)
    rna_te_norm = scaler_r.transform(norm_r_te)
    atac_tr_norm = scaler_a.fit_transform(norm_a_tr)
    atac_te_norm = scaler_a.transform(norm_a_te)

    del atac_all, atac_binary, atac_tfidf, atac_emb_all, atac_clean
    gc.collect()

    return rna_tr_norm, rna_te_norm, atac_tr_norm, atac_te_norm, svd_r, scaler_r

def compute_adaptive_parameter(data, adj_parameter):
    n = data.shape[0]
    if n <= 10000:
        w = data.norm(p=2, dim=1, keepdim=True)
        dist = 1.0 - torch.mm(data, data.t()) / (w * w.t()).clamp(min=1e-8)
        k_val = int(adj_parameter * n)
        k_val = max(1, min(k_val, dist.numel() - 1))
        param = torch.kthvalue(dist.reshape(-1), k_val).values.item()
        return param
    else:
        sample_idx = torch.randperm(n)[:3500]
        data_sample = data[sample_idx]
        w_sample = data_sample.norm(p=2, dim=1, keepdim=True)
        w_all = data.norm(p=2, dim=1, keepdim=True)
        dist_sample = 1.0 - torch.mm(data_sample, data.t()) / (w_sample * w_all.t()).clamp(min=1e-8)
        fraction = (adj_parameter * n) / (n * n)
        k_val = int(fraction * dist_sample.numel())
        k_val = max(1, min(k_val, dist_sample.numel() - 1))
        param = torch.kthvalue(dist_sample.reshape(-1), k_val).values.item()
        return param

def build_mogonet_trte_adj_safe(data_tr, data_all, trte_idx, adj_parameter, chunk_size=3500):
    n_tr = data_tr.shape[0]
    n_all = data_all.shape[0]
    num_te = len(trte_idx["te"])

    param = compute_adaptive_parameter(data_tr, adj_parameter)

    w_tr = data_tr.norm(p=2, dim=1, keepdim=True)
    rows_list, cols_list, vals_list = [], [], []

    for start in range(0, n_tr, chunk_size):
        end = min(start + chunk_size, n_tr)
        sub_data = data_tr[start:end]
        sub_w = w_tr[start:end]
        dist_chunk = 1.0 - torch.mm(sub_data, data_tr.t()) / (sub_w * w_tr.t()).clamp(min=1e-8)
        mask = (dist_chunk <= param)
        diag_mask = torch.arange(start, end, device=data_tr.device).unsqueeze(1) == torch.arange(n_tr, device=data_tr.device).unsqueeze(0)
        mask = mask & (~diag_mask)
        idx_r, idx_c = torch.nonzero(mask, as_tuple=True)
        sim = 1.0 - dist_chunk[idx_r, idx_c]

        rows_list.append((idx_r + start).cpu())
        cols_list.append(idx_c.cpu())
        vals_list.append(sim.cpu())

    rows_tr = torch.cat(rows_list).numpy() if len(rows_list) > 0 else np.array([], dtype=np.int64)
    cols_tr = torch.cat(cols_list).numpy() if len(cols_list) > 0 else np.array([], dtype=np.int64)
    vals_tr = torch.cat(vals_list).numpy() if len(vals_list) > 0 else np.array([], dtype=np.float32)

    adj_tr = sp.coo_matrix((vals_tr, (rows_tr, cols_tr)), shape=(n_tr, n_tr), dtype=np.float32)
    adj_tr = adj_tr.maximum(adj_tr.T) + sp.eye(n_tr, dtype=np.float32)
    deg_tr = np.array(adj_tr.sum(axis=1)).flatten()
    deg_tr[deg_tr == 0] = 1.0
    adj_tr = sp.diags(1.0 / deg_tr).dot(adj_tr).tocoo()

    data_te = data_all[trte_idx["te"]]
    w_te = data_te.norm(p=2, dim=1, keepdim=True)
    te_rows_list, te_cols_list, te_vals_list = [], [], []

    for start in range(0, num_te, chunk_size):
        end = min(start + chunk_size, num_te)
        sub_te = data_te[start:end]
        sub_w_te = w_te[start:end]
        dist_chunk = 1.0 - torch.mm(sub_te, data_tr.t()) / (sub_w_te * w_tr.t()).clamp(min=1e-8)
        mask = (dist_chunk <= param)
        idx_r, idx_c = torch.nonzero(mask, as_tuple=True)
        sim = 1.0 - dist_chunk[idx_r, idx_c]

        te_idx_real = (idx_r + start + n_tr).cpu()
        tr_idx_real = idx_c.cpu()
        te_rows_list.append(torch.cat([te_idx_real, tr_idx_real]))
        te_cols_list.append(torch.cat([tr_idx_real, te_idx_real]))
        te_vals_list.append(torch.cat([sim.cpu(), sim.cpu()]))

    rows_te = torch.cat(te_rows_list).numpy() if len(te_rows_list) > 0 else np.array([], dtype=np.int64)
    cols_te = torch.cat(te_cols_list).numpy() if len(te_cols_list) > 0 else np.array([], dtype=np.int64)
    vals_te = torch.cat(te_vals_list).numpy() if len(te_vals_list) > 0 else np.array([], dtype=np.float32)

    adj_te = sp.coo_matrix((vals_te, (rows_te, cols_te)), shape=(n_all, n_all), dtype=np.float32)
    adj_te = adj_te + sp.eye(n_all, dtype=np.float32)
    deg_te = np.array(adj_te.sum(axis=1)).flatten()
    deg_te[deg_te == 0] = 1.0
    adj_te = sp.diags(1.0 / deg_te).dot(adj_te).tocoo()

    indices_tr = torch.from_numpy(np.vstack((adj_tr.row, adj_tr.col)).astype(np.int64))
    values_tr = torch.from_numpy(adj_tr.data.astype(np.float32))
    adj_tr_tensor = torch.sparse_coo_tensor(indices_tr, values_tr, torch.Size(adj_tr.shape))

    indices_te = torch.from_numpy(np.vstack((adj_te.row, adj_te.col)).astype(np.int64))
    values_te = torch.from_numpy(adj_te.data.astype(np.float32))
    adj_te_tensor = torch.sparse_coo_tensor(indices_te, values_te, torch.Size(adj_te.shape))

    return adj_tr_tensor, adj_te_tensor

def execute_optimized_mogonet_training(train_rna, train_atac, y_train, test_rna, test_atac, num_classes, y_test=None, stress_test=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_tr = train_rna.shape[0]
    num_te = test_rna.shape[0]

    n_comp = min(64, train_rna.shape[1], train_atac.shape[1] - 1, num_tr - 1)

    tr_rna_norm, te_rna_norm, tr_atac_norm, te_atac_norm, svd_r, scaler_r = preprocess_single_cell_omics(
        train_rna, test_rna, train_atac, test_atac, n_components=n_comp
    )

    data_tr_list = [
        torch.FloatTensor(tr_rna_norm).to(device),
        torch.FloatTensor(tr_atac_norm).to(device)
    ]
    data_all_list = [
        torch.cat([torch.FloatTensor(tr_rna_norm), torch.FloatTensor(te_rna_norm)], dim=0).to(device),
        torch.cat([torch.FloatTensor(tr_atac_norm), torch.FloatTensor(te_atac_norm)], dim=0).to(device)
    ]
    trte_idx = {
        "tr": list(range(num_tr)),
        "te": list(range(num_tr, num_tr + num_te))
    }
    adj_parameter = 100
    adj_r_tr, adj_r_te = build_mogonet_trte_adj_safe(data_tr_list[0], data_all_list[0], trte_idx, adj_parameter)
    adj_a_tr, adj_a_te = build_mogonet_trte_adj_safe(data_tr_list[1], data_all_list[1], trte_idx, adj_parameter)

    adj_tr_list = [adj_r_tr.to(device), adj_a_tr.to(device)]
    adj_te_list = [adj_r_te.to(device), adj_a_te.to(device)]

    class_counts = np.bincount(y_train, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)
    class_priors = class_counts / len(y_train)

    smooth_weights = np.power(len(y_train) / (num_classes * class_counts), 0.6)
    sample_weights_smooth = smooth_weights[y_train]
    sample_weights_smooth = sample_weights_smooth / np.mean(sample_weights_smooth)

    labels_tr_tensor = torch.LongTensor(y_train)
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_classes)
    sample_weight_tr = torch.FloatTensor(sample_weights_smooth)
    if torch.cuda.is_available():
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()

    num_views = 2
    dim_list = [n_comp, n_comp]
    dim_he_list = [400, 400, 200]
    dim_hvcdn = pow(num_classes, num_views)

    model_dict = init_model_dict(num_views, num_classes, dim_list, dim_he_list, dim_hvcdn)
    for m in model_dict:
        model_dict[m] = model_dict[m].to(device)

    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-3
    num_epoch_pretrain = 100
    num_epoch = 200

    optim_dict = init_optim(num_views, model_dict, lr_e=lr_e_pretrain, lr_c=lr_c)
    for epoch in range(num_epoch_pretrain):
        train_epoch(
            data_tr_list, adj_tr_list, labels_tr_tensor,
            onehot_labels_tr_tensor, sample_weight_tr,
            model_dict, optim_dict, train_VCDN=False
        )
        
    optim_dict = init_optim(num_views, model_dict, lr_e=lr_e, lr_c=lr_c)
    
    best_loss = float('inf')
    best_model_dict = copy.deepcopy(model_dict)
    patience_counter = 0
    patience_limit = 20  

    for epoch in range(num_epoch + 1):
        loss_dict = train_epoch(
            data_tr_list, adj_tr_list, labels_tr_tensor,
            onehot_labels_tr_tensor, sample_weight_tr,
            model_dict, optim_dict, train_VCDN=True
        )
        
        total_loss = sum(loss_dict.values())
        loss_c = loss_dict.get("C", 0.0)

        if epoch % 20 == 0 or epoch == 0:
            print(f"    Epoch [{epoch:03d}/{num_epoch}], Total Loss: {total_loss:.4f}, VCDN Loss: {loss_c:.4f}")

        if total_loss < best_loss:
            best_loss = total_loss
            patience_counter = 0
            best_model_dict = copy.deepcopy(model_dict)  
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            print(f"Loss did not decrease for {patience_limit} consecutive epochs, Early Stopping triggered at Epoch {epoch})！")
            model_dict = best_model_dict 
            break

    for m in model_dict:
        model_dict[m].eval()

    with torch.no_grad():
        ci_test_tensors = []
        for v in range(num_views):
            c_feat = model_dict[f'E{v+1}'](data_all_list[v], adj_te_list[v])
            c_logits = model_dict[f'C{v+1}'](c_feat)
            ci_test_tensors.append(c_logits)

        vcdn_out = model_dict['C'](ci_test_tensors)
        prob_vcdn = F.softmax(vcdn_out[trte_idx["te"]], dim=1).cpu().numpy()

        prob_rna = F.softmax(ci_test_tensors[0][trte_idx["te"]], dim=1).cpu().numpy()
        prob_atac = F.softmax(ci_test_tensors[1][trte_idx["te"]], dim=1).cpu().numpy()

        ensemble_prob = 0.50 * prob_vcdn + 0.35 * prob_rna + 0.15 * prob_atac

    tau = 0.28
    calibrated_prob = ensemble_prob / np.power(class_priors + 1e-12, tau)
    preds = calibrated_prob.argmax(axis=1)

    stress_records = []
    if stress_test and y_test is not None:
        print(f"Executing Dropout continuous graph network testing")
        dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        for gamma in dropout_rates:
            np.random.seed(42)
            mask = np.random.binomial(n=1, p=(1.0 - gamma), size=test_rna.shape).astype(np.float32)
            test_rna_pert = test_rna * mask

            r_pert_emb = svd_r.transform(np.log1p(np.maximum(test_rna_pert, 0)))
            norm_r_pert = r_pert_emb / (np.linalg.norm(r_pert_emb, axis=1, keepdims=True) + 1e-8)
            rna_pert_norm = scaler_r.transform(norm_r_pert)

            data_pert_all = [
                torch.cat([data_tr_list[0], torch.FloatTensor(rna_pert_norm).to(device)], dim=0),
                data_all_list[1]
            ]
            _, adj_r_pert_te = build_mogonet_trte_adj_safe(data_tr_list[0], data_pert_all[0], trte_idx, adj_parameter=adj_parameter)
            adj_pert_te_list = [adj_r_pert_te.to(device), adj_te_list[1]]

            with torch.no_grad():
                ci_pert = []
                for v in range(num_views):
                    c_feat = model_dict[f'E{v+1}'](data_pert_all[v], adj_pert_te_list[v])
                    ci_pert.append(model_dict[f'C{v+1}'](c_feat))
                
                v_out = model_dict['C'](ci_pert)
                prob_v = F.softmax(v_out[trte_idx["te"]], dim=1).cpu().numpy()
                prob_r = F.softmax(ci_pert[0][trte_idx["te"]], dim=1).cpu().numpy()
                prob_a = F.softmax(ci_pert[1][trte_idx["te"]], dim=1).cpu().numpy()
                ens_p = 0.50 * prob_v + 0.35 * prob_r + 0.15 * prob_a
                cal_p = ens_p / np.power(class_priors + 1e-12, tau)
                pred_g = cal_p.argmax(axis=1)

            f1_g = f1_score(y_test, pred_g, average='macro')
            acc_g = accuracy_score(y_test, pred_g)
            kappa_g = cohen_kappa_score(y_test, pred_g)
            stress_records.append({'Dropout': gamma * 100, 'Macro_F1': f1_g, 'Acc': acc_g, 'Kappa': kappa_g})
            print(f"      Dropout {int(gamma*100):02d}% -> F1: {f1_g:.4f} | Acc: {acc_g:.4f}")

    del model_dict, data_tr_list, data_all_list, adj_tr_list, adj_te_list
    del labels_tr_tensor, onehot_labels_tr_tensor, sample_weight_tr
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return preds, stress_records

def run_mogonet_5fold_eval(data_path, out_root=BASE_RESULT_DIR, stress_test=False):
    if not os.path.exists(data_path):
        base_folder_name = os.path.basename(os.path.normpath(data_path))
        data_path = pathjoin(BASE_DATA_DIR, base_folder_name)

    base_name = os.path.basename(os.path.normpath(data_path))

    print(f"\n" + "=" * 65)
    print(f"Starting 5-fold evaluation:: {base_name}")
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
        print(f"\n-------------------- Fold {k_fold} --------------------")

        fold_dir = pathjoin(data_path, f'fold_{k_fold}')
        if not os.path.exists(fold_dir):
            raise FileNotFoundError(f"Cannot find {fold_dir}, please make sure 5-fold slices are generated for this dataset")

        train_rna = np.load(pathjoin(fold_dir, 'train_rna.npy'), allow_pickle=True)
        train_labels = np.load(pathjoin(fold_dir, 'train_labels.npy'), allow_pickle=True)
        test_rna = np.load(pathjoin(fold_dir, 'test_rna.npy'), allow_pickle=True)
        test_labels = np.load(pathjoin(fold_dir, 'test_labels.npy'), allow_pickle=True)

        if os.path.exists(pathjoin(fold_dir, 'train_atac.npy')):
            train_atac = np.load(pathjoin(fold_dir, 'train_atac.npy'), allow_pickle=True)
            test_atac = np.load(pathjoin(fold_dir, 'test_atac.npy'), allow_pickle=True)
        else:
            train_atac = np.load(pathjoin(fold_dir, 'train_peaks.npy'), allow_pickle=True)
            test_atac = np.load(pathjoin(fold_dir, 'test_peaks.npy'), allow_pickle=True)

        print(f"Sample size: Train set {train_rna.shape[0]} cells | Test set {test_rna.shape[0]} cells")

        all_lbls = np.concatenate([train_labels, test_labels])
        unique_classes = sorted(list(np.unique(all_lbls)))
        label_to_clean_str = {c: f"Type_{i:03d}" for i, c in enumerate(unique_classes)}
        tr_str_labels = [label_to_clean_str[c] for c in train_labels]
        te_str_labels = [label_to_clean_str[c] for c in test_labels]

        le = LabelEncoder()
        le.fit(sorted(list(label_to_clean_str.values())))
        y_train = le.transform(tr_str_labels)
        y_true = le.transform(te_str_labels)
        num_classes = len(unique_classes)

        print(f"Executing MOGONET Fold {k_fold} adaptive threshold topology alignment & asymmetric ensemble inference")
        t0 = time.time()
        y_pred, stress_k = execute_optimized_mogonet_training(
            train_rna, train_atac, y_train, test_rna, test_atac, num_classes=num_classes, y_test=y_true, stress_test=stress_test
        )
        if stress_k:
            for r in stress_k: r['Fold'] = k_fold; stress_all_folds_collector.append(r)
            plot_dir_k = pathjoin(out_root, base_name, "plot_data")
            os.makedirs(plot_dir_k, exist_ok=True)
            pd.DataFrame(stress_k).to_csv(pathjoin(plot_dir_k, f"mogonet_stress_results_fold{k_fold}.csv"), index=False)
        cost = time.time() - t0

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        kappa = cohen_kappa_score(y_true, y_pred)

        test_metrics['Acc'].append(acc)
        test_metrics['F1'].append(f1)
        test_metrics['Kappa'].append(kappa)

        np.save(pathjoin(preds_out_dir, f"mogonet_fold{k_fold}_preds.npy"), y_pred)
        np.save(pathjoin(preds_out_dir, f"mogonet_fold{k_fold}_trues.npy"), y_true)

        all_oof_preds.extend(y_pred)
        all_oof_trues.extend(y_true)

        print(f"Fold {k_fold} completed! Time: {cost:.1f}s | Acc: {acc:.4f} | F1: {f1:.4f} | Kappa: {kappa:.4f}")

        del train_rna, train_atac, test_rna, test_atac, y_pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof_preds_file = pathjoin(preds_out_dir, "mogonet_all_oof_preds.npy")
    oof_trues_file = pathjoin(preds_out_dir, "mogonet_all_oof_trues.npy")
    np.save(oof_preds_file, np.array(all_oof_preds))
    np.save(oof_trues_file, np.array(all_oof_trues))
    print(f"MOGONET full-sample 5-fold unbiased predicted labels saved to: {preds_out_dir}")

    try:
        from sklearn.metrics import classification_report
        all_labels_file = pathjoin(data_path, "Labels.npy")
        label_map_file = pathjoin(data_path, "label_map.npy")
        
        if os.path.exists(all_labels_file) and os.path.exists(label_map_file):
            all_labels = np.load(all_labels_file)
            load_data = np.load(label_map_file, allow_pickle=True)
            label_map = {i: name for i, name in enumerate(load_data)} if isinstance(load_data, np.ndarray) and load_data.size > 1 else (load_data.item() if hasattr(load_data, 'item') else load_data)

            unique, counts = np.unique(all_labels, return_counts=True)
            rare_ids = [u for u, c in zip(unique, counts) if (c / len(all_labels)) < 0.03]

            report = classification_report(all_oof_trues, all_oof_preds, output_dict=True, zero_division=0)
            rare_records = []
            for rid in rare_ids:
                if str(rid) in report:
                    m = report[str(rid)]
                    rare_records.append({
                        'Dataset': base_name,
                        'Model': 'MOGONET',
                        'CellType': label_map.get(rid, f"Class_{rid}"),
                        'Precision': round(m['precision'], 4),
                        'Recall': round(m['recall'], 4),
                        'F1-Score': round(m['f1-score'], 4),
                        'Support': int(m['support'])
                    })

            if len(rare_records) > 0:
                rare_df = pd.DataFrame(rare_records)
                rare_csv_path = pathjoin(out_root, base_name, "mogonet_rare_cell_results.csv")
                rare_df.to_csv(rare_csv_path, index=False)
                print(f"MOGONET rare cell identification results generated directly: {rare_csv_path}")
    except Exception as e:
        print(f"Warning occurred while automatically extracting rare cells: {e}")

    final_res = {k: np.mean(v) for k, v in test_metrics.items()}
    print(f" [{base_name}] MOGONET 5-Fold Final Performance Report:")
    for metric_name, val in final_res.items():
        print(f"  • {metric_name}: {val:.4f}")

    out_dir = pathjoin(out_root, base_name, "plot_data")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = pathjoin(out_dir, "mogonet_final_metrics_summary.csv")

    with open(summary_path, 'w') as f:
        header = "Dataset," + ",".join(final_res.keys())
        values = f"{base_name}," + ",".join([f"{v:.4f}" for v in final_res.values()])
        f.write(header + "\n")
        f.write(values + "\n")

    print(f"MOGONET complete metrics saved to: {summary_path}")
    if stress_test and len(stress_all_folds_collector) > 0:
        total_stress_df = pd.DataFrame(stress_all_folds_collector)
        summary_stress = total_stress_df.groupby('Dropout').agg({
            'Macro_F1': ['mean', 'std'],
            'Acc': ['mean', 'std'],
            'Kappa': ['mean', 'std']
        }).reset_index()

        stress_summary_path = pathjoin(out_dir, "mogonet_stress_results_summary.csv")
        summary_stress.to_csv(stress_summary_path, index=False)
        print("\n" + "=" * 65)
        print("5-Fold Stress Summary Completed]")
        print(f"Overall mean and standard deviation saved to: {stress_summary_path}")
        print("=" * 65)

def get_parser():
    parser = argparse.ArgumentParser(description="Memory-Safe MOGONET Benchmark Runner with Original Adaptive Threshold")
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
        print(f"Starting MOGONET automatic batch evaluation for {len(DATASETS)} datasets in total")
        print("=" * 70)
        for i, d in enumerate(DATASETS, 1):
            if os.path.exists(d):
                try:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    run_mogonet_5fold_eval(d, out_root=args.outdir, stress_test=args.stress_test)
                    time.sleep(2)
                except Exception as e:
                    print(f"Dataset {d} failed with error: {e}")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
            else:
                print(f"Skipping non-existent dataset: {d}")
        print("MOGONET evaluation for all datasets completed successfully！")
    else:
        run_mogonet_5fold_eval(args.expr, out_root=args.outdir, stress_test=args.stress_test)
