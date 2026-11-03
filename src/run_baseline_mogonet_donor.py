"""
Script Function: Cross-donor transfer benchmark for MOGONET, featuring multi-omics LSI/TF-IDF preprocessing, adaptive affinity graph construction, and cross-domain inference via View Correlation Discovery Network (VCDN).
"""
import os
import sys
import time
import random
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

if not hasattr(np, 'asscalar'):
    np.asscalar = lambda a: a.item()

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

mogonet_dir = os.path.expanduser("~/MOGONET-main")
if mogonet_dir not in sys.path:
    sys.path.insert(0, mogonet_dir)

try:
    from models import init_model_dict, init_optim
    from train_test import train_epoch, test_epoch
    from utils import one_hot_tensor
except ImportError as e:
    raise ImportError(f"error: {e}")

source_path = os.path.expanduser("~/scMORETA/dataset/pre_data/BMMC_Processed/InterDonor_BMMC/Source_Train_15078")
target_path = os.path.expanduser("~/scMORETA/dataset/pre_data/BMMC_Processed/InterDonor_BMMC/Target_Test_19593")
out_dir = os.path.expanduser("~/scMORETA/result/BMMC_Processed/plot_data")
preds_dir = os.path.expanduser("~/scMORETA/result/BMMC_Processed/predictions")
os.makedirs(out_dir, exist_ok=True)
os.makedirs(preds_dir, exist_ok=True)

def preprocess_single_cell_omics(src_rna, tgt_rna, src_atac, tgt_atac, n_components=64):
    n_src = src_atac.shape[0]

    r_tr = np.log1p(np.maximum(src_rna, 0))
    r_te = np.log1p(np.maximum(tgt_rna, 0))
    svd_r = TruncatedSVD(n_components=n_components, random_state=42)
    rna_tr_emb = svd_r.fit_transform(r_tr)
    rna_te_emb = svd_r.transform(r_te)

    atac_all = np.vstack([src_atac, tgt_atac])
    atac_binary = sp.csr_matrix(np.where(atac_all > 0, 1.0, 0.0).astype(np.float32))

    tfidf = TfidfTransformer(norm='l2', sublinear_tf=True)
    atac_tfidf = tfidf.fit_transform(atac_binary)

    svd_a = TruncatedSVD(n_components=n_components + 1, random_state=42)
    atac_emb_all = svd_a.fit_transform(atac_tfidf)
    atac_clean = atac_emb_all[:, 1:(n_components + 1)]

    atac_tr_emb = atac_clean[:n_src]
    atac_te_emb = atac_clean[n_src:]

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

    return rna_tr_norm, rna_te_norm, atac_tr_norm, atac_te_norm

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
        sub_w = w_te[start:end]
        dist_chunk = 1.0 - torch.mm(sub_te, data_tr.t()) / (sub_w * w_tr.t()).clamp(min=1e-8)
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

def run_mogonet_inter_donor(src_rna, src_atac, y_train, tgt_rna, tgt_atac, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_tr = src_rna.shape[0]
    num_te = tgt_rna.shape[0]

    n_comp = min(64, src_rna.shape[1], src_atac.shape[1] - 1, num_tr - 1)

    print(" (TF-IDF + LSI)...")
    tr_rna_norm, te_rna_norm, tr_atac_norm, te_atac_norm = preprocess_single_cell_omics(
        src_rna, tgt_rna, src_atac, tgt_atac, n_components=n_comp
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

    adj_parameter = 20
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
    for epoch in range(num_epoch + 1):
        loss_dict = train_epoch(
            data_tr_list, adj_tr_list, labels_tr_tensor,
            onehot_labels_tr_tensor, sample_weight_tr,
            model_dict, optim_dict, train_VCDN=True
        )
        if epoch % 50 == 0:
            loss_c = loss_dict.get("C", 0.0)
            print(f"      Epoch [{(epoch):3d}/{num_epoch}], VCDN Loss: {loss_c:.4f}")

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
    y_pred = calibrated_prob.argmax(axis=1)

    del model_dict, data_tr_list, data_all_list, adj_tr_list, adj_te_list
    del labels_tr_tensor, onehot_labels_tr_tensor, sample_weight_tr
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return y_pred

def main():
    seed_everything(42)
    print("=" * 70)
    print(" [Inter-Donor] MOGONET ")
    print("=" * 70)

    print("loading donor 15078 (Source) and donor 19593 (Target) data")
    src_rna = np.load(pathjoin(source_path, 'RNA_matrix.npy'), allow_pickle=True).astype(np.float32)
    src_atac = np.load(pathjoin(source_path, 'ATAC_gene_activity.npy'), allow_pickle=True).astype(np.float32)
    y_train_raw = np.load(pathjoin(source_path, 'Labels.npy'), allow_pickle=True)

    tgt_rna = np.load(pathjoin(target_path, 'RNA_matrix.npy'), allow_pickle=True).astype(np.float32)
    tgt_atac = np.load(pathjoin(target_path, 'ATAC_gene_activity.npy'), allow_pickle=True).astype(np.float32)
    y_test_raw = np.load(pathjoin(target_path, 'Labels.npy'), allow_pickle=True)

    print(f"train donor {src_rna.shape[0]} 细胞, test donor {tgt_rna.shape[0]} 细胞\n")

    all_lbls = np.concatenate([y_train_raw, y_test_raw])
    unique_classes = sorted(list(np.unique(all_lbls)))
    label_to_clean_str = {c: f"Type_{i:03d}" for i, c in enumerate(unique_classes)}
    tr_str_labels = [label_to_clean_str[c] for c in y_train_raw]
    te_str_labels = [label_to_clean_str[c] for c in y_test_raw]

    le = LabelEncoder().fit(sorted(list(label_to_clean_str.values())))
    y_train = le.transform(tr_str_labels)
    y_test = le.transform(te_str_labels)
    num_classes = len(unique_classes)

    t0 = time.time()
    y_pred = run_mogonet_inter_donor(src_rna, src_atac, y_train, tgt_rna, tgt_atac, num_classes)
    cost = time.time() - t0

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    kappa = cohen_kappa_score(y_test, y_pred)

    print("\n" + "=" * 65)
    print("=" * 65)
    print(f"  • Model:       MOGONET")
    print(f"  • Accuracy:   {acc:.4f}")
    print(f"  • Macro-F1:   {f1:.4f}")
    print(f"  • Kappa:      {kappa:.4f}")
    print(f"  • 耗时:       {cost:.1f}s")
    print("=" * 65)

    np.save(pathjoin(preds_dir, "mogonet_inter_donor_preds.npy"), y_pred)
    np.save(pathjoin(preds_dir, "mogonet_inter_donor_trues.npy"), y_test)

    res_record = [{
        'Model': 'MOGONET',
        'Accuracy': round(acc, 4),
        'Macro_F1': round(f1, 4),
        'Kappa': round(kappa, 4)
    }]
    df_mogonet = pd.DataFrame(res_record)
    save_path = pathjoin(out_dir, "inter_donor_mogonet_summary.csv")
    df_mogonet.to_csv(save_path, index=False)
    print(f"saved: {save_path}")

    total_summary_path = pathjoin(out_dir, "inter_donor_baselines_summary.csv")
    if os.path.exists(total_summary_path):
        try:
            df_total = pd.read_csv(total_summary_path)
            if 'MOGONET' in df_total['Model'].values:
                df_total.loc[df_total['Model'] == 'MOGONET', ['Accuracy', 'Macro_F1', 'Kappa']] = [round(acc, 4), round(f1, 4), round(kappa, 4)]
            else:
                df_total = pd.concat([df_total, df_mogonet], ignore_index=True)
            df_total.to_csv(total_summary_path, index=False)
            print(f"saved: {total_summary_path}")
        except Exception as e:
            print(f"error: {e}")

if __name__ == '__main__':
    main()
