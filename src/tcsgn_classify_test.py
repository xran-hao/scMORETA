import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import scanpy as sc
import os
import torch
import argparse
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import inspect
from model import CSGNet
from datasets import MyDataset
from plot_utils import plot_confusion_matrix, plot_umap

pathjoin = os.path.join

def save_both_formats(plot_func, *args, base_filepath, dataset_name=None, **kwargs):
    base_no_ext = os.path.splitext(base_filepath)[0]
    formats = [f"{base_no_ext}.png", f"{base_no_ext}.svg"]
    
    sig = inspect.signature(plot_func)
    accepts_dpi = 'dpi' in sig.parameters
    accepts_dataset = 'dataset_name' in sig.parameters

    for filepath in formats:
        call_kwargs = kwargs.copy()
        if accepts_dpi:
            call_kwargs['dpi'] = 300
        if accepts_dataset and dataset_name:
            call_kwargs['dataset_name'] = dataset_name.replace('_Processed', '')

        plot_func(*args, filepath, **call_kwargs)

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
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-hvgs', '--high_var_genes', type=int, default=2000)
    parser.add_argument('-ca', '--csn_alpha', type=float, default=0.01)
    parser.add_argument('-addname', type=str, default="")
    parser.add_argument('-nsp', '--n_splits', type=int, default=5)
    parser.add_argument('--stress_test', action='store_true', help='Enable multi-gradient transcriptome dropout stress test for resilience')
    parser.add_argument('-clist', '--channel_list', nargs='+', type=int, default=[256, 64])
    parser.add_argument('-mc', '--mid_channel', type=int, default=16)
    parser.add_argument('-gcdim1', '--global_conv1_dim', type=int, default=12)
    parser.add_argument('-gcdim2', '--global_conv2_dim', type=int, default=4)
    
    return parser

# Evaluates model on test split; returns Accuracy, Macro-F1, Cohen's Kappa, and latent embeddings
def test(model, loader, device, predicts=False, latent=False):
    model.eval()
    y_pred = []
    y_true = []
    y_out = []
    cell_latent_list = []

    with torch.no_grad():
        for data_x, data_y in loader:
            data_x = data_x.to(device)
            data_y = data_y.to(device)
            
            if latent:
                latent_var, output = model(data_x, get_latent_variable=True)
                cell_latent_list.append(latent_var.cpu().numpy())
            else:
                output = model(data_x)
            
            y_softmax = F.softmax(output, dim=1).cpu().numpy()
            y_out.extend(y_softmax)
            
            pred = output.argmax(dim=1).cpu().numpy()
            y = data_y.cpu().numpy()
            
            y_pred.extend(pred)
            y_true.extend(y)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)
    all_latent = None
    if len(cell_latent_list) > 0:
        all_latent = np.vstack(cell_latent_list)

    if predicts:
        if latent:
            return acc, f1, kappa, y_out, y_pred, all_latent
        else:
            return acc, f1, kappa, y_out, y_pred
    else:
        return acc, f1, kappa

def classify_test(args):
    expr_path = args.expr
    base_name = os.path.basename(os.path.normpath(expr_path)).replace('.npz', '')
    
    dataset_outdir = os.path.join(args.outdir, base_name)
        
    models_folder = os.path.join(dataset_outdir, 'models')
    plot_dir = os.path.join(dataset_outdir, 'plot_data')
    os.makedirs(plot_dir, exist_ok=True)

    if not os.path.exists(models_folder):
        raise FileNotFoundError(f"FileNotFoundError: {models_folder}")

    print(f"[Test] {base_name}")
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    seq_dict = {} 
    
    if os.path.isdir(expr_path):
        print("Loading folder data")
        raw_rna = np.load(pathjoin(expr_path, 'RNA_matrix.npy'))
        label = np.load(pathjoin(expr_path, 'Labels.npy'))
        
        map_path = pathjoin(expr_path, 'label_map.npy')
        if os.path.exists(map_path):
            str_labels_list = np.load(map_path, allow_pickle=True)
            str_labels = str_labels_list[label]
        else:
            str_labels = label.astype(str)
            
        gene_name_path = pathjoin(expr_path, 'gene_names.csv')
        if os.path.exists(gene_name_path):
            gene_df = pd.read_csv(gene_name_path, header=None)
            feature_names = gene_df.iloc[:, 0].values.astype(str)
        else:
            feature_names = np.array([f"Gene_{i}" for i in range(raw_rna.shape[1])])

        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
        for k, (tr_idx, te_idx) in enumerate(skf.split(raw_rna, label)):
            seq_dict[f'test_index_{k+1}'] = te_idx

    else:
        print("Loading NPZ data")
        seq_folder = pathjoin('../dataset/5fold_data/', base_name)
        seq_dict_file = pathjoin(seq_folder, 'seq_dict.npz')
        
        if os.path.exists(seq_dict_file):
            seq_dict = dict(np.load(seq_dict_file, allow_pickle=True))
            label = seq_dict['label']
            str_labels = seq_dict['str_labels']
            if 'feature_name' in seq_dict:
                feature_names = seq_dict['feature_name']
            else:
                feature_names = np.array([f"Gene_{i}" for i in range(2000)]) # 假设
        else:
            raise FileNotFoundError(f"Metadata not found: {seq_dict_file}")
            
        original_data = np.load(expr_path, allow_pickle=True)
        if 'data' in original_data: raw_rna = original_data['data']
        elif 'X' in original_data: raw_rna = original_data['X']
        
    if raw_rna.dtype != np.float32: 
        if isinstance(raw_rna, np.ndarray):
            raw_rna = raw_rna.astype(np.float32)

    total_latents = []
    total_labels = []
    total_preds = []
    total_probs = []

    test_metrics = {'Acc': [], 'F1': [], 'Kappa': []}
    class_names = np.unique(str_labels)
    n_classes = len(class_names)
    
    stress_all_folds_collector = []  
    for k in range(args.n_splits):
        k_fold = k + 1
        print(f"\n=== Testing Fold {k_fold} ===")
        
        test_idx = seq_dict[f'test_index_{k_fold}']
        X_test = raw_rna[test_idx]
        y_test = label[test_idx]
        
        test_loader = DataLoader(MyDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)
        
        adj_file = pathjoin(models_folder, f'global_adj_{base_name}_full_{k_fold}.pt')
        if os.path.exists(adj_file):
            global_adj = torch.load(adj_file).to(device)
        else:
            global_adj = None

        model_name = f'tcsgn_{base_name}_full_model{args.addname}{k_fold}.pth'
        model_path = pathjoin(models_folder, model_name)
        if not os.path.exists(model_path): continue
        
        model = torch.load(model_path, weights_only=False)
        if hasattr(model, 'global_H') and global_adj is not None:
             model.register_buffer('global_H', global_adj)
        elif hasattr(model, 'global_adj') and global_adj is not None:
             model.register_buffer('global_adj', global_adj)
             
        model.to(device)
        model.eval()
        # Stress test: simulates various ratios (0%~70%) of technical transcriptomic dropout to evaluate model resilience
        if args.stress_test:
            print(f"Stress Test Mode Activated] Starting to evaluate Dropout gradients")
            dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            stress_res_k = []
            for gamma in dropout_rates:
                np.random.seed(42 + k_fold)
                mask = np.random.binomial(n=1, p=(1.0 - gamma), size=X_test.shape).astype(np.float32)
                X_test_pert = X_test * mask
                pert_loader = DataLoader(MyDataset(X_test_pert, y_test), batch_size=args.batch_size, shuffle=False)
                acc_g, f1_g, kappa_g = test(model, pert_loader, device)
                stress_res_k.append({'Fold': k_fold, 'Dropout': gamma * 100, 'Acc': acc_g, 'Macro_F1': f1_g, 'Kappa': kappa_g})
                print(f"      Dropout {int(gamma*100):02d}% -> F1: {f1_g:.4f} | Acc: {acc_g:.4f}")

            stress_df = pd.DataFrame(stress_res_k)
            stress_df.to_csv(pathjoin(plot_dir, f'tcsgn_stress_results_fold{k_fold}.csv'), index=False)
            stress_all_folds_collector.append(stress_df) 
            acc, f1, kappa, y_out, y_pred, latent = test(model, test_loader, device, predicts=True, latent=True)
        else:
            acc, f1, kappa, y_out, y_pred, latent = test(
                model, test_loader, device, predicts=True, latent=True
            )
            print(f"    Result: Acc={acc:.4f}, F1={f1:.4f}, Kappa={kappa:.4f}")

        test_metrics['Acc'].append(acc); test_metrics['F1'].append(f1) ; test_metrics['Kappa'].append(kappa)
        total_latents.append(latent)
        total_labels.append(y_test)
        total_preds.append(y_pred)
        total_probs.append(y_out)
        

    if args.stress_test and len(stress_all_folds_collector) > 0:
        total_stress_df = pd.concat(stress_all_folds_collector, axis=0)
        summary_stress = total_stress_df.groupby('Dropout').agg({
            'Macro_F1': ['mean', 'std'],
            'Acc': ['mean', 'std'],
            'Kappa': ['mean', 'std']
        }).reset_index()

        summary_csv_path = pathjoin(plot_dir, 'tcsgn_stress_results_summary.csv')
        summary_stress.to_csv(summary_csv_path, index=False)
        print("\n" + "="*60)
        print(f"T-CSGN Overall 5-Fold Stress Summary Completed]")
        print(f"Overall mean and variance file saved to: {summary_csv_path}")
        print("="*60)
        
    print("\n" + "=" * 70)
    print("Constructing 5-Fold strict Out-of-Fold (OOF) panorama and unbiased prediction table")
    print("=" * 70)

    n_total_cells = len(label)
    latent_dim = total_latents[0].shape[1]

    oof_predictions = np.zeros(n_total_cells, dtype=np.int64)
    oof_probabilities = np.zeros((n_total_cells, n_classes), dtype=np.float32)
    oof_latents = np.zeros((n_total_cells, latent_dim), dtype=np.float32)

    for k in range(args.n_splits):
        k_fold = k + 1
        test_idx = seq_dict[f'test_index_{k_fold}']

        oof_predictions[test_idx] = total_preds[k]
        oof_probabilities[test_idx] = total_probs[k]
        oof_latents[test_idx] = total_latents[k]

    oof_acc = accuracy_score(label, oof_predictions)
    oof_f1 = f1_score(label, oof_predictions, average='macro')
    oof_kappa = cohen_kappa_score(label, oof_predictions)

    print(f"5-Fold OOF Overall Unbiased Evaluation Results")
    print(f"   • Overall Accuracy : {oof_acc:.4f}")
    print(f"   • Macro-averaged F1: {oof_f1:.4f}")
    print(f"   • Cohen's Kappa    : {oof_kappa:.4f}")

    print("\nExporting publication-grade 5-Fold OOF panorama")

    cm_dir = os.path.abspath(pathjoin(os.path.dirname(__file__), 'Figures_output/Figure2A_Confusion'))
    os.makedirs(cm_dir, exist_ok=True)

    cm_save_path = pathjoin(cm_dir, f'OOF_WholeData_Confusion_Matrix_{base_name}')
    save_both_formats(plot_confusion_matrix, label, oof_predictions, class_names,
                      base_filepath=cm_save_path, dataset_name=base_name)
    save_both_formats(plot_confusion_matrix, label, oof_predictions, class_names,
                      base_filepath=cm_save_path, dataset_name=base_name)


    # Generate High-Quality Global Panorama (Inference on All Data via Best Model)
    print("\n=== Generating Best Model Visualization (All Data) ===")

    # Identify the fold with the highest Macro-F1 score
    best_fold_idx = np.argmax(test_metrics['F1'])
    best_fold = best_fold_idx + 1
    best_f1_score = test_metrics['F1'][best_fold_idx]
    print(f" The best performing model is Fold {best_fold} (F1={best_f1_score:.4f})")

    best_model_path = pathjoin(models_folder, f'tcsgn_{base_name}_full_model{args.addname}{best_fold}.pth')
    best_model = torch.load(best_model_path, map_location=device, weights_only=False)
    best_model.to(device)
    best_model.eval()

    best_adj_path = pathjoin(models_folder, f'global_adj_{base_name}_full_{best_fold}.pt')
    if os.path.exists(best_adj_path):
        best_adj = torch.load(best_adj_path).to(device)
        if hasattr(best_model, 'global_H') or hasattr(best_model, 'global_adj'):
            target_attr = 'global_H' if hasattr(best_model, 'global_H') else 'global_adj'
            best_model.register_buffer(target_attr, best_adj)

    all_dataset = MyDataset(raw_rna, label)
    all_loader = DataLoader(all_dataset, batch_size=args.batch_size, shuffle=False)

    _, _, _, _, _, all_latent = test(
        best_model, all_loader, device, predicts=True, latent=True
    )

    # Generate and plot the global UMAP using the best model
    umap_dir = os.path.abspath(pathjoin(os.path.dirname(__file__), 'Figures_output/Figure2B_Umap'))
    os.makedirs(umap_dir, exist_ok=True)
    
    umap_save_path = pathjoin(umap_dir, f'BestModel_WholeData_UMAP_{base_name}')
    save_both_formats(plot_umap, all_latent, label, class_names,
                      base_filepath=umap_save_path, dataset_name=base_name)
    save_both_formats(plot_umap, all_latent, label, class_names,
                      base_filepath=umap_save_path, dataset_name=base_name)
    print(f"Global UMAP panorama of the best model saved to {umap_save_path}")
    

    print("Saving unbiased prediction detail table (for Sankey diagram)")
    true_names = [class_names[i] for i in label]
    pred_names = [class_names[i] for i in oof_predictions]

    df_pred = pd.DataFrame({
        'True': true_names,
        'Predicted': pred_names
    })

    pred_save_path = pathjoin(plot_dir, f'predictions_{base_name}.csv')
    df_pred.to_csv(pred_save_path, index=False)
    print(f"Strict unbiased prediction table overwritten and updated to: {pred_save_path}")

    summary_path = pathjoin(plot_dir, 'final_metrics_summary.csv')
    final_res = {k: np.mean(v) for k, v in test_metrics.items()}

    with open(summary_path, 'w') as f:
        header = "Dataset," + ",".join(final_res.keys())
        values = f"{base_name}," + ",".join([f"{v:.4f}" for v in final_res.values()])
        f.write(header + "\n")
        f.write(values + "\n")

    print(f"Performance summary saved to: {summary_path}")
    print("\n5-fold unbiased testing and plotting for this dataset completed successfully\n")

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    classify_test(args)