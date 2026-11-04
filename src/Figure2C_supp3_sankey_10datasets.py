"""
Script Function: Processes 5-fold Out-of-Fold (OOF) predictions across 10 single-cell multi-omics datasets to generate harmonized, publication-ready Sankey diagrams showing prediction-to-ground-truth flow.
"""
import os
import plotly.graph_objects as go
import plotly.colors as pc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors

result_root = "../result"

datasets = [
    "BMMC_Processed",
    "PBMC_10k_Processed",
    "MouseKidney_sciCAR_Processed",
    "SHARE_seq_Brain_Processed",
    "SHARE_seq_SkinA_Processed",
    "SHARE_seq_SkinB_Processed",
    "SNARE_seq_Cortex_Processed",
    "SNARE_seq_AdBC_Processed",
    "SNARE_seq_P0BC_Processed",
    "Human_Fetal_Brain_Processed"
]

SCALE_300_DPI = 300 / 96

def get_consistent_color_map(all_unique_labels):
    sorted_unique = np.unique(all_unique_labels)
    n_types = len(sorted_unique)
    
    if n_types <= 10:
        palette = sns.color_palette("tab10", n_types)
    elif n_types <= 20:
        palette = sns.color_palette("tab20", n_types)
    else:
        palette = sns.color_palette("husl", n_types)
        
    color_map = {ctype: mcolors.to_hex(palette[i]) for i, ctype in enumerate(sorted_unique)}
    return color_map

for dataset_name in datasets:
    csv_path = os.path.join(result_root, dataset_name, "plot_data", f"predictions_{dataset_name}.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(result_root, dataset_name, f"predictions_{dataset_name}.csv")
    
    if not os.path.exists(csv_path):
        print(f"Prediction file not found: {csv_path}, skipping.")
        continue

    display_title = dataset_name.replace('_Processed', '')
    print(f"Generating UMAP-harmonized OOF Sankey for: {display_title}")
    df = pd.read_csv(csv_path)
    # Aggregate transition frequencies between predicted and true classes to set up Sankey sources and targets
    df_sankey = df.groupby(['Predicted', 'True']).size().reset_index(name='Count')

    total_counts = df['True'].value_counts()
    sorted_labels = total_counts.index.tolist()
    n_classes = len(sorted_labels)
    all_nodes = sorted_labels + sorted_labels 

    all_dataset_labels = np.unique(np.concatenate([df['True'].unique(), df['Predicted'].unique()]))
    color_map = get_consistent_color_map(all_dataset_labels)

    node_colors = [color_map[label] for label in sorted_labels] * 2

    label_map_pred = {label: i for i, label in enumerate(sorted_labels)}
    label_map_true = {label: i + n_classes for i, label in enumerate(sorted_labels)}

    source_indices = df_sankey['Predicted'].map(label_map_pred).tolist()
    target_indices = df_sankey['True'].map(label_map_true).tolist()
    values = df_sankey['Count'].tolist()

    link_colors = []
    for src_idx in source_indices:
        label_name = sorted_labels[src_idx]
        hex_color = color_map[label_name]
        rgb = pc.hex_to_rgb(hex_color)
        rgba = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.40)" 
        link_colors.append(rgba)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=18,
            thickness=18,
            line=dict(color="white", width=0),
            label=all_nodes,
            color=node_colors
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color=link_colors
        )
    )])

    fig.update_layout(
        title=dict(
            text=f"{display_title}",
            x=0.5,
            y=0.96,
            xanchor='center',
            yanchor='top',
            font=dict(size=14, family="Arial", color="black")
        ),
        annotations=[
            dict(
                x=0.01,
                y=1.03,
                xref="paper",
                yref="paper",
                text="Predicted",
                showarrow=False,
                font=dict(size=12, family="Arial", color="black")
            ),
            dict(
                x=0.99,
                y=1.03,
                xref="paper",
                yref="paper",
                text="Ground Truth",
                showarrow=False,
                font=dict(size=12, family="Arial", color="black")
            )
        ],
        font=dict(family="Arial", size=10.5, color="black"),
        width=650,
        height=650,
        margin=dict(l=35, r=35, t=65, b=25),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    save_dir = os.path.join("Figures_output", "Figure2C_Sankey")
    os.makedirs(save_dir, exist_ok=True)
    
    base_save_path = os.path.join(save_dir, f"Sankey_OOF_{dataset_name}")
    png_path = f"{base_save_path}.png"
    svg_path = f"{base_save_path}.svg"

    try:
        fig.write_image(png_path, scale=SCALE_300_DPI)
        fig.write_image(svg_path)
        print(f" -> Successfully exported: {png_path} and {svg_path}")
    except Exception as e:
        print(f"Error exporting image: {e}")

print("All UMAP-harmonized Sankey diagrams generated successfully!")
