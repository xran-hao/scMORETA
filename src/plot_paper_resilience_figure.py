"""
Script Function: Generates Figure 6A & 6B benchmarking model resilience under transcriptomic dropout stress:
     Panel A: Performance decay curves across multi-gradient dropout rates (0%~70%);
     Panel B: Macro-averaged F1 and retention rates (%) at the critical 50% stress threshold.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['svg.fonttype'] = 'none'  
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.9

DPI = 300
data_dir = os.path.expanduser("~/scMORETA/result/PBMC_10k_Processed/plot_data")
save_dir = os.path.expanduser("~/scMORETA/src/Figures_output")
os.makedirs(save_dir, exist_ok=True)

LINE_COLORS = {
    'scMORETA (Ours)': '#46697D',  
    'scJoint':         '#6E8CA0',   
    'MOGONET':         '#7D9D8B',   
    'MultiKano':       '#C89E7A'    
}

BAR_PALETTE = {
    'Intact': '#7C9D8E',      
    '50% critical stress':  '#D4A373'       
}

def load_and_parse_csv(filename, model_name):
    path = os.path.join(data_dir, filename)
    df = pd.read_csv(path, skiprows=[1])  
    res = pd.DataFrame({
        'Model': model_name,
        'Dropout': df.iloc[:, 0].astype(float),
        'Mean_F1': df.iloc[:, 1].astype(float)
    })
    return res

df_tcsgn = load_and_parse_csv("tcsgn_stress_results_summary.csv", "scMORETA (Ours)")
df_scj   = load_and_parse_csv("scjoint_stress_results_summary.csv", "scJoint")
df_mogo  = load_and_parse_csv("mogonet_stress_results_summary.csv", "MOGONET")
df_mkano = load_and_parse_csv("multikano_stress_results_summary.csv", "MultiKano")

df_all = pd.concat([df_tcsgn, df_scj, df_mogo, df_mkano], ignore_index=True)
models_order = ['scMORETA (Ours)', 'MOGONET', 'MultiKano', 'scJoint']

fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2), dpi=DPI, gridspec_kw={'width_ratios': [1.2, 1.1]})

ax1 = axes[0]
df_plot = df_all[df_all['Dropout'] <= 70.0].copy()
# Plot Panel A: Trajectory curves of Macro-F1 under 0%~70% dropout with a dashed line at 50% critical stress
for name in models_order:
    sub = df_plot[df_plot['Model'] == name]
    ax1.plot(sub['Dropout'], sub['Mean_F1'], marker='o', markersize=6.5, linewidth=2.4, label=name, color=LINE_COLORS[name])

ax1.axvline(x=50, color='#888888', linestyle='--', linewidth=1.2, alpha=0.7)
ax1.text(51, 0.47, '50% critical stress', color='#555555', fontsize=11, style='italic', fontweight='bold')
ax1.set_xlabel("Masked Proportion (%)", fontsize=12)
ax1.set_ylabel("Macro-averaged F1 Score", fontsize=12)
ax1.set_xticks([0, 10, 20, 30, 40, 50, 60, 70])
ax1.set_ylim([0.3, 1.02])
ax1.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax1.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10, loc='upper right')
sns.despine(ax=ax1)

ax2 = axes[1]
df_comp = df_all[df_all['Dropout'].isin([0.0, 50.0])].copy()
df_comp['Condition'] = df_comp['Dropout'].map({0.0: 'Intact', 50.0: '50% critical stress'})

sns.barplot(
    x='Model', y='Mean_F1', hue='Condition', data=df_comp,
    order=models_order, palette=BAR_PALETTE, ax=ax2,
    edgecolor='#333333', linewidth=1.0, width=0.60
)
# Plot Panel B: Compare intact (0%) vs. stressed (50%) conditions, annotating F1 scores and performance retention rate (%)
for idx, m_name in enumerate(models_order):
    f0 = df_comp[(df_comp['Model'] == m_name) & (df_comp['Dropout'] == 0.0)]['Mean_F1'].values[0]
    f50 = df_comp[(df_comp['Model'] == m_name) & (df_comp['Dropout'] == 50.0)]['Mean_F1'].values[0]
    retention = (f50 / f0) * 100
    ax2.text(idx - 0.15, f0 + 0.015, f"{f0:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')
    ax2.text(idx + 0.15, f50 + 0.015, f"{f50:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')
    ax2.text(
        idx + 0.17, f50 + 0.06, f"Ret:\n{retention:.1f}%",
        ha='center', va='bottom', fontsize=8.5, fontweight='bold',
        color='#B22222' if m_name == 'scMORETA (Ours)' else '#555555',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFFFFF', edgecolor='#B22222' if m_name in ['scMORETA (Ours)', 'T-CSGN (Ours)'] else '#AAAAAA', linewidth=0.8, alpha=0.95)
    )

ax2.set_ylabel("Macro-averaged F1 Score", fontsize=12)
ax2.set_xlabel("")
ax2.set_ylim([0.3, 1])
ax2.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax2.set_xticklabels(['scMORETA\n(Ours)', 'MOGONET', 'MultiKano', 'scJoint'], fontsize=11)
ax2.legend(frameon=True, facecolor='white', loc='upper right', fontsize=10)
sns.despine(ax=ax2)

plt.tight_layout()
png_path = os.path.join(save_dir, "Figure6AB_Resilience_Benchmark_Clean.png")
svg_path = os.path.join(save_dir, "Figure6AB_Resilience_Benchmark_Clean.svg")

plt.savefig(png_path, format='png', dpi=DPI, bbox_inches='tight')
plt.savefig(svg_path, format='svg', bbox_inches='tight')

print("="*65)
print(f"PNG (300DPI): {png_path}")
print(f"SVG (300DPI): {svg_path}")
print("="*65)
