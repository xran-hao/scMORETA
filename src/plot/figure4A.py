import matplotlib.pyplot as plt
import numpy as np

metrics = ['AvgBIO', 'NMI', 'ARI', 'ASW_cell']

methods = [
    'scMORETA (Train 50%)',
    'scMORETA (Test 50% Unseen)',
    'scGPT (fine-tuned)',
    'scGPT (from-scratch)',
    'GLUE',
    'Seurat v4'
]

data = {
    'scMORETA (Train 50%)':       [0.9290, 0.9930, 0.9930, 0.8000],
    'scMORETA (Test 50% Unseen)': [0.8340, 0.8760, 0.8880, 0.7380],
    'scGPT (fine-tuned)':         [0.7580, 0.8070, 0.8220, 0.6450],
    'scGPT (from-scratch)':       [0.7280, 0.7750, 0.7590, 0.6510],
    'GLUE':                     [0.7470, 0.8150, 0.8060, 0.6190],
    'Seurat v4':                  [0.7220, 0.7840, 0.6910, 0.6910]
}

morandi_colors = [
    '#5C7C94',  
    '#D49B28',  
    '#8FA4B0', 
    '#7C9D8E',  
    '#D39E82',  
    '#9B8B82'   
]

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

fig, ax = plt.subplots(figsize=(12.5, 7.5))

x = np.arange(len(metrics))
num_methods = len(methods)
width = 0.135

max_vals = [max([data[m][i] for m in methods]) for i in range(len(metrics))]

for m_idx, method in enumerate(methods):
    offset = (m_idx - (num_methods - 1) / 2.0) * width
    values = data[method]

    bars = ax.bar(
        x + offset, values, width,
        label=method,
        color=morandi_colors[m_idx],
        edgecolor='#2b2b2b',
        linewidth=0.6,
        alpha=0.95,
        zorder=3
    )

    for i, (bar, val) in enumerate(zip(bars, values)):
        x_pos = bar.get_x() + bar.get_width() / 2.0
        y_pos = val + 0.01

        ax.text(
            x_pos, y_pos, f"{val:.4f}",
            rotation=90,
            ha='center', va='bottom',
            fontsize=11,
            color='#000000',
            zorder=5
        )

ax.set_ylim(0.3, 1)
ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(['0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'], fontsize=11.5)
ax.set_ylabel("Score", fontsize=12, fontweight='normal', labelpad=8)

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12, color='#222222')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

ax.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, -0.18),
    ncol=3,
    frameon=False,
    fontsize=10.5,
    handletextpad=0.5,
    columnspacing=1.8,
    labelspacing=0.4
)

plt.subplots_adjust(top=0.95, bottom=0.15, left=0.07, right=0.98)

plt.savefig("figure4A.png", dpi=300, bbox_inches='tight')
plt.savefig("figure4A.svg", format='svg', bbox_inches='tight')
plt.show()
