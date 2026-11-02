import matplotlib.pyplot as plt
import numpy as np

metrics = ['Acc', 'F1', 'Kappa']
methods = ['scMORETA (Ours)', 'scJoint', 'MOGONET', 'MultiKano']

data = {
    'scMORETA (Ours)': [0.8044, 0.7453, 0.7596],
    'scJoint':         [0.7590, 0.7087, 0.7076],
    'MOGONET':         [0.8025, 0.7237, 0.7569],
    'MultiKano':       [0.6983, 0.5495, 0.6342]
}

morandi_colors = [
    '#46697D',  
    '#6E8CA0',  
    '#7D9D8B',  
    '#D4A373'   
]

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

fig, ax = plt.subplots(figsize=(10.5, 6.5))

x = np.arange(len(metrics))
num_methods = len(methods)
width = 0.16  
for m_idx, method in enumerate(methods):
    offset = (m_idx - (num_methods - 1) / 2.0) * width
    values = data[method]
    
    bars = ax.bar(
        x + offset, values, width, 
        label=method, 
        color=morandi_colors[m_idx], 
        edgecolor='#2b2b2b', 
        linewidth=0.7, 
        alpha=0.92,
        zorder=3
    )
    
    for bar, val in zip(bars, values):
        x_pos = bar.get_x() + bar.get_width() / 2.0
        y_pos = val + 0.015  
        ax.text(
            x_pos, y_pos, f"{val:.4f}", 
            rotation=90, 
            ha='center', va='bottom', 
            fontsize=12, 
            fontweight='normal',
            color='#000000',
            zorder=5
        )

ax.set_ylim(0.3, 1)
ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(['0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'], fontsize=11.5)
ax.set_ylabel("Score", fontsize=12, fontweight='normal', labelpad=8)

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12, fontweight='normal', color='#222222')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

ax.legend(
    loc='lower center', 
    bbox_to_anchor=(0.5, -0.18), 
    ncol=4, 
    frameon=False, 
    fontsize=11,
    handletextpad=0.5,
    columnspacing=2.0
)

plt.subplots_adjust(top=0.95, bottom=0.15, left=0.08, right=0.98)

plt.savefig("FigureS5_Cross_Donor_Benchmark.png", dpi=300, bbox_inches='tight')
plt.savefig("FigureS5_Cross_Donor_Benchmark.svg", format='svg', bbox_inches='tight')
plt.show()