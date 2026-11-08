import matplotlib.pyplot as plt
import numpy as np

datasets = [
    'BMMC', 'PBMC_10k', 'MouseKidney_sciCAR', 'SHARE_seq_Brain', 'SHARE_seq_SkinA', 
    'SHARE_seq_SkinB', 'SNARE_seq_Cortex', 'SNARE_seq_AdBC', 'SNARE_seq_P0BC', 'Human_Fetal_Brain'
]

methods = ['scMORETA (Ours)', 'scJoint', 'MOGONET', 'MultiKano']

acc_data = {
    'scMORETA (Ours)': [0.8620, 0.9426, 0.8495, 0.8776, 0.8845, 0.8764, 0.8157, 0.9304, 0.7060, 0.6749],
    'scJoint':         [0.8380, 0.9232, 0.8148, 0.8652, 0.8821, 0.8677, 0.8532, 0.9294, 0.7063, 0.8803],
    'MOGONET':         [0.8404, 0.9116, 0.6861, 0.8221, 0.8308, 0.8098, 0.7890, 0.8867, 0.5057, 0.8673],
    'MultiKano':       [0.8464, 0.9313, 0.8291, 0.8676, 0.8637, 0.8316, 0.8250, 0.9178, 0.6966, 0.8333]
}

f1_data = {
    'scMORETA (Ours)': [0.8159, 0.9218, 0.8110, 0.8772, 0.8842, 0.8810, 0.7767, 0.9304, 0.6959, 0.7206],
    'scJoint':         [0.7907, 0.8642, 0.7315, 0.8626, 0.8692, 0.8617, 0.7440, 0.9018, 0.6625, 0.9043],
    'MOGONET':         [0.7941, 0.8859, 0.5839, 0.8150, 0.7919, 0.7786, 0.6754, 0.8802, 0.5013, 0.8937],
    'MultiKano':       [0.7872, 0.9035, 0.7569, 0.8680, 0.8564, 0.8412, 0.7324, 0.9065, 0.6788, 0.8840]
}

kappa_data = {
    'scMORETA (Ours)': [0.8482, 0.9337, 0.8252, 0.8674, 0.8702, 0.8627, 0.7903, 0.9139, 0.6189, 0.6221],
    'scJoint':         [0.8218, 0.9110, 0.7837, 0.8536, 0.8670, 0.8526, 0.8322, 0.9121, 0.6043, 0.8608],
    'MOGONET':         [0.8256, 0.8985, 0.6423, 0.8073, 0.8121, 0.7912, 0.7645, 0.8608, 0.4025, 0.8473],
    'MultiKano':       [0.8309, 0.9205, 0.8010, 0.8561, 0.8468, 0.8128, 0.8001, 0.8981, 0.5997, 0.8059]
}

morandi_colors = [
    '#B85B56',
    '#68889E',
    '#7C9D8E',
    '#D39E82'
]

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

fig, axes = plt.subplots(3, 1, figsize=(16, 20))

metrics_list = [
    ('Acc', acc_data),
    ('mF1', f1_data),
    ('Kappa', kappa_data)
]

x = np.arange(len(datasets))
width = 0.19

for ax_idx, (y_label, data_dict) in enumerate(metrics_list):
    ax = axes[ax_idx]
    
    for m_idx, method in enumerate(methods):
        offset = (m_idx - 1.5) * width 
        values = data_dict[method]
        
        bars = ax.bar(
            x + offset, values, width, 
            label=method, 
            color=morandi_colors[m_idx], 
            edgecolor='#2b2b2b', 
            linewidth=0.6, 
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
                fontsize=10, 
                color='#000000',
                zorder=5
            )
            
    ax.set_ylim(0.5, 1.)
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(['0.3','0.4','0.5', '0.6', '0.7', '0.8', '0.9', '1.0'], fontsize=10.5)
    ax.set_ylabel(y_label, fontsize=12, labelpad=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha='right', fontsize=10.5, color='#222222')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels, 
    loc='lower center', 
    bbox_to_anchor=(0.5, -0.02), 
    ncol=4, 
    frameon=False, 
    fontsize=11.5,
    handletextpad=0.5,
    columnspacing=2.2
)

plt.subplots_adjust(top=0.97, bottom=0.06, left=0.06, right=0.98, hspace=0.22)

plt.savefig("figure3.png", dpi=300, bbox_inches='tight')
plt.savefig("figure3.svg", dpi=300, bbox_inches='tight') 
plt.show()
