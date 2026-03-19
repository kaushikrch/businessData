"""
Generate figures for Diginetica analysis.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "/home/user/businessData/results"

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'figure.facecolor': 'white',
})

# ============================================================
# Figure 5: Diginetica position gradient
# ============================================================
pos = pd.read_csv(os.path.join(RESULTS_DIR, "diginetica_position_gradient.csv"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(pos['rank_position'], pos['click_rate'] * 100, 'o-', color='#4ECDC4',
         linewidth=2, markersize=5)
ax1.set_xlabel('Search Result Rank Position')
ax1.set_ylabel('Click Rate (%)')
ax1.set_title('A. Click Rate by Search Rank')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0.5, 20.5)

ax2.plot(pos['rank_position'], pos['purchase_rate'] * 10000, 's-', color='#FF6B6B',
         linewidth=2, markersize=5)
ax2.set_xlabel('Search Result Rank Position')
ax2.set_ylabel('Purchase Rate (per 10,000)')
ax2.set_title('B. Purchase Rate by Search Rank')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0.5, 20.5)

plt.suptitle('Attention vs Value by Search Engine Rank Position (Diginetica)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig5_diginetica_position_gradient.png"), bbox_inches='tight')
plt.close()
print("Saved fig5_diginetica_position_gradient.png")

# ============================================================
# Figure 6: Cross-dataset wedge comparison
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

datasets = ['REES46\n(view→cart)', 'Diginetica\n(rank→click)', 'Diginetica\n(rank→click|click→purchase)']
attention_ratios = [1.132, 1.130, np.nan]
value_ratios = [0.993, 0.896, 0.823]

x = np.arange(len(datasets))
width = 0.35

bars1 = ax.bar(x - width/2, attention_ratios, width, label='Attention Ratio (Early/Late)',
               color='#4ECDC4', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, value_ratios, width, label='Value Ratio (Early/Late)',
               color='#FF6B6B', edgecolor='black', linewidth=0.5)

ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='No effect')
ax.set_ylabel('Early / Late Ratio')
ax.set_title('The Attention-Value Wedge Across Datasets')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend(loc='upper right')
ax.set_ylim(0.7, 1.3)

for bar in bars1:
    if not np.isnan(bar.get_height()):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig6_cross_dataset_wedge.png"), bbox_inches='tight')
plt.close()
print("Saved fig6_cross_dataset_wedge.png")

print("All Diginetica figures generated.")
