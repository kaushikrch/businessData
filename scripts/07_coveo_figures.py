"""Generate figures for Coveo analysis."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "/home/user/businessData/results"

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.dpi': 150, 'figure.facecolor': 'white',
})

# ============================================================
# Figure 7: Coveo browsing position gradient
# ============================================================
pos = pd.read_csv(os.path.join(RESULTS_DIR, "coveo_browsing_position_gradient.csv"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(pos['view_position'], pos['cart_rate'] * 100, 'o-', color='#4ECDC4', linewidth=2, markersize=5)
ax1.set_xlabel('View Position in Session')
ax1.set_ylabel('Cart Rate (%)')
ax1.set_title('A. Cart Rate by View Position')
ax1.grid(True, alpha=0.3)

ax2.plot(pos['view_position'], pos['purchase_rate'] * 100, 's-', color='#FF6B6B', linewidth=2, markersize=5)
ax2.set_xlabel('View Position in Session')
ax2.set_ylabel('Purchase Rate (%)')
ax2.set_title('B. Purchase Rate by View Position')
ax2.grid(True, alpha=0.3)

plt.suptitle('Coveo: Cart and Purchase by Browsing Position', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig7_coveo_browsing_gradient.png"), bbox_inches='tight')
plt.close()
print("Saved fig7")

# ============================================================
# Figure 8: Coveo search rank gradient
# ============================================================
spos = pd.read_csv(os.path.join(RESULTS_DIR, "coveo_search_position_gradient.csv"))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(spos['rank_position'], spos['click_rate'] * 100, 'o-', color='#4ECDC4', linewidth=2, markersize=6)
ax.set_xlabel('Search Result Rank Position')
ax.set_ylabel('Click Rate (%)')
ax.set_title('Coveo: Click Rate by Search Rank Position')
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 20.5)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig8_coveo_search_gradient.png"), bbox_inches='tight')
plt.close()
print("Saved fig8")

# ============================================================
# Figure 9: Cross-dataset comparison (updated with 4 datasets)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

datasets = [
    'REES46\n(browse→cart)',
    'Coveo\n(browse→cart)',
    'Diginetica\n(rank→click)',
    'Coveo\n(rank→click)',
]
attention_ratios = [1.132, 1.268, 1.130, 2.198]
# For value ratios, use purchase_ratio for REES46/Diginetica, and purchase_ratio for Coveo browsing
# Coveo search has no purchase data from SERP directly
value_ratios = [0.993, 1.447, 0.896, np.nan]

x = np.arange(len(datasets))
width = 0.35

bars1 = ax.bar(x - width/2, attention_ratios, width,
               label='Attention Ratio (Early/Late)', color='#4ECDC4',
               edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, value_ratios, width,
               label='Value Ratio (Early/Late)', color='#FF6B6B',
               edgecolor='black', linewidth=0.5)

ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='No effect')
ax.set_ylabel('Early / Late Ratio')
ax.set_title('The Attention-Value Wedge Across Four Datasets')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend(loc='upper left')
ax.set_ylim(0.6, 2.5)

for bar in bars1:
    if not np.isnan(bar.get_height()):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.03,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    if not np.isnan(bar.get_height()):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.03,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig9_cross_dataset_wedge_4datasets.png"), bbox_inches='tight')
plt.close()
print("Saved fig9")

print("All Coveo figures generated.")
