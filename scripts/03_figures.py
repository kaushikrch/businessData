"""
Generate figures for the Attention-Value Wedge analysis.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

RESULTS_DIR = "/home/user/businessData/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'figure.facecolor': 'white',
})

# ============================================================
# Figure 1: REES46 Wedge Summary
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Panel A: Cart and Purchase rates by exposure
categories = ['Early Exposure\n(1st quartile)', 'Late Exposure\n(2nd-4th quartile)']
cart_rates = [0.0270, 0.0238]
purch_rates = [0.0395, 0.0398]

x = np.arange(len(categories))
width = 0.35

bars1 = axes[0].bar(x - width/2, [r*100 for r in cart_rates], width, label='Cart Rate', color='#4ECDC4', edgecolor='black', linewidth=0.5)
bars2 = axes[0].bar(x + width/2, [r*100 for r in purch_rates], width, label='Purchase Rate', color='#FF6B6B', edgecolor='black', linewidth=0.5)

axes[0].set_ylabel('Rate (%)')
axes[0].set_title('A. Cart vs Purchase by Exposure Position')
axes[0].set_xticks(x)
axes[0].set_xticklabels(categories)
axes[0].legend(loc='upper right')
axes[0].set_ylim(0, 5.5)

# Add value labels
for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=9)

# Panel B: Conditional purchase rates
cond_rates = [0.4930, 0.5073]
colors = ['#4ECDC4', '#556270']
bars = axes[1].bar(categories, [r*100 for r in cond_rates], color=colors, edgecolor='black', linewidth=0.5)
axes[1].set_ylabel('Purchase Rate | Cart (%)')
axes[1].set_title('B. Conversion Conditional on Cart')
axes[1].set_ylim(45, 55)

for bar in bars:
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10)

# Panel C: The Wedge — ratio comparison
ratios = {
    'Cart Rate\nRatio': 1.132,
    'Purchase Rate\nRatio': 0.993,
}
bars = axes[2].bar(ratios.keys(), ratios.values(), color=['#4ECDC4', '#FF6B6B'], edgecolor='black', linewidth=0.5)
axes[2].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
axes[2].set_ylabel('Early / Late Ratio')
axes[2].set_title('C. The Attention-Value Wedge')
axes[2].set_ylim(0.9, 1.2)

for bar in bars:
    axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig1_rees46_wedge.png"), bbox_inches='tight')
plt.close()
print("Saved fig1_rees46_wedge.png")

# ============================================================
# Figure 2: Position gradient (continuous)
# ============================================================
# Load processed data
proc_data = pd.read_parquet("/home/user/businessData/data_processed/rees46_views_processed.parquet")

# Bin normalized position into deciles
proc_data['position_bin'] = pd.cut(proc_data['norm_position'], bins=10, labels=False)

bin_stats = proc_data.groupby('position_bin').agg(
    cart_rate=('was_carted', 'mean'),
    purchase_rate=('was_purchased', 'mean'),
    n=('was_carted', 'count')
).reset_index()

fig, ax1 = plt.subplots(figsize=(8, 5))

x = bin_stats['position_bin']
ax1.plot(x, bin_stats['cart_rate']*100, 'o-', color='#4ECDC4', linewidth=2, markersize=6, label='Cart Rate')
ax1.plot(x, bin_stats['purchase_rate']*100, 's-', color='#FF6B6B', linewidth=2, markersize=6, label='Purchase Rate')

ax1.set_xlabel('Position Decile (0 = First Viewed → 9 = Last Viewed)')
ax1.set_ylabel('Rate (%)')
ax1.set_title('Cart and Purchase Rates by Within-Session View Position (REES46)')
ax1.legend()
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig2_rees46_position_gradient.png"), bbox_inches='tight')
plt.close()
print("Saved fig2_rees46_position_gradient.png")

# ============================================================
# Figure 3: Category heterogeneity
# ============================================================
cat_df = pd.read_csv(os.path.join(RESULTS_DIR, "rees46_category_heterogeneity.csv"))

fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(cat_df))
width = 0.35

ax.bar(x - width/2, cat_df['cart_diff_pp'], width, label='Cart Rate Diff (pp)', color='#4ECDC4', edgecolor='black', linewidth=0.5)
ax.bar(x + width/2, cat_df['purchase_diff_pp'], width, label='Purchase Rate Diff (pp)', color='#FF6B6B', edgecolor='black', linewidth=0.5)

ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xlabel('Category')
ax.set_ylabel('Early-Late Difference (percentage points)')
ax.set_title('Attention-Value Wedge by Category (REES46)')
ax.set_xticks(x)
ax.set_xticklabels(cat_df['category'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig3_rees46_category_heterogeneity.png"), bbox_inches='tight')
plt.close()
print("Saved fig3_rees46_category_heterogeneity.png")

# ============================================================
# Figure 4: YOOCHOOSE results
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# All sessions
categories = ['Early\n(1st quartile)', 'Late\n(2nd-4th)']
rates_all = [38.75, 35.22]
rates_buy = [60.98, 51.56]

axes[0].bar(categories, rates_all, color=['#4ECDC4', '#556270'], edgecolor='black', linewidth=0.5)
axes[0].set_ylabel('Purchase Rate (%)')
axes[0].set_title('A. All Sessions')
axes[0].set_ylim(0, 70)
for i, v in enumerate(rates_all):
    axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)

axes[1].bar(categories, rates_buy, color=['#4ECDC4', '#556270'], edgecolor='black', linewidth=0.5)
axes[1].set_ylabel('Purchase Rate (%)')
axes[1].set_title('B. Buying Sessions Only')
axes[1].set_ylim(0, 70)
for i, v in enumerate(rates_buy):
    axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)

plt.suptitle('Purchase Rate by Click Position (YOOCHOOSE)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig4_yoochoose_wedge.png"), bbox_inches='tight')
plt.close()
print("Saved fig4_yoochoose_wedge.png")

print("\nAll figures generated.")
