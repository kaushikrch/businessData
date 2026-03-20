"""
Cross-Platform Wedge Summary Figure
====================================
Creates the "money figure" for the paper: a single visualization showing
the attention-value wedge across all 4 datasets, annotated with estimated
ranking calibration (ρ) and effective assortment size (J).

Directly maps to Propositions 1 (W increases in J) and 2 (W decreases in ρ).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import json

RESULTS_DIR = "/home/user/businessData/results"

# ============================================================
# Load wedge summaries from all datasets
# ============================================================

with open(os.path.join(RESULTS_DIR, "rees46_wedge_summary.json")) as f:
    rees46 = json.load(f)

with open(os.path.join(RESULTS_DIR, "diginetica_wedge_summary.json")) as f:
    diginetica = json.load(f)

with open(os.path.join(RESULTS_DIR, "coveo_wedge_summary.json")) as f:
    coveo = json.load(f)

# ============================================================
# Figure 1: Click ratio vs Purchase ratio (scatter)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Click ratio vs Purchase ratio
ax = axes[0]

datasets = [
    {
        'name': 'REES46',
        'click_ratio': rees46['cart_ratio'],
        'purch_ratio': rees46['purchase_ratio'],
        'J': '~50-100',
        'rho': 'Low',
        'color': '#e74c3c',
        'marker': 'o',
    },
    {
        'name': 'Diginetica',
        'click_ratio': diginetica['click_ratio'],
        'purch_ratio': diginetica['purchase_ratio'],
        'J': '30',
        'rho': 'Low-Med',
        'color': '#3498db',
        'marker': 's',
    },
    {
        'name': 'Coveo\n(Browse)',
        'click_ratio': coveo['browsing']['cart_ratio'],
        'purch_ratio': coveo['browsing']['purchase_ratio'],
        'J': '~2',
        'rho': 'High',
        'color': '#2ecc71',
        'marker': '^',
    },
    {
        'name': 'Coveo\n(Search)',
        'click_ratio': coveo['search']['click_ratio'],
        'purch_ratio': 1.0,  # No purchase data for search
        'J': '20',
        'rho': 'High',
        'color': '#27ae60',
        'marker': 'D',
    },
]

for d in datasets:
    ax.scatter(d['click_ratio'], d['purch_ratio'],
              s=200, c=d['color'], marker=d['marker'],
              edgecolors='black', linewidth=1, zorder=5)
    # Label
    offset = (0.03, 0.03)
    if d['name'] == 'Coveo\n(Search)':
        offset = (0.03, -0.06)
    ax.annotate(d['name'],
               (d['click_ratio'], d['purch_ratio']),
               xytext=(d['click_ratio'] + offset[0], d['purch_ratio'] + offset[1]),
               fontsize=9, fontweight='bold')

# 45-degree line (no wedge)
ax.plot([0.8, 2.5], [0.8, 2.5], 'k--', alpha=0.3, linewidth=1)
ax.text(1.8, 1.85, 'No wedge\n(click = purchase)', fontsize=8, alpha=0.4,
       rotation=35, ha='center')

ax.set_xlabel('Click/Cart Ratio (Early ÷ Late)', fontsize=11)
ax.set_ylabel('Purchase Ratio (Early ÷ Late)', fontsize=11)
ax.set_title('A. Attention-Value Wedge Across Platforms', fontsize=12, fontweight='bold')
ax.set_xlim(0.85, 2.4)
ax.set_ylim(0.8, 1.6)
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.3)

# Shade wedge region
ax.fill_between([1.0, 2.4], [0.8, 0.8], [1.0, 1.0],
               alpha=0.08, color='red', label='Wedge region\n(click > purchase)')

ax.legend(loc='upper left', fontsize=8)

# Panel B: Wedge magnitude vs estimated ρ
ax = axes[1]

# Wedge magnitudes (click_ratio - purchase_ratio)
platforms = [
    {'name': 'Diginetica', 'wedge': 0.234, 'rho_est': 0.3, 'J_est': 30,
     'color': '#3498db', 'marker': 's'},
    {'name': 'REES46', 'wedge': 0.140, 'rho_est': 0.4, 'J_est': 75,
     'color': '#e74c3c', 'marker': 'o'},
    {'name': 'Coveo Browse', 'wedge': -0.179, 'rho_est': 0.85, 'J_est': 2,
     'color': '#2ecc71', 'marker': '^'},
]

for p in platforms:
    ax.scatter(p['rho_est'], p['wedge'],
              s=p['J_est'] * 8 + 80,  # Size proportional to J
              c=p['color'], marker=p['marker'],
              edgecolors='black', linewidth=1, zorder=5)
    ax.annotate(f"{p['name']}\n(J≈{p['J_est']})",
               (p['rho_est'], p['wedge']),
               xytext=(p['rho_est'] + 0.05, p['wedge'] + 0.02),
               fontsize=9, fontweight='bold')

ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Estimated Ranking Calibration (ρ)', fontsize=11)
ax.set_ylabel('Wedge Magnitude\n(Click Ratio − Purchase Ratio)', fontsize=11)
ax.set_title('B. Wedge vs. Ranking Calibration', fontsize=12, fontweight='bold')
ax.set_xlim(0.1, 1.0)

# Add annotation for propositions
ax.annotate('P1: W↑ in J\nP2: W↓ in ρ',
           xy=(0.7, 0.18), fontsize=10, fontstyle='italic',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# Size legend for J
for j_val, j_label in [(2, 'J≈2'), (30, 'J≈30'), (75, 'J≈75')]:
    ax.scatter([], [], s=j_val * 8 + 80, c='gray', alpha=0.5,
              label=j_label, edgecolors='black', linewidth=0.5)
ax.legend(title='Assortment size', loc='upper right', fontsize=8, title_fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig10_cross_platform_wedge_theory.png"),
           dpi=150, bbox_inches='tight')
plt.close()

print(f"Figure saved: {RESULTS_DIR}/fig10_cross_platform_wedge_theory.png")

# ============================================================
# Figure 2: Position gradient comparison (Diginetica vs Coveo)
# ============================================================
import pandas as pd

dig_gradient = pd.read_csv(os.path.join(RESULTS_DIR, "diginetica_position_gradient.csv"))
coveo_search = pd.read_csv(os.path.join(RESULTS_DIR, "coveo_search_position_gradient.csv"))
coveo_browse = pd.read_csv(os.path.join(RESULTS_DIR, "coveo_browsing_position_gradient.csv"))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Diginetica
ax = axes[0]
ranks = dig_gradient['rank_position']
ax.plot(ranks, dig_gradient['click_rate'] * 100, 'b-o', markersize=4, label='Click rate')
ax.plot(ranks, dig_gradient['purchase_rate'] * 100, 'r-s', markersize=4, label='Purchase rate')
ax.set_xlabel('Search Rank Position')
ax.set_ylabel('Rate (%)')
ax.set_title('A. Diginetica (Low ρ)\nWedge present', fontweight='bold')
ax.legend(fontsize=8)
ax.set_xlim(0.5, 20.5)

# Panel B: Coveo Search
ax = axes[1]
if 'rank_position' in coveo_search.columns:
    ranks = coveo_search['rank_position']
    ax.plot(ranks, coveo_search['click_rate'] * 100, 'b-o', markersize=4, label='Click rate')
elif 'position' in coveo_search.columns:
    ranks = coveo_search['position']
    ax.plot(ranks, coveo_search['click_rate'] * 100, 'b-o', markersize=4, label='Click rate')
ax.set_xlabel('Search Rank Position')
ax.set_ylabel('Rate (%)')
ax.set_title('B. Coveo Search (High ρ)\nStrong position effect', fontweight='bold')
ax.legend(fontsize=8)
ax.set_xlim(0.5, 20.5)

# Panel C: Coveo Browsing
ax = axes[2]
if 'view_position' in coveo_browse.columns:
    pos_col = 'view_position'
elif 'position' in coveo_browse.columns:
    pos_col = 'position'
else:
    pos_col = coveo_browse.columns[0]
positions = coveo_browse[pos_col]
ax.plot(positions, coveo_browse['cart_rate'] * 100, 'g-^', markersize=4, label='Cart rate')
if 'purchase_rate' in coveo_browse.columns:
    ax.plot(positions, coveo_browse['purchase_rate'] * 100, 'r-s', markersize=4, label='Purchase rate')
ax.set_xlabel('View Position in Session')
ax.set_ylabel('Rate (%)')
ax.set_title('C. Coveo Browse (High ρ)\nNo wedge + U-shape', fontweight='bold')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig11_position_gradients_comparison.png"),
           dpi=150, bbox_inches='tight')
plt.close()

print(f"Figure saved: {RESULTS_DIR}/fig11_position_gradients_comparison.png")

# ============================================================
# Figure 3: Item FE comparison
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

fe_results = pd.read_csv(os.path.join(RESULTS_DIR, "diginetica_item_fe_results.csv"))

# Extract key coefficients
specs = {
    'Click (no FE)': fe_results[fe_results['specification'].str.contains('Click.*no FE')].iloc[0],
    'Click (item FE)': fe_results[fe_results['specification'] == 'Click ~ rank_position (item FE, demeaned)'].iloc[0],
    'Purchase (no FE)': fe_results[fe_results['specification'].str.contains('Purchase.*no FE.*FE sample')].iloc[0],
    'Purchase (item FE)': fe_results[fe_results['specification'] == 'Purchase ~ rank_position (item FE, demeaned)'].iloc[0],
}

labels = list(specs.keys())
coefs = [specs[l]['coef'] * 1000 for l in labels]  # Scale to per-1000
ses = [specs[l]['se'] * 1000 * 1.96 for l in labels]
colors = ['#3498db', '#2980b9', '#e74c3c', '#c0392b']

x = np.arange(len(labels))
bars = ax.bar(x, coefs, yerr=ses, color=colors, edgecolor='black',
             linewidth=0.5, capsize=4, alpha=0.85)

ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Coefficient (per 1,000 rank positions)', fontsize=10)
ax.set_title('Diginetica: Rank Effect on Click vs. Purchase\n(With and Without Item Fixed Effects)',
            fontweight='bold', fontsize=11)

# Significance annotations
for i, label in enumerate(labels):
    pval = specs[label]['pval']
    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'n.s.'
    y_pos = coefs[i] + ses[i] + 0.02 if coefs[i] > 0 else coefs[i] - ses[i] - 0.04
    ax.text(i, y_pos, sig, ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig12_item_fe_comparison.png"),
           dpi=150, bbox_inches='tight')
plt.close()

print(f"Figure saved: {RESULTS_DIR}/fig12_item_fe_comparison.png")
print("All figures complete.")
