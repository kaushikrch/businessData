"""
YOOCHOOSE: Attention-Value Wedge Analysis
============================================================
Tests H1-H3 using click + buy data from RecSys Challenge 2015.

Exposure order: within-session click order (rank by timestamp).
Attention: click event (all rows in clicks.dat are attention events).
Downstream: buy event (matched by session + item).

Key identification caveat:
  Click order is endogenous — items clicked earlier are selected by the user.
  However, the key test is whether BEING clicked earlier predicts purchase
  conditional on being clicked at all. This is about position-in-sequence,
  not about whether the item was clicked.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import json
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

RAW_DIR = "/home/user/businessData/data_raw/yoochoose"
PROC_DIR = "/home/user/businessData/data_processed"
RESULTS_DIR = "/home/user/businessData/results"
LOGS_DIR = "/home/user/businessData/logs"

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

log_lines = []
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    log_lines.append(line)

# ============================================================
# STEP A: Load data
# ============================================================
log("Loading YOOCHOOSE clicks...")
clicks = pd.read_csv(
    os.path.join(RAW_DIR, "yoochoose-clicks.dat"),
    header=None,
    names=['session_id', 'timestamp', 'item_id', 'category'],
    parse_dates=['timestamp'],
    dtype={'session_id': int, 'item_id': int, 'category': str}
)

log(f"Total clicks: {len(clicks):,}")

log("Loading YOOCHOOSE buys...")
buys = pd.read_csv(
    os.path.join(RAW_DIR, "yoochoose-buys.dat"),
    header=None,
    names=['session_id', 'timestamp', 'item_id', 'price', 'quantity'],
    parse_dates=['timestamp'],
    dtype={'session_id': int, 'item_id': int}
)
log(f"Total buys: {len(buys):,}")

# Identify buying sessions
buying_sessions = set(buys['session_id'].unique())
log(f"Sessions with purchases: {len(buying_sessions):,}")
log(f"Total sessions: {clicks['session_id'].nunique():,}")

# ============================================================
# STEP A2: Sample for tractability
# ============================================================
# Full dataset is 33M rows — sample sessions for speed
log("Sampling sessions for tractability...")
np.random.seed(42)
all_sessions = clicks['session_id'].unique()

# Stratified sample: oversample buying sessions
buying_sess_list = list(buying_sessions)
non_buying_sess_list = list(set(all_sessions) - buying_sessions)

# Take all buying sessions (they're fewer) + random sample of non-buying
n_sample_nonbuy = min(500_000, len(non_buying_sess_list))
sampled_nonbuy = np.random.choice(non_buying_sess_list, n_sample_nonbuy, replace=False)
sampled_sessions = set(buying_sess_list) | set(sampled_nonbuy)

clicks_s = clicks[clicks['session_id'].isin(sampled_sessions)].copy()
buys_s = buys[buys['session_id'].isin(sampled_sessions)].copy()

log(f"Sampled clicks: {len(clicks_s):,}")
log(f"Sampled sessions: {len(sampled_sessions):,}")
log(f"  of which buying: {len(buying_sess_list):,}")

# ============================================================
# STEP B: Construct variables
# ============================================================
log("Constructing variables...")

# Sort by session + time
clicks_s = clicks_s.sort_values(['session_id', 'timestamp']).reset_index(drop=True)

# Click rank within session
clicks_s['click_rank'] = clicks_s.groupby('session_id').cumcount() + 1

# Session length
session_lengths = clicks_s.groupby('session_id').size().rename('session_length')
clicks_s = clicks_s.merge(session_lengths, on='session_id', how='left')

# Filter sessions with >=2 clicks
clicks_s = clicks_s[clicks_s['session_length'] >= 2].copy()

# Normalized position
clicks_s['norm_position'] = (clicks_s['click_rank'] - 1) / (clicks_s['session_length'] - 1)

# Early exposure definitions
clicks_s['early_exposure'] = (clicks_s['norm_position'] <= 0.25).astype(int)
clicks_s['first_half'] = (clicks_s['norm_position'] <= 0.5).astype(int)

# Mark which items in each session were purchased
buy_pairs = buys_s[['session_id', 'item_id']].drop_duplicates()
buy_pairs['was_purchased'] = 1
clicks_s = clicks_s.merge(buy_pairs, on=['session_id', 'item_id'], how='left')
clicks_s['was_purchased'] = clicks_s['was_purchased'].fillna(0).astype(int)

# Session-level purchase indicator
clicks_s['session_has_purchase'] = clicks_s['session_id'].isin(buying_sessions).astype(int)

# Category: clean up
clicks_s['is_special_offer'] = (clicks_s['category'] == 'S').astype(int)
clicks_s['has_category'] = clicks_s['category'].str.match(r'^\d{1,2}$', na=False).astype(int)

log(f"Analysis dataset: {len(clicks_s):,} clicks")
log(f"Purchase rate (item-level): {clicks_s['was_purchased'].mean():.4f}")
log(f"Purchase rate | early_exposure=1: {clicks_s.loc[clicks_s['early_exposure']==1, 'was_purchased'].mean():.4f}")
log(f"Purchase rate | early_exposure=0: {clicks_s.loc[clicks_s['early_exposure']==0, 'was_purchased'].mean():.4f}")

# ============================================================
# STEP C: Main regressions
# ============================================================
log("=" * 60)
log("RUNNING MAIN REGRESSIONS")
log("=" * 60)

results_table = []

def run_lpm(y_var, x_vars, data, label):
    """Run LPM with robust SEs."""
    subset = data.dropna(subset=[y_var] + x_vars)
    if len(subset) < 100:
        log(f"  SKIP {label}: too few obs ({len(subset)})")
        return None

    X = sm.add_constant(subset[x_vars])
    y = subset[y_var]

    model = sm.OLS(y, X).fit(cov_type='HC1')

    result = {
        'specification': label,
        'outcome': y_var,
        'N': len(subset),
        'y_mean': y.mean(),
        'R2': model.rsquared,
    }

    for var in x_vars:
        result[f'coef_{var}'] = model.params.get(var, np.nan)
        result[f'se_{var}'] = model.bse.get(var, np.nan)
        result[f'pval_{var}'] = model.pvalues.get(var, np.nan)

    log(f"  {label}: N={len(subset):,}, y_mean={y.mean():.4f}")
    for var in x_vars:
        coef = model.params.get(var, np.nan)
        se = model.bse.get(var, np.nan)
        pval = model.pvalues.get(var, np.nan)
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        log(f"    {var}: {coef:.6f} (SE={se:.6f}) p={pval:.4f} {sig}")

    return result

# All clicked items: does early position predict purchase?
# This tests H2 (effect of early exposure on purchase)
# Note: H1 is trivially satisfied since early items ARE attention by construction
log("\n--- H2: Purchase ~ Early Exposure (all clicks) ---")
r = run_lpm('was_purchased', ['early_exposure'], clicks_s, 'Y-M1: Purchase ~ EarlyExposure')
if r: results_table.append(r)

r = run_lpm('was_purchased', ['norm_position'], clicks_s, 'Y-M2: Purchase ~ NormPosition')
if r: results_table.append(r)

r = run_lpm('was_purchased', ['first_half'], clicks_s, 'Y-M3: Purchase ~ FirstHalf')
if r: results_table.append(r)

# Conditional: among buying sessions only
log("\n--- H3: Purchase ~ Early Exposure | Buying Sessions ---")
buying_clicks = clicks_s[clicks_s['session_has_purchase'] == 1].copy()
log(f"  Buying session clicks: {len(buying_clicks):,}")

r = run_lpm('was_purchased', ['early_exposure'], buying_clicks, 'Y-M4: Purchase ~ EarlyExposure | BuyingSession')
if r: results_table.append(r)

r = run_lpm('was_purchased', ['norm_position'], buying_clicks, 'Y-M5: Purchase ~ NormPosition | BuyingSession')
if r: results_table.append(r)

# ============================================================
# STEP D: Quantify the wedge
# ============================================================
log("\n" + "=" * 60)
log("QUANTIFYING THE ATTENTION-VALUE WEDGE (YOOCHOOSE)")
log("=" * 60)

# In YOOCHOOSE, all items in the dataset are clicked (attention = 1 for all)
# The wedge test is: does click order predict purchase?
# Early items get attention by definition; do they also get value?

purch_early = clicks_s.loc[clicks_s['early_exposure']==1, 'was_purchased'].mean()
purch_late = clicks_s.loc[clicks_s['early_exposure']==0, 'was_purchased'].mean()
purch_ratio = purch_early / purch_late if purch_late > 0 else np.nan

# Among buying sessions
purch_early_bs = buying_clicks.loc[buying_clicks['early_exposure']==1, 'was_purchased'].mean()
purch_late_bs = buying_clicks.loc[buying_clicks['early_exposure']==0, 'was_purchased'].mean()

wedge_results = {
    'purchase_rate_early_all': purch_early,
    'purchase_rate_late_all': purch_late,
    'purchase_diff_pp_all': (purch_early - purch_late) * 100,
    'purchase_ratio_all': purch_ratio,
    'purchase_rate_early_buying': purch_early_bs,
    'purchase_rate_late_buying': purch_late_bs,
    'purchase_diff_pp_buying': (purch_early_bs - purch_late_bs) * 100,
    'note': 'All items are clicked; wedge = whether early-clicked items have higher purchase rates',
}

log(f"All sessions: purchase_early={purch_early:.4f}, purchase_late={purch_late:.4f}, diff={(purch_early-purch_late)*100:.2f}pp")
log(f"Buying sessions: purchase_early={purch_early_bs:.4f}, purchase_late={purch_late_bs:.4f}, diff={(purch_early_bs-purch_late_bs)*100:.2f}pp")

# ============================================================
# STEP E: Robustness
# ============================================================
log("\n" + "=" * 60)
log("ROBUSTNESS")
log("=" * 60)

# E1: Session length heterogeneity
log("\n--- E1: Session length heterogeneity ---")
median_len = clicks_s['session_length'].median()
for group, label in [
    (clicks_s['session_length'] <= median_len, 'Short sessions'),
    (clicks_s['session_length'] > median_len, 'Long sessions'),
]:
    subset = clicks_s[group]
    pe = subset.loc[subset['early_exposure']==1, 'was_purchased'].mean()
    pl = subset.loc[subset['early_exposure']==0, 'was_purchased'].mean()
    log(f"  {label} (N={len(subset):,}): purch_early={pe:.4f}, purch_late={pl:.4f}, diff={(pe-pl)*100:.2f}pp")

# E2: Special offer items
log("\n--- E2: Special offer heterogeneity ---")
for group, label in [(1, 'Special offer'), (0, 'Regular')]:
    subset = clicks_s[clicks_s['is_special_offer'] == group]
    if len(subset) > 100:
        pe = subset.loc[subset['early_exposure']==1, 'was_purchased'].mean()
        pl = subset.loc[subset['early_exposure']==0, 'was_purchased'].mean()
        log(f"  {label} (N={len(subset):,}): purch_early={pe:.4f}, purch_late={pl:.4f}, diff={(pe-pl)*100:.2f}pp")

# ============================================================
# Save results
# ============================================================
results_df = pd.DataFrame(results_table)
results_df.to_csv(os.path.join(RESULTS_DIR, "yoochoose_regression_results.csv"), index=False)

with open(os.path.join(RESULTS_DIR, "yoochoose_wedge_summary.json"), 'w') as f:
    json.dump(wedge_results, f, indent=2, default=str)

with open(os.path.join(LOGS_DIR, "yoochoose_analysis.log"), 'w') as f:
    f.write('\n'.join(log_lines))

log("\nDone. Results saved.")
