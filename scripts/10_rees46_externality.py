"""
REES46: Within-Session Attention Externality Analysis
=====================================================
Tests Proposition 4 (Attention Externality): Does acting on item j
reduce engagement with subsequent items in the session?

Three tests:
1. Cart spillover: After carting item j, do subsequent items receive fewer carts?
2. Attention saturation: Does cumulative engagement reduce later item conversion?
3. Session-level: Do sessions with early carts have lower late-session conversion?
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import json
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

RAW_DIR = "/home/user/businessData/data_raw/rees46"
PROCESSED_DIR = "/home/user/businessData/data_processed"
RESULTS_DIR = "/home/user/businessData/results"
LOGS_DIR = "/home/user/businessData/logs"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

log_lines = []
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    log_lines.append(line)

# ============================================================
# STEP A: Load processed data
# ============================================================
log("Loading REES46 processed data...")

# Load both shards
shard_files = [
    os.path.join(RAW_DIR, "shard_0.parquet"),
    os.path.join(RAW_DIR, "shard_1.parquet"),
]

dfs = []
for f in shard_files:
    if os.path.exists(f):
        dfs.append(pd.read_parquet(f))
        log(f"  Loaded {f}: {len(dfs[-1]):,} rows")

if not dfs:
    log("ERROR: No shard files found. Trying processed parquet...")
    processed_file = os.path.join(PROCESSED_DIR, "rees46_views_processed.parquet")
    if os.path.exists(processed_file):
        df_raw = pd.read_parquet(processed_file)
        log(f"  Loaded processed: {len(df_raw):,} rows")
    else:
        raise FileNotFoundError("No REES46 data found")
else:
    df_raw = pd.concat(dfs, ignore_index=True)
    del dfs

log(f"Total raw events: {len(df_raw):,}")
log(f"Columns: {df_raw.columns.tolist()}")
log(f"Event types: {df_raw['event_type'].value_counts().to_dict()}")

# ============================================================
# STEP B: Build within-session ordered dataset
# ============================================================
log("\nBuilding within-session ordered dataset...")

# Filter to views only for position ordering
views = df_raw[df_raw['event_type'] == 'view'].copy()
views['event_time'] = pd.to_datetime(views['event_time'])

# Get cart and purchase events for outcome matching
carts = df_raw[df_raw['event_type'] == 'cart'][['user_session', 'product_id']].drop_duplicates()
carts['was_carted'] = 1
purchases_raw = df_raw[df_raw['event_type'] == 'purchase'][['user_session', 'product_id']].drop_duplicates()
purchases_raw['was_purchased'] = 1

del df_raw

# Sort and rank within session
views = views.sort_values(['user_session', 'event_time'])
views['view_rank'] = views.groupby('user_session').cumcount() + 1
views['session_length'] = views.groupby('user_session')['view_rank'].transform('max')

# Filter to multi-item sessions
views = views[views['session_length'] >= 3].copy()
log(f"Views in sessions with >= 3 items: {len(views):,}")
log(f"Sessions: {views['user_session'].nunique():,}")

# Merge outcomes
views = views.merge(carts, on=['user_session', 'product_id'], how='left')
views['was_carted'] = views['was_carted'].fillna(0).astype(np.int8)
views = views.merge(purchases_raw, on=['user_session', 'product_id'], how='left')
views['was_purchased'] = views['was_purchased'].fillna(0).astype(np.int8)

del carts, purchases_raw

log(f"Cart rate: {views['was_carted'].mean():.4f}")
log(f"Purchase rate: {views['was_purchased'].mean():.4f}")

# ============================================================
# STEP C: Construct externality variables
# ============================================================
log("\nConstructing attention externality variables...")

# For each item at position k, compute:
# 1. cumulative_carts_before: number of items carted at positions 1..k-1
# 2. cumulative_purchases_before: number of items purchased at positions 1..k-1
# 3. any_cart_before: indicator for at least one cart before position k
# 4. any_purchase_before: indicator for at least one purchase before position k

views = views.sort_values(['user_session', 'view_rank'])

# Cumulative carts and purchases BEFORE current item (shift by 1)
views['cum_carts'] = views.groupby('user_session')['was_carted'].cumsum()
views['cum_carts_before'] = views['cum_carts'] - views['was_carted']
views['any_cart_before'] = (views['cum_carts_before'] > 0).astype(np.int8)

views['cum_purchases'] = views.groupby('user_session')['was_purchased'].cumsum()
views['cum_purchases_before'] = views['cum_purchases'] - views['was_purchased']
views['any_purchase_before'] = (views['cum_purchases_before'] > 0).astype(np.int8)

# Normalized position
views['norm_position'] = (views['view_rank'] - 1) / (views['session_length'] - 1)

# Items remaining in session
views['items_remaining'] = views['session_length'] - views['view_rank']

log(f"Items with any_cart_before=1: {views['any_cart_before'].mean():.3f}")
log(f"Items with any_purchase_before=1: {views['any_purchase_before'].mean():.3f}")

# ============================================================
# STEP D: Test 1 — Cart spillover
# ============================================================
log("\n" + "=" * 60)
log("TEST 1: CART SPILLOVER (Proposition 4)")
log("=" * 60)
log("Does carting an earlier item reduce cart/purchase probability of later items?")

results_table = []

# Simple comparison: cart rate of items AFTER a cart event vs. before any cart
# Restrict to items at positions 2+ (position 1 cannot have anything before it)
later_items = views[views['view_rank'] > 1].copy()

log(f"\nItems at position 2+: {len(later_items):,}")
log(f"  With prior cart: {later_items['any_cart_before'].sum():,}")
log(f"  Without prior cart: {(1 - later_items['any_cart_before']).sum():,}")

# Raw comparison
for outcome in ['was_carted', 'was_purchased']:
    rate_after = later_items.loc[later_items['any_cart_before'] == 1, outcome].mean()
    rate_before = later_items.loc[later_items['any_cart_before'] == 0, outcome].mean()
    diff = rate_after - rate_before
    log(f"\n  {outcome} rate:")
    log(f"    After prior cart:  {rate_after:.4f}")
    log(f"    Before any cart:   {rate_before:.4f}")
    log(f"    Difference:        {diff*100:.3f} pp")

# Regression: outcome ~ any_cart_before + norm_position + log(session_length)
log("\nRegression: Cart/Purchase ~ any_cart_before + controls")
later_items['log_session_length'] = np.log(later_items['session_length'])

for outcome in ['was_carted', 'was_purchased']:
    X = later_items[['any_cart_before', 'norm_position', 'log_session_length']].copy()
    X = sm.add_constant(X)
    y = later_items[outcome]

    model = sm.OLS(y, X).fit(cov_type='HC1')

    coef = model.params['any_cart_before']
    se = model.bse['any_cart_before']
    pval = model.pvalues['any_cart_before']
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""

    log(f"\n  {outcome} ~ any_cart_before:")
    log(f"    Coefficient: {coef:.6f} (SE={se:.6f}, p={pval:.4f}) {sig}")
    log(f"    norm_position: {model.params['norm_position']:.6f}")
    log(f"    N={len(later_items):,}, R²={model.rsquared:.4f}")

    results_table.append({
        'test': 'cart_spillover',
        'specification': f'{outcome} ~ any_cart_before + controls',
        'coef_any_cart_before': coef,
        'se': se,
        'pval': pval,
        'coef_norm_position': model.params['norm_position'],
        'N': len(later_items),
        'R2': model.rsquared,
    })

# ============================================================
# STEP E: Test 2 — Cumulative attention saturation
# ============================================================
log("\n" + "=" * 60)
log("TEST 2: ATTENTION SATURATION")
log("=" * 60)
log("Does cumulative engagement (# carts so far) reduce later conversion?")

for outcome in ['was_carted', 'was_purchased']:
    X = later_items[['cum_carts_before', 'norm_position', 'log_session_length']].copy()
    X = sm.add_constant(X)
    y = later_items[outcome]

    model = sm.OLS(y, X).fit(cov_type='HC1')

    coef = model.params['cum_carts_before']
    se = model.bse['cum_carts_before']
    pval = model.pvalues['cum_carts_before']
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""

    log(f"\n  {outcome} ~ cum_carts_before:")
    log(f"    Coefficient: {coef:.6f} (SE={se:.6f}, p={pval:.4f}) {sig}")
    log(f"    Interpretation: Each additional prior cart changes {outcome} by {coef*100:.3f} pp")

    results_table.append({
        'test': 'attention_saturation',
        'specification': f'{outcome} ~ cum_carts_before + controls',
        'coef_any_cart_before': coef,
        'se': se,
        'pval': pval,
        'coef_norm_position': model.params['norm_position'],
        'N': len(later_items),
        'R2': model.rsquared,
    })

# ============================================================
# STEP F: Test 3 — Session-level early commitment
# ============================================================
log("\n" + "=" * 60)
log("TEST 3: SESSION-LEVEL EARLY COMMITMENT")
log("=" * 60)
log("Do sessions with early carts (first half) have lower late-session conversion?")

# Split each session into first half and second half
views['is_second_half'] = (views['norm_position'] > 0.5).astype(np.int8)

# Compute session-level indicator: any cart in first half?
first_half = views[views['is_second_half'] == 0]
session_early_cart = first_half.groupby('user_session')['was_carted'].max().reset_index()
session_early_cart.columns = ['user_session', 'early_cart_in_session']

# Get second-half items only
second_half = views[views['is_second_half'] == 1].copy()
second_half = second_half.merge(session_early_cart, on='user_session', how='left')
second_half['early_cart_in_session'] = second_half['early_cart_in_session'].fillna(0).astype(np.int8)

log(f"\nSecond-half items: {len(second_half):,}")
log(f"Sessions with early cart: {second_half['early_cart_in_session'].mean():.3f}")

for outcome in ['was_carted', 'was_purchased']:
    rate_with = second_half.loc[second_half['early_cart_in_session'] == 1, outcome].mean()
    rate_without = second_half.loc[second_half['early_cart_in_session'] == 0, outcome].mean()
    log(f"\n  Second-half {outcome} rate:")
    log(f"    Sessions with early cart:    {rate_with:.4f}")
    log(f"    Sessions without early cart: {rate_without:.4f}")
    log(f"    Difference: {(rate_with - rate_without)*100:.3f} pp")

    # Regression with controls
    X = second_half[['early_cart_in_session', 'norm_position', 'log_session_length']].copy()
    X = sm.add_constant(X)
    y = second_half[outcome]

    model = sm.OLS(y, X).fit(cov_type='HC1')

    coef = model.params['early_cart_in_session']
    se = model.bse['early_cart_in_session']
    pval = model.pvalues['early_cart_in_session']
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""

    log(f"    Regression: {coef:.6f} (SE={se:.6f}, p={pval:.4f}) {sig}")

    results_table.append({
        'test': 'session_early_commitment',
        'specification': f'2nd-half {outcome} ~ early_cart_in_session + controls',
        'coef_any_cart_before': coef,
        'se': se,
        'pval': pval,
        'coef_norm_position': model.params['norm_position'],
        'N': len(second_half),
        'R2': model.rsquared,
    })

# ============================================================
# STEP G: Heterogeneity — externality by session length
# ============================================================
log("\n" + "=" * 60)
log("HETEROGENEITY: EXTERNALITY BY SESSION LENGTH")
log("=" * 60)

median_length = views['session_length'].median()
log(f"Median session length: {median_length}")

for length_label, cond in [
    ('Short sessions', later_items['session_length'] <= median_length),
    ('Long sessions', later_items['session_length'] > median_length),
]:
    sub = later_items[cond].copy()
    log(f"\n{length_label} (N={len(sub):,}):")

    for outcome in ['was_carted', 'was_purchased']:
        rate_after = sub.loc[sub['any_cart_before'] == 1, outcome].mean()
        rate_before = sub.loc[sub['any_cart_before'] == 0, outcome].mean()
        diff = rate_after - rate_before
        log(f"  {outcome}: after_cart={rate_after:.4f}, before_cart={rate_before:.4f}, "
            f"diff={diff*100:.3f}pp")

# ============================================================
# Save results
# ============================================================
results_df = pd.DataFrame(results_table)
results_df.to_csv(os.path.join(RESULTS_DIR, "rees46_externality_results.csv"), index=False)

with open(os.path.join(LOGS_DIR, "rees46_externality.log"), 'w') as f:
    f.write('\n'.join(log_lines))

log(f"\nResults saved to {RESULTS_DIR}/rees46_externality_results.csv")
log("Done.")
