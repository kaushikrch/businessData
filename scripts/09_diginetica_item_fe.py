"""
Diginetica: Item Fixed Effects Analysis
========================================
Tests Proposition 2 (ranking calibration) with within-item variation.

KEY IDENTIFICATION STRATEGY: The same product appears at different rank
positions across different queries. By including item fixed effects, we
absorb time-invariant product quality q_j and isolate the pure positional
effect of rank on clicks vs. purchases.

If rank affects clicks but not purchases (conditional on item FE),
the wedge is driven by positional salience b(r), not quality sorting.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import json
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

RAW_DIR = "/home/user/businessData/data_raw/diginetica"
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
# STEP A: Load and construct item-position dataset
# ============================================================
log("Loading Diginetica data...")

queries = pd.read_csv(os.path.join(RAW_DIR, "train-queries.csv"), sep=';')
queries = queries[queries['is.test'] == False].copy()
clicks = pd.read_csv(os.path.join(RAW_DIR, "train-clicks.csv"), sep=';')
purchases = pd.read_csv(os.path.join(RAW_DIR, "train-purchases.csv"), sep=';')
products = pd.read_csv(os.path.join(RAW_DIR, "products.csv"), sep=';')

log(f"Train queries: {len(queries):,}")
log(f"Clicks: {len(clicks):,}, Purchases: {len(purchases):,}")

# Use 20% sample for tractability with item FE
np.random.seed(42)
sample_mask = np.random.random(len(queries)) < 0.20
queries_sample = queries[sample_mask].copy()
log(f"Sampled {len(queries_sample):,} queries")

MAX_RANK = 30

rows = []
for _, row in queries_sample.iterrows():
    if pd.isna(row['items']) or row['items'] == '':
        continue
    items_str = str(row['items']).strip()
    if not items_str:
        continue
    items = [int(x.strip()) for x in items_str.split(',') if x.strip()]
    items = items[:MAX_RANK]
    for rank, item_id in enumerate(items, 1):
        rows.append((
            row['queryId'],
            row['sessionId'],
            item_id,
            rank,
            len(items),
        ))

df = pd.DataFrame(rows, columns=[
    'queryId', 'sessionId', 'item_id', 'rank_position', 'list_length'
])
df['rank_position'] = df['rank_position'].astype(np.int16)
df['list_length'] = df['list_length'].astype(np.int16)
del rows

log(f"Item-position dataset: {len(df):,} rows")

# Mark clicks and purchases
click_pairs = clicks[['queryId', 'itemId']].drop_duplicates()
click_pairs.columns = ['queryId', 'item_id']
click_pairs['was_clicked'] = 1
df = df.merge(click_pairs, on=['queryId', 'item_id'], how='left')
df['was_clicked'] = df['was_clicked'].fillna(0).astype(np.int8)

purch_pairs = purchases[['sessionId', 'itemId']].drop_duplicates()
purch_pairs.columns = ['sessionId', 'item_id']
purch_pairs['was_purchased'] = 1
df = df.merge(purch_pairs, on=['sessionId', 'item_id'], how='left')
df['was_purchased'] = df['was_purchased'].fillna(0).astype(np.int8)

# Exposure variables
df['norm_rank'] = (df['rank_position'] - 1) / (df['list_length'] - 1)
df.loc[df['list_length'] == 1, 'norm_rank'] = 0
df['early_exposure'] = (df['norm_rank'] <= 0.25).astype(np.int8)
df['top5'] = (df['rank_position'] <= 5).astype(np.int8)

log(f"Click rate: {df['was_clicked'].mean():.4f}")
log(f"Purchase rate: {df['was_purchased'].mean():.6f}")

# ============================================================
# STEP B: Check item repetition across queries
# ============================================================
log("\n" + "=" * 60)
log("ITEM REPETITION ANALYSIS")
log("=" * 60)

item_counts = df.groupby('item_id').agg(
    n_appearances=('queryId', 'count'),
    n_queries=('queryId', 'nunique'),
    rank_min=('rank_position', 'min'),
    rank_max=('rank_position', 'max'),
    rank_std=('rank_position', 'std'),
    click_rate=('was_clicked', 'mean'),
    purch_rate=('was_purchased', 'mean'),
).reset_index()

log(f"Total unique items: {len(item_counts):,}")
log(f"Items appearing in >= 2 queries: {(item_counts['n_queries'] >= 2).sum():,}")
log(f"Items appearing in >= 5 queries: {(item_counts['n_queries'] >= 5).sum():,}")
log(f"Items appearing in >= 10 queries: {(item_counts['n_queries'] >= 10).sum():,}")
log(f"Items appearing in >= 30 queries: {(item_counts['n_queries'] >= 30).sum():,}")

# Items with rank variation (key for identification)
items_with_var = item_counts[item_counts['rank_std'] > 0]
log(f"\nItems with rank variation (rank_std > 0): {len(items_with_var):,}")
log(f"Mean rank std: {items_with_var['rank_std'].mean():.2f}")
log(f"Mean rank range: {(items_with_var['rank_max'] - items_with_var['rank_min']).mean():.1f}")

# ============================================================
# STEP C: Item Fixed Effects Regressions
# ============================================================
log("\n" + "=" * 60)
log("ITEM FIXED EFFECTS REGRESSIONS")
log("=" * 60)

# Filter to items appearing in at least 5 queries (sufficient variation for FE)
MIN_APPEARANCES = 5
frequent_items = item_counts[item_counts['n_queries'] >= MIN_APPEARANCES]['item_id']
df_fe = df[df['item_id'].isin(frequent_items)].copy()
log(f"Items with >= {MIN_APPEARANCES} query appearances: {len(frequent_items):,}")
log(f"Observations in FE sample: {len(df_fe):,}")
log(f"Click rate in FE sample: {df_fe['was_clicked'].mean():.4f}")
log(f"Purchase rate in FE sample: {df_fe['was_purchased'].mean():.6f}")

results_table = []

# --- Method 1: Within-item demeaning (Mundlak approach) ---
log("\n--- Method 1: Within-item demeaned regressions ---")

# Demean variables within item
for var in ['was_clicked', 'was_purchased', 'rank_position', 'norm_rank',
            'early_exposure', 'top5']:
    item_means = df_fe.groupby('item_id')[var].transform('mean')
    df_fe[f'{var}_dm'] = df_fe[var] - item_means

# Click ~ rank_position (demeaned)
log("\nClick ~ rank_position (within-item demeaned):")
X = sm.add_constant(df_fe['rank_position_dm'])
model = sm.OLS(df_fe['was_clicked_dm'], X).fit(cov_type='HC1')
log(f"  rank_position coef: {model.params['rank_position_dm']:.6f} "
    f"(SE={model.bse['rank_position_dm']:.6f}, p={model.pvalues['rank_position_dm']:.4f})")
results_table.append({
    'specification': 'Click ~ rank_position (item FE, demeaned)',
    'coef': model.params['rank_position_dm'],
    'se': model.bse['rank_position_dm'],
    'pval': model.pvalues['rank_position_dm'],
    'N': len(df_fe),
    'n_items': len(frequent_items),
})

# Purchase ~ rank_position (demeaned)
log("\nPurchase ~ rank_position (within-item demeaned):")
model = sm.OLS(df_fe['was_purchased_dm'], X).fit(cov_type='HC1')
log(f"  rank_position coef: {model.params['rank_position_dm']:.6f} "
    f"(SE={model.bse['rank_position_dm']:.6f}, p={model.pvalues['rank_position_dm']:.4f})")
results_table.append({
    'specification': 'Purchase ~ rank_position (item FE, demeaned)',
    'coef': model.params['rank_position_dm'],
    'se': model.bse['rank_position_dm'],
    'pval': model.pvalues['rank_position_dm'],
    'N': len(df_fe),
    'n_items': len(frequent_items),
})

# Click ~ early_exposure (demeaned)
log("\nClick ~ early_exposure (within-item demeaned):")
X = sm.add_constant(df_fe['early_exposure_dm'])
model = sm.OLS(df_fe['was_clicked_dm'], X).fit(cov_type='HC1')
log(f"  early_exposure coef: {model.params['early_exposure_dm']:.6f} "
    f"(SE={model.bse['early_exposure_dm']:.6f}, p={model.pvalues['early_exposure_dm']:.4f})")
results_table.append({
    'specification': 'Click ~ early_exposure (item FE, demeaned)',
    'coef': model.params['early_exposure_dm'],
    'se': model.bse['early_exposure_dm'],
    'pval': model.pvalues['early_exposure_dm'],
    'N': len(df_fe),
    'n_items': len(frequent_items),
})

# Purchase ~ early_exposure (demeaned)
log("\nPurchase ~ early_exposure (within-item demeaned):")
model = sm.OLS(df_fe['was_purchased_dm'], X).fit(cov_type='HC1')
log(f"  early_exposure coef: {model.params['early_exposure_dm']:.6f} "
    f"(SE={model.bse['early_exposure_dm']:.6f}, p={model.pvalues['early_exposure_dm']:.4f})")
results_table.append({
    'specification': 'Purchase ~ early_exposure (item FE, demeaned)',
    'coef': model.params['early_exposure_dm'],
    'se': model.bse['early_exposure_dm'],
    'pval': model.pvalues['early_exposure_dm'],
    'N': len(df_fe),
    'n_items': len(frequent_items),
})

# --- Method 2: Top5 (sharper cutoff, demeaned) ---
log("\nClick ~ top5 (within-item demeaned):")
X = sm.add_constant(df_fe['top5_dm'])
model = sm.OLS(df_fe['was_clicked_dm'], X).fit(cov_type='HC1')
log(f"  top5 coef: {model.params['top5_dm']:.6f} "
    f"(SE={model.bse['top5_dm']:.6f}, p={model.pvalues['top5_dm']:.4f})")
results_table.append({
    'specification': 'Click ~ top5 (item FE, demeaned)',
    'coef': model.params['top5_dm'],
    'se': model.bse['top5_dm'],
    'pval': model.pvalues['top5_dm'],
    'N': len(df_fe),
    'n_items': len(frequent_items),
})

log("\nPurchase ~ top5 (within-item demeaned):")
model = sm.OLS(df_fe['was_purchased_dm'], X).fit(cov_type='HC1')
log(f"  top5 coef: {model.params['top5_dm']:.6f} "
    f"(SE={model.bse['top5_dm']:.6f}, p={model.pvalues['top5_dm']:.4f})")
results_table.append({
    'specification': 'Purchase ~ top5 (item FE, demeaned)',
    'coef': model.params['top5_dm'],
    'se': model.bse['top5_dm'],
    'pval': model.pvalues['top5_dm'],
    'N': len(df_fe),
    'n_items': len(frequent_items),
})

# ============================================================
# STEP D: Conditional conversion with item FE
# ============================================================
log("\n" + "=" * 60)
log("CONDITIONAL CONVERSION WITH ITEM FE")
log("=" * 60)

clicked_fe = df_fe[df_fe['was_clicked'] == 1].copy()
log(f"Clicked items in FE sample: {len(clicked_fe):,}")

# Only use items with >=2 clicks for conditional FE
item_click_counts = clicked_fe.groupby('item_id').size()
items_multi_click = item_click_counts[item_click_counts >= 2].index
clicked_fe_multi = clicked_fe[clicked_fe['item_id'].isin(items_multi_click)].copy()
log(f"Items with >= 2 clicks: {len(items_multi_click):,}")
log(f"Observations: {len(clicked_fe_multi):,}")

if len(clicked_fe_multi) > 100:
    for var in ['was_purchased', 'rank_position', 'early_exposure']:
        item_means = clicked_fe_multi.groupby('item_id')[var].transform('mean')
        clicked_fe_multi[f'{var}_dm'] = clicked_fe_multi[var] - item_means

    log("\nPurchase|Click ~ rank_position (item FE, demeaned):")
    X = sm.add_constant(clicked_fe_multi['rank_position_dm'])
    model = sm.OLS(clicked_fe_multi['was_purchased_dm'], X).fit(cov_type='HC1')
    log(f"  rank_position coef: {model.params['rank_position_dm']:.6f} "
        f"(SE={model.bse['rank_position_dm']:.6f}, p={model.pvalues['rank_position_dm']:.4f})")
    results_table.append({
        'specification': 'Purchase|Click ~ rank_position (item FE)',
        'coef': model.params['rank_position_dm'],
        'se': model.bse['rank_position_dm'],
        'pval': model.pvalues['rank_position_dm'],
        'N': len(clicked_fe_multi),
        'n_items': len(items_multi_click),
    })

    log("\nPurchase|Click ~ early_exposure (item FE, demeaned):")
    X = sm.add_constant(clicked_fe_multi['early_exposure_dm'])
    model = sm.OLS(clicked_fe_multi['was_purchased_dm'], X).fit(cov_type='HC1')
    log(f"  early_exposure coef: {model.params['early_exposure_dm']:.6f} "
        f"(SE={model.bse['early_exposure_dm']:.6f}, p={model.pvalues['early_exposure_dm']:.4f})")
    results_table.append({
        'specification': 'Purchase|Click ~ early_exposure (item FE)',
        'coef': model.params['early_exposure_dm'],
        'se': model.bse['early_exposure_dm'],
        'pval': model.pvalues['early_exposure_dm'],
        'N': len(clicked_fe_multi),
        'n_items': len(items_multi_click),
    })

# ============================================================
# STEP E: Comparison — no FE vs item FE
# ============================================================
log("\n" + "=" * 60)
log("COMPARISON: NO FE vs ITEM FE")
log("=" * 60)

# No FE baseline on same sample
log("\nBaseline (no FE) on FE-eligible sample:")
X = sm.add_constant(df_fe['rank_position'])
model_no_fe_click = sm.OLS(df_fe['was_clicked'], X).fit(cov_type='HC1')
log(f"  Click ~ rank_position (no FE): {model_no_fe_click.params['rank_position']:.6f} "
    f"(SE={model_no_fe_click.bse['rank_position']:.6f})")

model_no_fe_purch = sm.OLS(df_fe['was_purchased'], X).fit(cov_type='HC1')
log(f"  Purchase ~ rank_position (no FE): {model_no_fe_purch.params['rank_position']:.6f} "
    f"(SE={model_no_fe_purch.bse['rank_position']:.6f})")

results_table.append({
    'specification': 'Click ~ rank_position (no FE, FE sample)',
    'coef': model_no_fe_click.params['rank_position'],
    'se': model_no_fe_click.bse['rank_position'],
    'pval': model_no_fe_click.pvalues['rank_position'],
    'N': len(df_fe),
    'n_items': len(frequent_items),
})
results_table.append({
    'specification': 'Purchase ~ rank_position (no FE, FE sample)',
    'coef': model_no_fe_purch.params['rank_position'],
    'se': model_no_fe_purch.bse['rank_position'],
    'pval': model_no_fe_purch.pvalues['rank_position'],
    'N': len(df_fe),
    'n_items': len(frequent_items),
})

# ============================================================
# Save results
# ============================================================
results_df = pd.DataFrame(results_table)
results_df.to_csv(os.path.join(RESULTS_DIR, "diginetica_item_fe_results.csv"), index=False)

with open(os.path.join(LOGS_DIR, "diginetica_item_fe.log"), 'w') as f:
    f.write('\n'.join(log_lines))

log(f"\nResults saved to {RESULTS_DIR}/diginetica_item_fe_results.csv")
log("Done.")
