"""
Diginetica (CIKM Cup 2016): Attention-Value Wedge Analysis
============================================================
Tests H1-H5 using SEARCH ENGINE RANKED exposure order.

KEY ADVANTAGE: The 'items' column in train-queries contains the platform's
default ranking of products. This is closer to exogenous exposure order than
user-chosen browsing sequences (REES46/YOOCHOOSE), because the ranking is
determined by the search engine algorithm, not user choice.

Exposure order: position in the search engine's ranked result list.
Attention: click (appears in train-clicks.csv for that query).
Downstream: purchase (appears in train-purchases.csv for that session).

Identification note:
  Search engine rankings are NOT random — they are algorithmic and likely
  correlated with item quality. However, conditional on appearing in the
  same result set (same query), position variation is driven by the algorithm's
  ranking function, which provides arguably better identification than
  user-chosen browsing order. This is still not a randomized experiment,
  but it is a meaningful improvement in identification strength.
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
# STEP A: Load and parse data
# ============================================================
log("Loading Diginetica data...")

# Queries with ranked item lists
queries = pd.read_csv(os.path.join(RAW_DIR, "train-queries.csv"), sep=';')
queries = queries[queries['is.test'] == False].copy()
log(f"Train queries: {len(queries):,}")

# Clicks on search results
clicks = pd.read_csv(os.path.join(RAW_DIR, "train-clicks.csv"), sep=';')
log(f"Clicks: {len(clicks):,}")

# Purchases
purchases = pd.read_csv(os.path.join(RAW_DIR, "train-purchases.csv"), sep=';')
log(f"Purchases: {len(purchases):,}")

# Products (for price info)
products = pd.read_csv(os.path.join(RAW_DIR, "products.csv"), sep=';')
log(f"Products: {len(products):,}")

# Item views (browsing after clicking from SERP)
item_views = pd.read_csv(os.path.join(RAW_DIR, "train-item-views.csv"), sep=';')
log(f"Item views: {len(item_views):,}")

# ============================================================
# STEP B: Construct item-level dataset from ranked lists
# ============================================================
log("Constructing item-position dataset from search rankings...")

# Each query has a ranked list of items. We need to explode this into
# item-level rows with their rank position.
# MEMORY OPTIMIZATION: Sample 10% of queries and cap result lists at top 30 items.
# The wedge hypothesis is about early vs late exposure, so top-30 captures the
# economically relevant variation while keeping memory manageable.
np.random.seed(42)
sample_mask = np.random.random(len(queries)) < 0.10
queries_sample = queries[sample_mask].copy()
log(f"  Sampled {len(queries_sample):,} queries from {len(queries):,}")
MAX_RANK = 30  # Only keep top 30 items per query

rows = []
for _, row in queries_sample.iterrows():
    if pd.isna(row['items']) or row['items'] == '':
        continue
    items_str = str(row['items']).strip()
    if not items_str:
        continue
    items = [int(x.strip()) for x in items_str.split(',') if x.strip()]
    items = items[:MAX_RANK]  # Cap at top MAX_RANK
    for rank, item_id in enumerate(items, 1):
        rows.append((
            row['queryId'],
            row['sessionId'],
            row['timeframe'],
            pd.notna(row['searchstring.tokens']) and row['searchstring.tokens'] != '',
            item_id,
            rank,
            len(items),
        ))

log(f"  Building DataFrame from {len(rows):,} item-position pairs...")
df = pd.DataFrame(rows, columns=[
    'queryId', 'sessionId', 'timeframe', 'has_search_tokens',
    'item_id', 'rank_position', 'list_length'
])
# Use smaller dtypes
df['rank_position'] = df['rank_position'].astype(np.int16)
df['list_length'] = df['list_length'].astype(np.int16)
df['queryId'] = df['queryId'].astype(np.int32)
df['sessionId'] = df['sessionId'].astype(np.int32)
del rows
log(f"  Item-position dataset shape: {df.shape}")

# ============================================================
# STEP B2: Mark which items were clicked and purchased
# ============================================================
log("Marking clicks and purchases...")

# Clicks: match by queryId + itemId
click_pairs = clicks[['queryId', 'itemId']].drop_duplicates()
click_pairs.columns = ['queryId', 'item_id']
click_pairs['was_clicked'] = 1
df = df.merge(click_pairs, on=['queryId', 'item_id'], how='left')
df['was_clicked'] = df['was_clicked'].fillna(0).astype(int)

# Purchases: match by sessionId + itemId
purch_pairs = purchases[['sessionId', 'itemId']].drop_duplicates()
purch_pairs.columns = ['sessionId', 'item_id']
purch_pairs['was_purchased'] = 1
df = df.merge(purch_pairs, on=['sessionId', 'item_id'], how='left')
df['was_purchased'] = df['was_purchased'].fillna(0).astype(int)

# Item views (clicked through to product page): match by sessionId + itemId
view_pairs = item_views[['sessionId', 'itemId']].drop_duplicates()
view_pairs.columns = ['sessionId', 'item_id']
view_pairs['was_viewed'] = 1
df = df.merge(view_pairs, on=['sessionId', 'item_id'], how='left')
df['was_viewed'] = df['was_viewed'].fillna(0).astype(int)

# Merge product price
products_price = products[['itemId', 'pricelog2']].copy()
products_price.columns = ['item_id', 'pricelog2']
df = df.merge(products_price, on='item_id', how='left')

log(f"Click rate: {df['was_clicked'].mean():.4f}")
log(f"View rate: {df['was_viewed'].mean():.4f}")
log(f"Purchase rate: {df['was_purchased'].mean():.4f}")

# ============================================================
# STEP B3: Construct exposure-order variables
# ============================================================
log("Constructing exposure-order variables...")

# Normalized rank position: 0 = rank 1, 1 = last rank
df['norm_rank'] = (df['rank_position'] - 1) / (df['list_length'] - 1)
df.loc[df['list_length'] == 1, 'norm_rank'] = 0

# Early exposure: top quartile of ranked list
df['early_exposure'] = (df['norm_rank'] <= 0.25).astype(int)
df['top5'] = (df['rank_position'] <= 5).astype(int)
df['top3'] = (df['rank_position'] <= 3).astype(int)

log(f"Dataset shape: {df.shape}")
log(f"Early exposure (top quartile): {df['early_exposure'].mean():.3f}")

# Descriptive stats by position
log("\nDescriptive stats by early/late exposure:")
for group in [1, 0]:
    sub = df[df['early_exposure'] == group]
    label = 'Early' if group == 1 else 'Late'
    log(f"  {label}: click={sub['was_clicked'].mean():.4f}, "
        f"view={sub['was_viewed'].mean():.4f}, "
        f"purchase={sub['was_purchased'].mean():.4f}, "
        f"N={len(sub):,}")

# ============================================================
# STEP C: Main regressions
# ============================================================
log("\n" + "=" * 60)
log("MAIN REGRESSIONS")
log("=" * 60)

results_table = []

def run_lpm(y_var, x_vars, data, label, controls=None):
    """Run LPM with robust SEs."""
    all_vars = [y_var] + x_vars + (controls or [])
    subset = data.dropna(subset=all_vars)
    if len(subset) < 100:
        log(f"  SKIP {label}: too few obs ({len(subset)})")
        return None

    X = subset[x_vars].copy()
    if controls:
        X = pd.concat([X, subset[controls]], axis=1)
    X = sm.add_constant(X)
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

# --- H1: Click ~ Rank Position ---
log("\n--- H1: Click (Attention) ~ Search Rank Position ---")
r = run_lpm('was_clicked', ['early_exposure'], df, 'D-M1a: Click ~ EarlyExposure')
if r: results_table.append(r)

r = run_lpm('was_clicked', ['top5'], df, 'D-M1b: Click ~ Top5')
if r: results_table.append(r)

r = run_lpm('was_clicked', ['norm_rank'], df, 'D-M1c: Click ~ NormRank')
if r: results_table.append(r)

# With price control
r = run_lpm('was_clicked', ['early_exposure'], df, 'D-M1d: Click ~ EarlyExposure + price',
            controls=['pricelog2'])
if r: results_table.append(r)

# --- H2: Purchase ~ Rank Position ---
log("\n--- H2: Purchase ~ Search Rank Position ---")
r = run_lpm('was_purchased', ['early_exposure'], df, 'D-M2a: Purchase ~ EarlyExposure')
if r: results_table.append(r)

r = run_lpm('was_purchased', ['top5'], df, 'D-M2b: Purchase ~ Top5')
if r: results_table.append(r)

r = run_lpm('was_purchased', ['norm_rank'], df, 'D-M2c: Purchase ~ NormRank')
if r: results_table.append(r)

r = run_lpm('was_purchased', ['early_exposure'], df, 'D-M2d: Purchase ~ EarlyExposure + price',
            controls=['pricelog2'])
if r: results_table.append(r)

# --- H3: Purchase | Click (conditional conversion) ---
log("\n--- H3: Purchase | Click ~ Rank Position ---")
clicked_items = df[df['was_clicked'] == 1].copy()
log(f"  Clicked items: {len(clicked_items):,}")

r = run_lpm('was_purchased', ['early_exposure'], clicked_items, 'D-M3a: Purchase|Click ~ EarlyExposure')
if r: results_table.append(r)

r = run_lpm('was_purchased', ['norm_rank'], clicked_items, 'D-M3b: Purchase|Click ~ NormRank')
if r: results_table.append(r)

r = run_lpm('was_purchased', ['early_exposure'], clicked_items, 'D-M3c: Purchase|Click ~ EarlyExposure + price',
            controls=['pricelog2'])
if r: results_table.append(r)

# --- H3 alt: Purchase | View (conditional on viewing product page) ---
log("\n--- H3 alt: Purchase | View ~ Rank Position ---")
viewed_items = df[df['was_viewed'] == 1].copy()
log(f"  Viewed items: {len(viewed_items):,}")

r = run_lpm('was_purchased', ['early_exposure'], viewed_items, 'D-M3d: Purchase|View ~ EarlyExposure')
if r: results_table.append(r)

r = run_lpm('was_purchased', ['norm_rank'], viewed_items, 'D-M3e: Purchase|View ~ NormRank')
if r: results_table.append(r)

# ============================================================
# STEP D: Quantify the wedge
# ============================================================
log("\n" + "=" * 60)
log("QUANTIFYING THE ATTENTION-VALUE WEDGE (DIGINETICA)")
log("=" * 60)

# Click (attention) effect
click_early = df.loc[df['early_exposure']==1, 'was_clicked'].mean()
click_late = df.loc[df['early_exposure']==0, 'was_clicked'].mean()
click_ratio = click_early / click_late if click_late > 0 else np.nan

# Purchase (value) effect
purch_early = df.loc[df['early_exposure']==1, 'was_purchased'].mean()
purch_late = df.loc[df['early_exposure']==0, 'was_purchased'].mean()
purch_ratio = purch_early / purch_late if purch_late > 0 else np.nan

# Conditional purchase | click
cond_early = clicked_items.loc[clicked_items['early_exposure']==1, 'was_purchased'].mean()
cond_late = clicked_items.loc[clicked_items['early_exposure']==0, 'was_purchased'].mean()
cond_ratio = cond_early / cond_late if cond_late > 0 else np.nan

wedge_results = {
    'click_rate_early': click_early,
    'click_rate_late': click_late,
    'click_diff_pp': (click_early - click_late) * 100,
    'click_ratio': click_ratio,
    'purchase_rate_early': purch_early,
    'purchase_rate_late': purch_late,
    'purchase_diff_pp': (purch_early - purch_late) * 100,
    'purchase_ratio': purch_ratio,
    'cond_purchase_early': cond_early,
    'cond_purchase_late': cond_late,
    'cond_purchase_diff_pp': (cond_early - cond_late) * 100,
    'cond_purchase_ratio': cond_ratio,
    'wedge_click_ratio_minus_purchase_ratio': click_ratio - purch_ratio if not np.isnan(click_ratio) and not np.isnan(purch_ratio) else np.nan,
}

log(f"Click rate: early={click_early:.4f}, late={click_late:.4f}, diff={(click_early-click_late)*100:.2f}pp, ratio={click_ratio:.3f}")
log(f"Purchase rate: early={purch_early:.4f}, late={purch_late:.4f}, diff={(purch_early-purch_late)*100:.2f}pp, ratio={purch_ratio:.3f}")
log(f"Purchase|Click: early={cond_early:.4f}, late={cond_late:.4f}, diff={(cond_early-cond_late)*100:.2f}pp, ratio={cond_ratio:.3f}")
log(f"WEDGE (click_ratio - purchase_ratio): {wedge_results['wedge_click_ratio_minus_purchase_ratio']:.3f}")

# ============================================================
# STEP E: Robustness & Heterogeneity
# ============================================================
log("\n" + "=" * 60)
log("ROBUSTNESS AND HETEROGENEITY")
log("=" * 60)

# --- E1: Top-3 vs rest (sharper cutoff) ---
log("\n--- E1: Top-3 position analysis ---")
for group, label in [(1, 'Top 3'), (0, 'Rank 4+')]:
    sub = df[df['top3'] == group]
    ce = sub['was_clicked'].mean()
    pe = sub['was_purchased'].mean()
    log(f"  {label} (N={len(sub):,}): click={ce:.4f}, purchase={pe:.4f}")

# --- E2: Query type heterogeneity (H4/H5 proxy) ---
log("\n--- E2: Search query vs category browse (intent proxy) ---")
# Sessions with search tokens = text search (potentially lower intent / higher uncertainty)
# Sessions without = category browse (potentially higher intent)
for group, label in [(True, 'Text search (lower intent)'), (False, 'Category browse (higher intent)')]:
    sub = df[df['has_search_tokens'] == group]
    if len(sub) < 100:
        log(f"  {label}: too few observations")
        continue
    ce = sub.loc[sub['early_exposure']==1, 'was_clicked'].mean()
    cl = sub.loc[sub['early_exposure']==0, 'was_clicked'].mean()
    pe = sub.loc[sub['early_exposure']==1, 'was_purchased'].mean()
    pl = sub.loc[sub['early_exposure']==0, 'was_purchased'].mean()
    log(f"  {label} (N={len(sub):,}):")
    log(f"    Click diff: {(ce-cl)*100:.2f}pp (early={ce:.4f}, late={cl:.4f})")
    log(f"    Purchase diff: {(pe-pl)*100:.2f}pp (early={pe:.4f}, late={pl:.4f})")

# --- E3: List length as uncertainty proxy ---
log("\n--- E3: List length heterogeneity ---")
median_len = df['list_length'].median()
for cond, label in [
    (df['list_length'] <= median_len, 'Short result list'),
    (df['list_length'] > median_len, 'Long result list'),
]:
    sub = df[cond]
    ce = sub.loc[sub['early_exposure']==1, 'was_clicked'].mean()
    cl = sub.loc[sub['early_exposure']==0, 'was_clicked'].mean()
    pe = sub.loc[sub['early_exposure']==1, 'was_purchased'].mean()
    pl = sub.loc[sub['early_exposure']==0, 'was_purchased'].mean()
    log(f"  {label} (N={len(sub):,}):")
    log(f"    Click diff: {(ce-cl)*100:.2f}pp")
    log(f"    Purchase diff: {(pe-pl)*100:.2f}pp")

# --- E4: Price heterogeneity ---
log("\n--- E4: Price heterogeneity ---")
price_valid = df.dropna(subset=['pricelog2'])
if len(price_valid) > 0:
    median_price = price_valid['pricelog2'].median()
    for cond, label in [
        (price_valid['pricelog2'] > median_price, 'High price'),
        (price_valid['pricelog2'] <= median_price, 'Low price'),
    ]:
        sub = price_valid[cond]
        ce = sub.loc[sub['early_exposure']==1, 'was_clicked'].mean()
        cl = sub.loc[sub['early_exposure']==0, 'was_clicked'].mean()
        pe = sub.loc[sub['early_exposure']==1, 'was_purchased'].mean()
        pl = sub.loc[sub['early_exposure']==0, 'was_purchased'].mean()
        log(f"  {label} (N={len(sub):,}):")
        log(f"    Click diff: {(ce-cl)*100:.2f}pp")
        log(f"    Purchase diff: {(pe-pl)*100:.2f}pp")

# --- E5: Position gradient (rank-by-rank) ---
log("\n--- E5: Click and purchase rates by rank position (top 20) ---")
pos_stats = df[df['rank_position'] <= 20].groupby('rank_position').agg(
    click_rate=('was_clicked', 'mean'),
    purchase_rate=('was_purchased', 'mean'),
    n=('was_clicked', 'count'),
).reset_index()
pos_stats.to_csv(os.path.join(RESULTS_DIR, "diginetica_position_gradient.csv"), index=False)

for _, row in pos_stats.iterrows():
    log(f"  Rank {int(row['rank_position']):2d}: click={row['click_rate']:.4f}, "
        f"purchase={row['purchase_rate']:.5f}, N={int(row['n']):,}")

# ============================================================
# Save all results
# ============================================================
results_df = pd.DataFrame(results_table)
results_df.to_csv(os.path.join(RESULTS_DIR, "diginetica_regression_results.csv"), index=False)

with open(os.path.join(RESULTS_DIR, "diginetica_wedge_summary.json"), 'w') as f:
    json.dump(wedge_results, f, indent=2, default=str)

with open(os.path.join(LOGS_DIR, "diginetica_analysis.log"), 'w') as f:
    f.write('\n'.join(log_lines))

log("\nDone. Results saved to results/ directory.")
