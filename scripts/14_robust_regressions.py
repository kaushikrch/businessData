"""
Robust Regressions: Clustered Standard Errors
==============================================
Re-runs all key regressions with session-clustered standard errors
alongside HC1 (heteroskedasticity-robust) for comparison.

Motivation: A Marketing Science referee will demand clustering at the
session level because observations within the same session are not
independent — the same user's click/cart/purchase decisions are
correlated within a browsing session. HC1 standard errors will be
too small, inflating t-statistics.

Clustering variable:
  - REES46: user_session
  - Diginetica: queryId
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from datetime import datetime

np.random.seed(42)

# Paths
RAW_DIR_REES = "/home/user/businessData/data_raw/rees46"
RAW_DIR_DIGI = "/home/user/businessData/data_raw/diginetica"
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


def run_robust_regression(y_var, x_var, data, cluster_var, dataset, spec_label):
    """
    Run OLS (LPM) with both HC1 and clustered standard errors.
    Returns a dict with coefficients and SEs under both approaches.
    """
    subset = data[[y_var, x_var, cluster_var]].dropna().copy()
    if len(subset) < 100:
        log(f"  SKIP {spec_label}: too few observations ({len(subset)})")
        return None

    X = sm.add_constant(subset[[x_var]])
    y = subset[y_var]
    groups = subset[cluster_var]

    # HC1 (heteroskedasticity-robust)
    model_hc1 = sm.OLS(y, X).fit(cov_type='HC1')

    # Clustered standard errors
    model_cl = sm.OLS(y, X).fit(
        cov_type='cluster',
        cov_kwds={'groups': groups}
    )

    n_clusters = groups.nunique()

    result = {
        'dataset': dataset,
        'specification': spec_label,
        'outcome': y_var,
        'coef': model_hc1.params[x_var],
        'se_hc1': model_hc1.bse[x_var],
        'se_clustered': model_cl.bse[x_var],
        'pval_hc1': model_hc1.pvalues[x_var],
        'pval_clustered': model_cl.pvalues[x_var],
        'N': len(subset),
        'n_clusters': n_clusters,
    }

    se_ratio = result['se_clustered'] / result['se_hc1'] if result['se_hc1'] > 0 else np.nan
    sig_hc1 = "***" if result['pval_hc1'] < 0.001 else "**" if result['pval_hc1'] < 0.01 else "*" if result['pval_hc1'] < 0.05 else ""
    sig_cl = "***" if result['pval_clustered'] < 0.001 else "**" if result['pval_clustered'] < 0.01 else "*" if result['pval_clustered'] < 0.05 else ""

    log(f"  {spec_label}:")
    log(f"    coef={result['coef']:.6f}")
    log(f"    HC1:       SE={result['se_hc1']:.6f}, p={result['pval_hc1']:.4f} {sig_hc1}")
    log(f"    Clustered: SE={result['se_clustered']:.6f}, p={result['pval_clustered']:.4f} {sig_cl}")
    log(f"    SE ratio (clustered/HC1): {se_ratio:.3f}")
    log(f"    N={result['N']:,}, clusters={n_clusters:,}")

    return result


# ============================================================
# PART 1: REES46 — Cluster by user_session
# ============================================================
log("=" * 60)
log("PART 1: REES46 — Clustered by user_session")
log("=" * 60)

log("Loading REES46 data (2 shards)...")
df0 = pd.read_parquet(os.path.join(RAW_DIR_REES, "shard_0.parquet"))
df1 = pd.read_parquet(os.path.join(RAW_DIR_REES, "shard_1.parquet"))
df_rees = pd.concat([df0, df1], ignore_index=True)
del df0, df1
log(f"Total rows: {len(df_rees):,}")

# Parse timestamps and sort
log("Parsing timestamps...")
df_rees['event_time'] = pd.to_datetime(df_rees['event_time'], format='mixed', utc=True)
df_rees = df_rees.sort_values(['user_session', 'event_time']).reset_index(drop=True)

# Construct within-session view order
log("Constructing within-session view order...")
views = df_rees[df_rees['event_type'] == 'view'].copy()
views['view_rank'] = views.groupby('user_session').cumcount() + 1

session_view_counts = views.groupby('user_session')['view_rank'].transform('max')
views['session_length'] = session_view_counts

# Filter to sessions with >= 2 views
views = views[views['session_length'] >= 2].copy()
log(f"Views with session_length >= 2: {len(views):,}")

# Normalized position
views['norm_position'] = (views['view_rank'] - 1) / (views['session_length'] - 1)
views['early_exposure'] = (views['norm_position'] <= 0.25).astype(int)

# Merge cart/purchase indicators
carts = df_rees[df_rees['event_type'] == 'cart'][['user_session', 'product_id']].drop_duplicates()
carts['was_carted'] = 1
views = views.merge(carts, on=['user_session', 'product_id'], how='left')
views['was_carted'] = views['was_carted'].fillna(0).astype(int)

purchases = df_rees[df_rees['event_type'] == 'purchase'][['user_session', 'product_id']].drop_duplicates()
purchases['was_purchased'] = 1
views = views.merge(purchases, on=['user_session', 'product_id'], how='left')
views['was_purchased'] = views['was_purchased'].fillna(0).astype(int)

del df_rees  # free memory

log(f"REES46 views dataset: {views.shape}")
log(f"Cart rate: {views['was_carted'].mean():.4f}, Purchase rate: {views['was_purchased'].mean():.4f}")
log(f"Unique sessions: {views['user_session'].nunique():,}")

results_table = []

# Regression 1: Cart ~ early_exposure
log("\n--- R1: Cart ~ early_exposure ---")
r = run_robust_regression('was_carted', 'early_exposure', views, 'user_session',
                          'REES46', 'Cart ~ early_exposure')
if r: results_table.append(r)

# Regression 2: Purchase ~ early_exposure
log("\n--- R2: Purchase ~ early_exposure ---")
r = run_robust_regression('was_purchased', 'early_exposure', views, 'user_session',
                          'REES46', 'Purchase ~ early_exposure')
if r: results_table.append(r)

# Regression 3: Purchase|Cart ~ early_exposure
log("\n--- R3: Purchase|Cart ~ early_exposure ---")
carted_views = views[views['was_carted'] == 1].copy()
log(f"  Carted items: {len(carted_views):,}")
r = run_robust_regression('was_purchased', 'early_exposure', carted_views, 'user_session',
                          'REES46', 'Purchase|Cart ~ early_exposure')
if r: results_table.append(r)

del views, carted_views  # free memory

# ============================================================
# PART 2: Diginetica — Cluster by queryId
# ============================================================
log("\n" + "=" * 60)
log("PART 2: Diginetica — Clustered by queryId")
log("=" * 60)

log("Loading Diginetica data...")
queries = pd.read_csv(os.path.join(RAW_DIR_DIGI, "train-queries.csv"), sep=';')
queries = queries[queries['is.test'] == False].copy()
log(f"Train queries (non-test): {len(queries):,}")

clicks = pd.read_csv(os.path.join(RAW_DIR_DIGI, "train-clicks.csv"), sep=';')
purchases_digi = pd.read_csv(os.path.join(RAW_DIR_DIGI, "train-purchases.csv"), sep=';')
log(f"Clicks: {len(clicks):,}, Purchases: {len(purchases_digi):,}")

# Sample 10% of queries
sample_mask = np.random.random(len(queries)) < 0.10
queries_sample = queries[sample_mask].copy()
log(f"Sampled {len(queries_sample):,} queries from {len(queries):,}")

MAX_RANK = 30

# Build item-position rows
log("Building item-position dataset...")
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
        rows.append((row['queryId'], row['sessionId'], item_id, rank, len(items)))

df_digi = pd.DataFrame(rows, columns=['queryId', 'sessionId', 'item_id', 'rank_position', 'list_length'])
df_digi['rank_position'] = df_digi['rank_position'].astype(np.int16)
df_digi['list_length'] = df_digi['list_length'].astype(np.int16)
df_digi['queryId'] = df_digi['queryId'].astype(np.int32)
df_digi['sessionId'] = df_digi['sessionId'].astype(np.int32)
del rows
log(f"Item-position dataset: {df_digi.shape}")

# Mark clicks
click_pairs = clicks[['queryId', 'itemId']].drop_duplicates()
click_pairs.columns = ['queryId', 'item_id']
click_pairs['was_clicked'] = 1
df_digi = df_digi.merge(click_pairs, on=['queryId', 'item_id'], how='left')
df_digi['was_clicked'] = df_digi['was_clicked'].fillna(0).astype(int)

# Mark purchases
purch_pairs = purchases_digi[['sessionId', 'itemId']].drop_duplicates()
purch_pairs.columns = ['sessionId', 'item_id']
purch_pairs['was_purchased'] = 1
df_digi = df_digi.merge(purch_pairs, on=['sessionId', 'item_id'], how='left')
df_digi['was_purchased'] = df_digi['was_purchased'].fillna(0).astype(int)

# Construct exposure-order variables
df_digi['norm_rank'] = (df_digi['rank_position'] - 1) / (df_digi['list_length'] - 1)
df_digi.loc[df_digi['list_length'] == 1, 'norm_rank'] = 0
df_digi['early_exposure'] = (df_digi['norm_rank'] <= 0.25).astype(int)

log(f"Click rate: {df_digi['was_clicked'].mean():.4f}")
log(f"Purchase rate: {df_digi['was_purchased'].mean():.4f}")
log(f"Unique queries: {df_digi['queryId'].nunique():,}")

# Regression 1: Click ~ early_exposure
log("\n--- D-R1: Click ~ early_exposure ---")
r = run_robust_regression('was_clicked', 'early_exposure', df_digi, 'queryId',
                          'Diginetica', 'Click ~ early_exposure')
if r: results_table.append(r)

# Regression 2: Purchase ~ early_exposure
log("\n--- D-R2: Purchase ~ early_exposure ---")
r = run_robust_regression('was_purchased', 'early_exposure', df_digi, 'queryId',
                          'Diginetica', 'Purchase ~ early_exposure')
if r: results_table.append(r)

# Regression 3: Click ~ rank_position
log("\n--- D-R3: Click ~ rank_position ---")
r = run_robust_regression('was_clicked', 'rank_position', df_digi, 'queryId',
                          'Diginetica', 'Click ~ rank_position')
if r: results_table.append(r)

# Regression 4: Purchase ~ rank_position
log("\n--- D-R4: Purchase ~ rank_position ---")
r = run_robust_regression('was_purchased', 'rank_position', df_digi, 'queryId',
                          'Diginetica', 'Purchase ~ rank_position')
if r: results_table.append(r)


# ============================================================
# Save results
# ============================================================
log("\n" + "=" * 60)
log("SAVING RESULTS")
log("=" * 60)

results_df = pd.DataFrame(results_table)
out_path = os.path.join(RESULTS_DIR, "robust_regression_results.csv")
results_df.to_csv(out_path, index=False)
log(f"Saved {len(results_df)} rows to {out_path}")

# Summary: how much do clustered SEs differ from HC1?
log("\n--- Summary: SE inflation from clustering ---")
for _, row in results_df.iterrows():
    ratio = row['se_clustered'] / row['se_hc1'] if row['se_hc1'] > 0 else np.nan
    sig_change = ""
    if row['pval_hc1'] < 0.05 and row['pval_clustered'] >= 0.05:
        sig_change = " ** LOSES SIGNIFICANCE **"
    elif row['pval_hc1'] >= 0.05 and row['pval_clustered'] < 0.05:
        sig_change = " ** GAINS SIGNIFICANCE **"
    log(f"  {row['dataset']} | {row['specification']}: SE ratio={ratio:.3f}{sig_change}")

# Save log
with open(os.path.join(LOGS_DIR, "14_robust_regressions.log"), 'w') as f:
    f.write('\n'.join(log_lines))

log("\nDone.")
