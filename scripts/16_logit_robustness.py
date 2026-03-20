"""
Logit Robustness Check
======================
Re-runs key specifications under Logit (instead of LPM) for robustness.

Motivation: Linear Probability Models (LPM) are easy to interpret but can
produce predicted probabilities outside [0,1] and assume constant marginal
effects. Logit addresses both issues. A referee may request this as a
standard robustness check.

For each specification we report:
  - LPM coefficient (for comparison)
  - Logit coefficient (log-odds)
  - Marginal effect at the mean (for economic interpretation)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
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


def run_lpm_and_logit(y_var, x_var, data, dataset, spec_label):
    """
    Run both LPM (OLS with HC1) and Logit on the same data.
    Returns a dict with LPM coef, Logit coef, and marginal effect at mean.
    """
    subset = data[[y_var, x_var]].dropna().copy()
    if len(subset) < 100:
        log(f"  SKIP {spec_label}: too few observations ({len(subset)})")
        return None

    X = sm.add_constant(subset[[x_var]])
    y = subset[y_var]

    # Check outcome has variation
    if y.nunique() < 2:
        log(f"  SKIP {spec_label}: no variation in outcome (all {y.iloc[0]})")
        return None

    # LPM
    lpm_model = sm.OLS(y, X).fit(cov_type='HC1')
    coef_lpm = lpm_model.params[x_var]
    se_lpm = lpm_model.bse[x_var]

    # Logit
    try:
        logit_model = Logit(y, X).fit(disp=0, maxiter=100)
        coef_logit = logit_model.params[x_var]
        se_logit = logit_model.bse[x_var]

        # Marginal effect at the mean
        margeff = logit_model.get_margeff(at='mean')
        margeff_logit = margeff.margeff[0]  # first (and only) variable's marginal effect
    except Exception as e:
        log(f"  WARNING: Logit failed for {spec_label}: {e}")
        log(f"  Falling back to LPM-only result.")
        coef_logit = np.nan
        se_logit = np.nan
        margeff_logit = np.nan

    result = {
        'dataset': dataset,
        'specification': spec_label,
        'outcome': y_var,
        'coef_lpm': coef_lpm,
        'coef_logit': coef_logit,
        'margeff_logit': margeff_logit,
        'se_lpm': se_lpm,
        'se_logit': se_logit,
        'N': len(subset),
    }

    sig_lpm = "***" if lpm_model.pvalues[x_var] < 0.001 else "**" if lpm_model.pvalues[x_var] < 0.01 else "*" if lpm_model.pvalues[x_var] < 0.05 else ""

    log(f"  {spec_label}:")
    log(f"    LPM:    coef={coef_lpm:.6f} (SE={se_lpm:.6f}) {sig_lpm}")
    log(f"    Logit:  coef={coef_logit:.6f} (SE={se_logit:.6f})")
    log(f"    Marginal effect at mean: {margeff_logit:.6f}")
    log(f"    N={len(subset):,}")

    return result


# ============================================================
# PART 1: Diginetica
# ============================================================
log("=" * 60)
log("PART 1: Diginetica — Logit Robustness")
log("=" * 60)

log("Loading Diginetica data...")
queries = pd.read_csv(os.path.join(RAW_DIR_DIGI, "train-queries.csv"), sep=';')
queries = queries[queries['is.test'] == False].copy()
log(f"Train queries (non-test): {len(queries):,}")

clicks = pd.read_csv(os.path.join(RAW_DIR_DIGI, "train-clicks.csv"), sep=';')
purchases_digi = pd.read_csv(os.path.join(RAW_DIR_DIGI, "train-purchases.csv"), sep=';')

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

log(f"Diginetica dataset: {df_digi.shape}")
log(f"Click rate: {df_digi['was_clicked'].mean():.4f}, Purchase rate: {df_digi['was_purchased'].mean():.4f}")

results_table = []

# D-L1: Click ~ early_exposure
log("\n--- D-L1: Click ~ early_exposure (Logit) ---")
r = run_lpm_and_logit('was_clicked', 'early_exposure', df_digi, 'Diginetica',
                       'Click ~ early_exposure')
if r: results_table.append(r)

# D-L2: Purchase ~ early_exposure
log("\n--- D-L2: Purchase ~ early_exposure (Logit) ---")
r = run_lpm_and_logit('was_purchased', 'early_exposure', df_digi, 'Diginetica',
                       'Purchase ~ early_exposure')
if r: results_table.append(r)

# D-L3: Click ~ rank_position
log("\n--- D-L3: Click ~ rank_position (Logit) ---")
r = run_lpm_and_logit('was_clicked', 'rank_position', df_digi, 'Diginetica',
                       'Click ~ rank_position')
if r: results_table.append(r)

# D-L4: Purchase ~ rank_position
log("\n--- D-L4: Purchase ~ rank_position (Logit) ---")
r = run_lpm_and_logit('was_purchased', 'rank_position', df_digi, 'Diginetica',
                       'Purchase ~ rank_position')
if r: results_table.append(r)

del df_digi  # free memory


# ============================================================
# PART 2: REES46
# ============================================================
log("\n" + "=" * 60)
log("PART 2: REES46 — Logit Robustness")
log("=" * 60)

log("Loading REES46 data (2 shards)...")
df0 = pd.read_parquet(os.path.join(RAW_DIR_REES, "shard_0.parquet"))
df1 = pd.read_parquet(os.path.join(RAW_DIR_REES, "shard_1.parquet"))
df_rees = pd.concat([df0, df1], ignore_index=True)
del df0, df1

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

# R-L1: Cart ~ early_exposure
log("\n--- R-L1: Cart ~ early_exposure (Logit) ---")
r = run_lpm_and_logit('was_carted', 'early_exposure', views, 'REES46',
                       'Cart ~ early_exposure')
if r: results_table.append(r)

# R-L2: Purchase ~ early_exposure
log("\n--- R-L2: Purchase ~ early_exposure (Logit) ---")
r = run_lpm_and_logit('was_purchased', 'early_exposure', views, 'REES46',
                       'Purchase ~ early_exposure')
if r: results_table.append(r)


# ============================================================
# Save results
# ============================================================
log("\n" + "=" * 60)
log("SAVING RESULTS")
log("=" * 60)

results_df = pd.DataFrame(results_table)
out_path = os.path.join(RESULTS_DIR, "logit_comparison.csv")
results_df.to_csv(out_path, index=False)
log(f"Saved {len(results_df)} rows to {out_path}")

# Summary comparison
log("\n--- Summary: LPM vs Logit marginal effects ---")
for _, row in results_df.iterrows():
    diff = abs(row['coef_lpm'] - row['margeff_logit']) if not np.isnan(row['margeff_logit']) else np.nan
    log(f"  {row['dataset']} | {row['specification']}:")
    log(f"    LPM coef={row['coef_lpm']:.6f}, Logit ME={row['margeff_logit']:.6f}, diff={diff:.6f}")

# Save log
with open(os.path.join(LOGS_DIR, "16_logit_robustness.log"), 'w') as f:
    f.write('\n'.join(log_lines))

log("\nDone.")
