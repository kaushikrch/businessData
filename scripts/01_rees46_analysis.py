"""
REES46 Multi-Category Store: Attention-Value Wedge Analysis
============================================================
Tests H1-H5 using view/cart/purchase funnel data.

Exposure order: within-session view order (rank by timestamp).
Attention: view event (all rows are at minimum views).
Downstream: cart event, purchase event.
Conditional: purchase | cart, purchase | view.

Key identification caveat:
  Exposure order is endogenous — items viewed earlier may differ systematically
  from items viewed later. Results are ASSOCIATIONS, not causal estimates,
  unless we can find plausible exogenous variation.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
import os
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Paths
RAW_DIR = "/home/user/businessData/data_raw/rees46"
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
# STEP A: Load and inspect data
# ============================================================
log("Loading REES46 data (2 shards)...")
df0 = pd.read_parquet(os.path.join(RAW_DIR, "shard_0.parquet"))
df1 = pd.read_parquet(os.path.join(RAW_DIR, "shard_1.parquet"))
df = pd.concat([df0, df1], ignore_index=True)
del df0, df1

log(f"Total rows: {len(df):,}")
log(f"Columns: {list(df.columns)}")
log(f"Event type distribution:\n{df['event_type'].value_counts()}")

# Parse timestamps
log("Parsing timestamps...")
df['event_time'] = pd.to_datetime(df['event_time'], format='mixed', utc=True)
df = df.sort_values(['user_session', 'event_time']).reset_index(drop=True)

# ============================================================
# STEP A2: Session-level construction
# ============================================================
log("Constructing session-level variables...")

# Filter to sessions with at least 2 events (need ordering)
session_sizes = df.groupby('user_session').size()
valid_sessions = session_sizes[session_sizes >= 2].index
log(f"Sessions with >=2 events: {len(valid_sessions):,} out of {session_sizes.shape[0]:,}")

df = df[df['user_session'].isin(valid_sessions)].copy()
log(f"Rows after filtering: {len(df):,}")

# ============================================================
# STEP B: Construct variables
# ============================================================
log("Constructing analysis variables...")

# Within-session view order (1-based rank of each item's FIRST view)
views = df[df['event_type'] == 'view'].copy()
views['view_rank'] = views.groupby('user_session').cumcount() + 1

# Session length (total views)
session_view_counts = views.groupby('user_session').size().rename('session_length')
views = views.merge(session_view_counts, on='user_session', how='left')

# Normalized position: view_rank / session_length (0 = first, 1 = last)
views['norm_position'] = (views['view_rank'] - 1) / (views['session_length'] - 1)
views.loc[views['session_length'] == 1, 'norm_position'] = 0

# Early exposure: top quartile of session
views['early_exposure'] = (views['norm_position'] <= 0.25).astype(int)
# Also binary: first half vs second half
views['first_half'] = (views['norm_position'] <= 0.5).astype(int)

# Determine which items were carted and purchased in each session
carts = df[df['event_type'] == 'cart'][['user_session', 'product_id']].drop_duplicates()
carts['was_carted'] = 1

purchases = df[df['event_type'] == 'purchase'][['user_session', 'product_id']].drop_duplicates()
purchases['was_purchased'] = 1

# Merge outcomes onto views
views = views.merge(carts, on=['user_session', 'product_id'], how='left')
views['was_carted'] = views['was_carted'].fillna(0).astype(int)

views = views.merge(purchases, on=['user_session', 'product_id'], how='left')
views['was_purchased'] = views['was_purchased'].fillna(0).astype(int)

# Extract top-level category
views['category_top'] = views['category_code'].fillna('unknown').str.split('.').str[0]

# Price log
views['log_price'] = np.log1p(views['price'].fillna(0))

log(f"Views dataset shape: {views.shape}")
log(f"Cart rate: {views['was_carted'].mean():.4f}")
log(f"Purchase rate: {views['was_purchased'].mean():.4f}")
log(f"Cart rate | early_exposure=1: {views.loc[views['early_exposure']==1, 'was_carted'].mean():.4f}")
log(f"Cart rate | early_exposure=0: {views.loc[views['early_exposure']==0, 'was_carted'].mean():.4f}")
log(f"Purchase rate | early_exposure=1: {views.loc[views['early_exposure']==1, 'was_purchased'].mean():.4f}")
log(f"Purchase rate | early_exposure=0: {views.loc[views['early_exposure']==0, 'was_purchased'].mean():.4f}")

# Save processed data
views.to_parquet(os.path.join(PROC_DIR, "rees46_views_processed.parquet"), index=False)
log("Saved processed views data.")

# ============================================================
# STEP C: Main regressions
# ============================================================
log("=" * 60)
log("RUNNING MAIN REGRESSIONS")
log("=" * 60)

results_table = []

def run_lpm(y_var, x_vars, data, label, controls=None):
    """Run a Linear Probability Model and return results dict."""
    subset = data.dropna(subset=[y_var] + x_vars + (controls or []))
    if len(subset) < 100:
        log(f"  SKIP {label}: too few observations ({len(subset)})")
        return None

    X = subset[x_vars].copy()
    if controls:
        X = pd.concat([X, subset[controls]], axis=1)
    X = sm.add_constant(X)
    y = subset[y_var]

    try:
        model = sm.OLS(y, X).fit(cov_type='HC1')  # robust SEs

        result = {
            'specification': label,
            'outcome': y_var,
            'N': len(subset),
            'y_mean': y.mean(),
        }

        for var in x_vars:
            result[f'coef_{var}'] = model.params.get(var, np.nan)
            result[f'se_{var}'] = model.bse.get(var, np.nan)
            result[f'pval_{var}'] = model.pvalues.get(var, np.nan)

        result['R2'] = model.rsquared

        log(f"  {label}: N={len(subset):,}, y_mean={y.mean():.4f}")
        for var in x_vars:
            coef = model.params.get(var, np.nan)
            se = model.bse.get(var, np.nan)
            pval = model.pvalues.get(var, np.nan)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            log(f"    {var}: {coef:.6f} (SE={se:.6f}) p={pval:.4f} {sig}")

        return result
    except Exception as e:
        log(f"  ERROR in {label}: {e}")
        return None


# --- Model 1: Cart ~ early_exposure (H1/H2 test) ---
log("\n--- Model 1: Cart ~ Early Exposure ---")
r = run_lpm('was_carted', ['early_exposure'], views, 'M1a: Cart ~ EarlyExposure')
if r: results_table.append(r)

r = run_lpm('was_carted', ['early_exposure'], views, 'M1b: Cart ~ EarlyExposure + controls',
            controls=['log_price'])
if r: results_table.append(r)

# --- Model 2: Purchase ~ early_exposure (H2 test) ---
log("\n--- Model 2: Purchase ~ Early Exposure ---")
r = run_lpm('was_purchased', ['early_exposure'], views, 'M2a: Purchase ~ EarlyExposure')
if r: results_table.append(r)

r = run_lpm('was_purchased', ['early_exposure'], views, 'M2b: Purchase ~ EarlyExposure + controls',
            controls=['log_price'])
if r: results_table.append(r)

# --- Model 3: Purchase | Cart (H3 test - conditional conversion) ---
log("\n--- Model 3: Purchase | Cart ~ Early Exposure ---")
carted_items = views[views['was_carted'] == 1].copy()
log(f"  Carted items: {len(carted_items):,}")
r = run_lpm('was_purchased', ['early_exposure'], carted_items, 'M3a: Purchase|Cart ~ EarlyExposure')
if r: results_table.append(r)

r = run_lpm('was_purchased', ['early_exposure'], carted_items, 'M3b: Purchase|Cart ~ EarlyExposure + controls',
            controls=['log_price'])
if r: results_table.append(r)

# --- Model 4: Using continuous position ---
log("\n--- Model 4: Continuous position effects ---")
r = run_lpm('was_carted', ['norm_position'], views, 'M4a: Cart ~ NormPosition')
if r: results_table.append(r)

r = run_lpm('was_purchased', ['norm_position'], views, 'M4b: Purchase ~ NormPosition')
if r: results_table.append(r)

r = run_lpm('was_purchased', ['norm_position'], carted_items, 'M4c: Purchase|Cart ~ NormPosition')
if r: results_table.append(r)

# --- Model 5: Using first_half binary ---
log("\n--- Model 5: First half effects ---")
r = run_lpm('was_carted', ['first_half'], views, 'M5a: Cart ~ FirstHalf')
if r: results_table.append(r)

r = run_lpm('was_purchased', ['first_half'], views, 'M5b: Purchase ~ FirstHalf')
if r: results_table.append(r)

r = run_lpm('was_purchased', ['first_half'], carted_items, 'M5c: Purchase|Cart ~ FirstHalf')
if r: results_table.append(r)

# ============================================================
# STEP D: Quantify the wedge
# ============================================================
log("\n" + "=" * 60)
log("QUANTIFYING THE ATTENTION-VALUE WEDGE")
log("=" * 60)

wedge_results = {}

# Cart effect (attention proxy)
cart_early = views.loc[views['early_exposure']==1, 'was_carted'].mean()
cart_late = views.loc[views['early_exposure']==0, 'was_carted'].mean()
cart_diff = cart_early - cart_late
cart_ratio = cart_early / cart_late if cart_late > 0 else np.nan

# Purchase effect (value)
purch_early = views.loc[views['early_exposure']==1, 'was_purchased'].mean()
purch_late = views.loc[views['early_exposure']==0, 'was_purchased'].mean()
purch_diff = purch_early - purch_late
purch_ratio = purch_early / purch_late if purch_late > 0 else np.nan

# Conditional purchase (purchase | cart)
cond_early = carted_items.loc[carted_items['early_exposure']==1, 'was_purchased'].mean()
cond_late = carted_items.loc[carted_items['early_exposure']==0, 'was_purchased'].mean()
cond_diff = cond_early - cond_late

wedge_results = {
    'cart_rate_early': cart_early,
    'cart_rate_late': cart_late,
    'cart_diff_pp': cart_diff * 100,  # percentage points
    'cart_ratio': cart_ratio,
    'purchase_rate_early': purch_early,
    'purchase_rate_late': purch_late,
    'purchase_diff_pp': purch_diff * 100,
    'purchase_ratio': purch_ratio,
    'conditional_purchase_early': cond_early,
    'conditional_purchase_late': cond_late,
    'conditional_purchase_diff_pp': cond_diff * 100,
    'wedge_cart_minus_purchase_ratio': cart_ratio - purch_ratio if not np.isnan(cart_ratio) and not np.isnan(purch_ratio) else np.nan,
}

log(f"Cart rate: early={cart_early:.4f}, late={cart_late:.4f}, diff={cart_diff*100:.2f}pp, ratio={cart_ratio:.3f}")
log(f"Purchase rate: early={purch_early:.4f}, late={purch_late:.4f}, diff={purch_diff*100:.2f}pp, ratio={purch_ratio:.3f}")
log(f"Purchase|Cart: early={cond_early:.4f}, late={cond_late:.4f}, diff={cond_diff*100:.2f}pp")
log(f"WEDGE (cart_ratio - purchase_ratio): {wedge_results['wedge_cart_minus_purchase_ratio']:.3f}")

# ============================================================
# STEP E: Robustness & Heterogeneity
# ============================================================
log("\n" + "=" * 60)
log("ROBUSTNESS AND HETEROGENEITY ANALYSES")
log("=" * 60)

# --- E1: Category subgroup analysis (H4 proxy) ---
log("\n--- E1: Category subgroup analysis ---")
top_cats = views['category_top'].value_counts().head(8).index.tolist()
top_cats = [c for c in top_cats if c != 'unknown']

cat_results = []
for cat in top_cats[:6]:
    cat_data = views[views['category_top'] == cat]
    cat_carted = cat_data[cat_data['was_carted'] == 1]

    n = len(cat_data)
    cart_e = cat_data.loc[cat_data['early_exposure']==1, 'was_carted'].mean()
    cart_l = cat_data.loc[cat_data['early_exposure']==0, 'was_carted'].mean()
    purch_e = cat_data.loc[cat_data['early_exposure']==1, 'was_purchased'].mean()
    purch_l = cat_data.loc[cat_data['early_exposure']==0, 'was_purchased'].mean()

    cond_e = cat_carted.loc[cat_carted['early_exposure']==1, 'was_purchased'].mean() if len(cat_carted[cat_carted['early_exposure']==1]) > 0 else np.nan
    cond_l = cat_carted.loc[cat_carted['early_exposure']==0, 'was_purchased'].mean() if len(cat_carted[cat_carted['early_exposure']==0]) > 0 else np.nan

    cat_results.append({
        'category': cat,
        'N': n,
        'cart_rate_early': cart_e,
        'cart_rate_late': cart_l,
        'cart_diff_pp': (cart_e - cart_l) * 100,
        'purchase_rate_early': purch_e,
        'purchase_rate_late': purch_l,
        'purchase_diff_pp': (purch_e - purch_l) * 100,
        'cond_purchase_early': cond_e,
        'cond_purchase_late': cond_l,
    })

    log(f"  {cat} (N={n:,}): cart_diff={((cart_e-cart_l)*100):.2f}pp, purch_diff={((purch_e-purch_l)*100):.2f}pp")

cat_df = pd.DataFrame(cat_results)
cat_df.to_csv(os.path.join(RESULTS_DIR, "rees46_category_heterogeneity.csv"), index=False)

# --- E2: Price-based uncertainty proxy (H4) ---
log("\n--- E2: Price dispersion as uncertainty proxy ---")
# High-price items may have more uncertainty
median_price = views['price'].median()
views['high_price'] = (views['price'] > median_price).astype(int)

for price_group, label in [(1, 'High price'), (0, 'Low price')]:
    subset = views[views['high_price'] == price_group]
    carted_sub = subset[subset['was_carted'] == 1]

    cart_e = subset.loc[subset['early_exposure']==1, 'was_carted'].mean()
    cart_l = subset.loc[subset['early_exposure']==0, 'was_carted'].mean()
    purch_e = subset.loc[subset['early_exposure']==1, 'was_purchased'].mean()
    purch_l = subset.loc[subset['early_exposure']==0, 'was_purchased'].mean()

    log(f"  {label}: cart_diff={(cart_e-cart_l)*100:.2f}pp, purch_diff={(purch_e-purch_l)*100:.2f}pp")

# --- E3: Session length as intent proxy (H5) ---
log("\n--- E3: Session length as intent proxy ---")
# Short focused sessions = higher intent; long browsing = lower intent
median_session_len = views['session_length'].median()
views['short_session'] = (views['session_length'] <= median_session_len).astype(int)

for intent_group, label in [(1, 'Short session (high intent)'), (0, 'Long session (low intent)')]:
    subset = views[views['short_session'] == intent_group]
    carted_sub = subset[subset['was_carted'] == 1]

    cart_e = subset.loc[subset['early_exposure']==1, 'was_carted'].mean()
    cart_l = subset.loc[subset['early_exposure']==0, 'was_carted'].mean()
    purch_e = subset.loc[subset['early_exposure']==1, 'was_purchased'].mean()
    purch_l = subset.loc[subset['early_exposure']==0, 'was_purchased'].mean()

    log(f"  {label}: cart_diff={(cart_e-cart_l)*100:.2f}pp, purch_diff={(purch_e-purch_l)*100:.2f}pp")

# --- E4: Brand presence as uncertainty proxy ---
log("\n--- E4: Brand presence as uncertainty proxy ---")
views['has_brand'] = (views['brand'].notna() & (views['brand'] != '')).astype(int)

for brand_group, label in [(1, 'Has brand (lower uncertainty)'), (0, 'No brand (higher uncertainty)')]:
    subset = views[views['has_brand'] == brand_group]

    cart_e = subset.loc[subset['early_exposure']==1, 'was_carted'].mean()
    cart_l = subset.loc[subset['early_exposure']==0, 'was_carted'].mean()
    purch_e = subset.loc[subset['early_exposure']==1, 'was_purchased'].mean()
    purch_l = subset.loc[subset['early_exposure']==0, 'was_purchased'].mean()

    log(f"  {label}: cart_diff={(cart_e-cart_l)*100:.2f}pp, purch_diff={(purch_e-purch_l)*100:.2f}pp")

# ============================================================
# Save results
# ============================================================
results_df = pd.DataFrame(results_table)
results_df.to_csv(os.path.join(RESULTS_DIR, "rees46_regression_results.csv"), index=False)

with open(os.path.join(RESULTS_DIR, "rees46_wedge_summary.json"), 'w') as f:
    json.dump(wedge_results, f, indent=2, default=str)

# Save log
with open(os.path.join(LOGS_DIR, "rees46_analysis.log"), 'w') as f:
    f.write('\n'.join(log_lines))

log("\nDone. Results saved to results/ directory.")
