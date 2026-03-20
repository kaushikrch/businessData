"""
REES46: Session Fixed Effects Externality Test (Redesign)
=========================================================
The first externality test (script 10) found positive spillovers due to
session-level selection: high-intent users cart many items. This redesign
uses session fixed effects to remove between-session heterogeneity.

KEY IDEA: Within the SAME session, does the position effect on cart/purchase
change AFTER a cart event has occurred? If carting item j "uses up" some
attention budget, then subsequent items should show a WEAKER position effect
(the consumer is less responsive to browsing order after committing).

Test: Interact position with post-cart indicator, with session FE.
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
# STEP A: Load data
# ============================================================
log("Loading REES46 data...")

shard_files = [
    os.path.join(RAW_DIR, "shard_0.parquet"),
    os.path.join(RAW_DIR, "shard_1.parquet"),
]
dfs = []
for f in shard_files:
    if os.path.exists(f):
        dfs.append(pd.read_parquet(f))
        log(f"  Loaded {f}: {len(dfs[-1]):,} rows")
df_raw = pd.concat(dfs, ignore_index=True)
del dfs

# Separate event types
views = df_raw[df_raw['event_type'] == 'view'].copy()
views['event_time'] = pd.to_datetime(views['event_time'])
carts = df_raw[df_raw['event_type'] == 'cart'][['user_session', 'product_id']].drop_duplicates()
carts['was_carted'] = 1
purchases = df_raw[df_raw['event_type'] == 'purchase'][['user_session', 'product_id']].drop_duplicates()
purchases['was_purchased'] = 1
del df_raw

# Build session-ordered dataset
views = views.sort_values(['user_session', 'event_time'])
views['view_rank'] = views.groupby('user_session').cumcount() + 1
views['session_length'] = views.groupby('user_session')['view_rank'].transform('max')
views = views[views['session_length'] >= 4].copy()

views = views.merge(carts, on=['user_session', 'product_id'], how='left')
views['was_carted'] = views['was_carted'].fillna(0).astype(np.int8)
views = views.merge(purchases, on=['user_session', 'product_id'], how='left')
views['was_purchased'] = views['was_purchased'].fillna(0).astype(np.int8)
del carts, purchases

views['norm_position'] = (views['view_rank'] - 1) / (views['session_length'] - 1)
views['log_session_length'] = np.log(views['session_length'])

log(f"Views in sessions with >= 4 items: {len(views):,}")
log(f"Sessions: {views['user_session'].nunique():,}")

# ============================================================
# STEP B: Construct post-cart indicator
# ============================================================
log("\nConstructing post-cart indicators...")

views = views.sort_values(['user_session', 'view_rank'])
views['cum_carts'] = views.groupby('user_session')['was_carted'].cumsum()
views['cum_carts_before'] = views['cum_carts'] - views['was_carted']
views['post_cart'] = (views['cum_carts_before'] > 0).astype(np.int8)

# Interaction: position effect × post-cart
views['position_x_postcart'] = views['norm_position'] * views['post_cart']

log(f"Items in post-cart state: {views['post_cart'].mean():.4f}")

# ============================================================
# STEP C: Session FE via demeaning
# ============================================================
log("\n" + "=" * 60)
log("SESSION FIXED EFFECTS: POSITION × POST-CART INTERACTION")
log("=" * 60)

# For session FE, restrict to sessions with at least one cart
# (sessions without any cart have post_cart=0 for all items, no variation)
sessions_with_cart = views.groupby('user_session')['was_carted'].max()
sessions_with_cart = sessions_with_cart[sessions_with_cart == 1].index
views_cart_sessions = views[views['user_session'].isin(sessions_with_cart)].copy()
log(f"Sessions with at least one cart: {len(sessions_with_cart):,}")
log(f"Items in cart sessions: {len(views_cart_sessions):,}")

# Also need variation in post_cart within session
# (exclude sessions where cart is on first or last item only)
session_postcart_var = views_cart_sessions.groupby('user_session')['post_cart'].std()
sessions_with_postcart_var = session_postcart_var[session_postcart_var > 0].index
views_var = views_cart_sessions[views_cart_sessions['user_session'].isin(sessions_with_postcart_var)].copy()
log(f"Sessions with post_cart variation: {len(sessions_with_postcart_var):,}")
log(f"Items in these sessions: {len(views_var):,}")

results_table = []

# Demean within session
log("\nDemeaning within session...")
for var in ['was_carted', 'was_purchased', 'norm_position', 'post_cart', 'position_x_postcart']:
    session_means = views_var.groupby('user_session')[var].transform('mean')
    views_var[f'{var}_dm'] = views_var[var] - session_means

# Test 1: Does post_cart reduce subsequent cart/purchase? (session FE)
log("\n--- Test 1: Cart/Purchase ~ post_cart (session FE) ---")
for outcome in ['was_carted', 'was_purchased']:
    X = views_var[['post_cart_dm', 'norm_position_dm']].copy()
    X = sm.add_constant(X)
    model = sm.OLS(views_var[f'{outcome}_dm'], X).fit(cov_type='HC1')

    coef = model.params['post_cart_dm']
    se = model.bse['post_cart_dm']
    pval = model.pvalues['post_cart_dm']
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""

    log(f"  {outcome} ~ post_cart (session FE):")
    log(f"    Coefficient: {coef:.6f} (SE={se:.6f}, p={pval:.4f}) {sig}")
    log(f"    norm_position: {model.params['norm_position_dm']:.6f}")

    results_table.append({
        'test': 'session_fe_postcart',
        'specification': f'{outcome} ~ post_cart (session FE)',
        'coef': coef,
        'se': se,
        'pval': pval,
        'N': len(views_var),
        'n_sessions': len(sessions_with_postcart_var),
    })

# Test 2: Does the position effect weaken after a cart? (interaction)
log("\n--- Test 2: Cart/Purchase ~ norm_position + post_cart + position×post_cart (session FE) ---")
for outcome in ['was_carted', 'was_purchased']:
    X = views_var[['norm_position_dm', 'post_cart_dm', 'position_x_postcart_dm']].copy()
    X = sm.add_constant(X)
    model = sm.OLS(views_var[f'{outcome}_dm'], X).fit(cov_type='HC1')

    coef_interact = model.params['position_x_postcart_dm']
    se_interact = model.bse['position_x_postcart_dm']
    pval_interact = model.pvalues['position_x_postcart_dm']
    sig = "***" if pval_interact < 0.001 else "**" if pval_interact < 0.01 else "*" if pval_interact < 0.05 else ""

    log(f"  {outcome} ~ position × post_cart (session FE):")
    log(f"    norm_position:       {model.params['norm_position_dm']:.6f}")
    log(f"    post_cart:           {model.params['post_cart_dm']:.6f}")
    log(f"    position×post_cart:  {coef_interact:.6f} (SE={se_interact:.6f}, p={pval_interact:.4f}) {sig}")

    results_table.append({
        'test': 'session_fe_interaction',
        'specification': f'{outcome} ~ position × post_cart (session FE)',
        'coef': coef_interact,
        'se': se_interact,
        'pval': pval_interact,
        'N': len(views_var),
        'n_sessions': len(sessions_with_postcart_var),
    })

# ============================================================
# STEP D: Compare pre-cart vs post-cart position gradients
# ============================================================
log("\n" + "=" * 60)
log("POSITION GRADIENT: PRE-CART vs POST-CART")
log("=" * 60)

for state, label in [(0, 'Pre-cart'), (1, 'Post-cart')]:
    sub = views_var[views_var['post_cart'] == state]
    log(f"\n{label} items (N={len(sub):,}):")
    log(f"  Cart rate: {sub['was_carted'].mean():.4f}")
    log(f"  Purchase rate: {sub['was_purchased'].mean():.4f}")

    # Position effect within this state
    if len(sub) > 100:
        X = sm.add_constant(sub['norm_position'])
        for outcome in ['was_carted', 'was_purchased']:
            model = sm.OLS(sub[outcome], X).fit(cov_type='HC1')
            coef = model.params['norm_position']
            pval = model.pvalues['norm_position']
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            log(f"  {outcome} ~ norm_position: {coef:.6f} (p={pval:.4f}) {sig}")

# ============================================================
# STEP E: Full-sample comparison (no session FE, for reference)
# ============================================================
log("\n" + "=" * 60)
log("REFERENCE: NO SESSION FE (full sample)")
log("=" * 60)

# On the full dataset, show that post_cart is positive (selection)
# Then show that session FE reverses or attenuates the sign
later = views[views['view_rank'] > 1].copy()
for outcome in ['was_carted', 'was_purchased']:
    X = later[['post_cart', 'norm_position', 'log_session_length']].copy()
    X = sm.add_constant(X)
    model = sm.OLS(later[outcome], X).fit(cov_type='HC1')
    coef = model.params['post_cart']
    pval = model.pvalues['post_cart']
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    log(f"  {outcome} ~ post_cart (NO session FE): {coef:.6f} (p={pval:.4f}) {sig}")

    results_table.append({
        'test': 'no_fe_reference',
        'specification': f'{outcome} ~ post_cart (no session FE)',
        'coef': coef,
        'se': model.bse['post_cart'],
        'pval': pval,
        'N': len(later),
        'n_sessions': later['user_session'].nunique(),
    })

# ============================================================
# Save results
# ============================================================
results_df = pd.DataFrame(results_table)
results_df.to_csv(os.path.join(RESULTS_DIR, "rees46_session_fe_externality.csv"), index=False)

with open(os.path.join(LOGS_DIR, "rees46_session_fe_externality.log"), 'w') as f:
    f.write('\n'.join(log_lines))

log(f"\nResults saved.")
log("Done.")
