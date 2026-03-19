"""
Coveo SIGIR eCom 2021: Attention-Value Wedge Analysis
======================================================
Tests H1-H5 using TWO exposure mechanisms:
  A) Browsing order within sessions (like REES46)
  B) Search engine rank order (like Diginetica)

Dataset: ~36M browsing events, ~820K search events, ~66K products.
Product actions: detail (view), add (cart), purchase, remove.

Memory strategy: Stream browsing_train.csv in chunks; only keep
aggregated session-level statistics.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import json
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

RAW_DIR = "/home/user/businessData/data_raw/coveo/train"
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
# PART A: Browsing Session Analysis (View → Add → Purchase)
# ============================================================
log("=" * 60)
log("PART A: BROWSING SESSION ANALYSIS")
log("=" * 60)

log("Loading browsing data in chunks (36M rows)...")

# We need: for each session, the order items were viewed (detail),
# which were added to cart, which were purchased.
# Strategy: read in chunks, keep only event_product rows with a product action.

CHUNK_SIZE = 2_000_000
session_items = {}  # session_id -> list of (timestamp, sku, action)
n_rows = 0

for chunk in pd.read_csv(os.path.join(RAW_DIR, "browsing_train.csv"),
                          chunksize=CHUNK_SIZE,
                          usecols=['session_id_hash', 'event_type', 'product_action',
                                   'product_sku_hash', 'server_timestamp_epoch_ms']):
    # Keep only product events with an action
    chunk = chunk[chunk['event_type'] == 'event_product'].copy()
    chunk = chunk[chunk['product_action'].isin(['detail', 'add', 'purchase'])].copy()
    chunk = chunk.dropna(subset=['product_sku_hash'])

    for _, row in chunk.iterrows():
        sid = row['session_id_hash']
        if sid not in session_items:
            session_items[sid] = []
        session_items[sid].append((
            row['server_timestamp_epoch_ms'],
            row['product_sku_hash'],
            row['product_action'],
        ))

    n_rows += len(chunk)
    log(f"  Processed chunk, kept {n_rows:,} product events so far, "
        f"{len(session_items):,} sessions")

    # Memory check: if too many sessions, sample
    if len(session_items) > 500_000:
        log("  Memory limit: sampling 300K sessions...")
        np.random.seed(42)
        keys = list(session_items.keys())
        keep = set(np.random.choice(keys, 300_000, replace=False))
        session_items = {k: v for k, v in session_items.items() if k in keep}

log(f"Total product events: {n_rows:,}")
log(f"Total sessions: {len(session_items):,}")

# ============================================================
# A2: Build item-level dataset from sessions
# ============================================================
log("Building item-level dataset from browsing sessions...")

rows = []
for sid, events in session_items.items():
    # Sort by timestamp
    events.sort(key=lambda x: x[0])

    # Get unique items in view order (first view of each)
    seen_items = {}
    cart_items = set()
    purchase_items = set()

    for ts, sku, action in events:
        if action == 'detail' and sku not in seen_items:
            seen_items[sku] = len(seen_items) + 1  # view position
        elif action == 'add':
            cart_items.add(sku)
        elif action == 'purchase':
            purchase_items.add(sku)

    if len(seen_items) < 2:
        continue  # Need at least 2 viewed items

    n_items = len(seen_items)
    for sku, pos in seen_items.items():
        rows.append((
            sid,
            sku,
            pos,
            n_items,
            1 if sku in cart_items else 0,
            1 if sku in purchase_items else 0,
        ))

# Free memory
del session_items
log(f"Item-level rows: {len(rows):,}")

df = pd.DataFrame(rows, columns=[
    'session_id', 'sku', 'view_position', 'session_length',
    'was_carted', 'was_purchased'
])
del rows

# Normalized position
df['norm_position'] = (df['view_position'] - 1) / (df['session_length'] - 1)
df.loc[df['session_length'] == 1, 'norm_position'] = 0
df['early_exposure'] = (df['norm_position'] <= 0.25).astype(int)
df['top3'] = (df['view_position'] <= 3).astype(int)

log(f"Dataset shape: {df.shape}")
log(f"Cart rate: {df['was_carted'].mean():.4f}")
log(f"Purchase rate: {df['was_purchased'].mean():.4f}")
log(f"Early exposure fraction: {df['early_exposure'].mean():.3f}")

# ============================================================
# A3: Merge price info
# ============================================================
log("Merging product metadata...")
products = pd.read_csv(os.path.join(RAW_DIR, "sku_to_content.csv"),
                        usecols=['product_sku_hash', 'price_bucket', 'category_hash'])
products.columns = ['sku', 'price_bucket', 'category_hash']
products['price_bucket'] = pd.to_numeric(products['price_bucket'], errors='coerce')
df = df.merge(products, on='sku', how='left')

log(f"Price coverage: {df['price_bucket'].notna().mean():.3f}")
log(f"Category coverage: {df['category_hash'].notna().mean():.3f}")

# ============================================================
# A4: Regressions
# ============================================================
log("\n" + "=" * 60)
log("PART A REGRESSIONS: Browsing Order")
log("=" * 60)

results_table = []

def run_lpm(y_var, x_vars, data, label, controls=None):
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

# H1: Cart ~ View Position
log("\n--- H1: Cart (Attention) ~ View Position ---")
for spec, xvar, lbl in [
    ('early_exposure', ['early_exposure'], 'C-A1a: Cart ~ EarlyExposure'),
    ('top3', ['top3'], 'C-A1b: Cart ~ Top3'),
    ('norm_position', ['norm_position'], 'C-A1c: Cart ~ NormPosition'),
]:
    r = run_lpm('was_carted', xvar, df, lbl)
    if r: results_table.append(r)

# With price control
r = run_lpm('was_carted', ['early_exposure'], df, 'C-A1d: Cart ~ EarlyExposure + price',
            controls=['price_bucket'])
if r: results_table.append(r)

# H2: Purchase ~ View Position
log("\n--- H2: Purchase ~ View Position ---")
for spec, xvar, lbl in [
    ('early_exposure', ['early_exposure'], 'C-A2a: Purchase ~ EarlyExposure'),
    ('norm_position', ['norm_position'], 'C-A2c: Purchase ~ NormPosition'),
]:
    r = run_lpm('was_purchased', xvar, df, lbl)
    if r: results_table.append(r)

r = run_lpm('was_purchased', ['early_exposure'], df, 'C-A2d: Purchase ~ EarlyExposure + price',
            controls=['price_bucket'])
if r: results_table.append(r)

# H3: Purchase | Cart
log("\n--- H3: Purchase | Cart ~ View Position ---")
carted = df[df['was_carted'] == 1].copy()
log(f"  Carted items: {len(carted):,}")

r = run_lpm('was_purchased', ['early_exposure'], carted, 'C-A3a: Purchase|Cart ~ EarlyExposure')
if r: results_table.append(r)

r = run_lpm('was_purchased', ['norm_position'], carted, 'C-A3b: Purchase|Cart ~ NormPosition')
if r: results_table.append(r)

r = run_lpm('was_purchased', ['early_exposure'], carted, 'C-A3c: Purchase|Cart ~ EarlyExposure + price',
            controls=['price_bucket'])
if r: results_table.append(r)

# ============================================================
# A5: Quantify the wedge
# ============================================================
log("\n" + "=" * 60)
log("QUANTIFYING THE ATTENTION-VALUE WEDGE (COVEO BROWSING)")
log("=" * 60)

cart_early = df.loc[df['early_exposure']==1, 'was_carted'].mean()
cart_late = df.loc[df['early_exposure']==0, 'was_carted'].mean()
cart_ratio = cart_early / cart_late if cart_late > 0 else np.nan

purch_early = df.loc[df['early_exposure']==1, 'was_purchased'].mean()
purch_late = df.loc[df['early_exposure']==0, 'was_purchased'].mean()
purch_ratio = purch_early / purch_late if purch_late > 0 else np.nan

cond_early = carted.loc[carted['early_exposure']==1, 'was_purchased'].mean() if len(carted[carted['early_exposure']==1]) > 0 else np.nan
cond_late = carted.loc[carted['early_exposure']==0, 'was_purchased'].mean() if len(carted[carted['early_exposure']==0]) > 0 else np.nan
cond_ratio = cond_early / cond_late if cond_late and cond_late > 0 else np.nan

wedge_a = {
    'source': 'coveo_browsing',
    'cart_rate_early': cart_early, 'cart_rate_late': cart_late,
    'cart_diff_pp': (cart_early - cart_late) * 100, 'cart_ratio': cart_ratio,
    'purchase_rate_early': purch_early, 'purchase_rate_late': purch_late,
    'purchase_diff_pp': (purch_early - purch_late) * 100, 'purchase_ratio': purch_ratio,
    'cond_purchase_early': cond_early, 'cond_purchase_late': cond_late,
    'cond_purchase_diff_pp': (cond_early - cond_late) * 100 if cond_early and cond_late else np.nan,
    'cond_purchase_ratio': cond_ratio,
}

log(f"Cart rate: early={cart_early:.4f}, late={cart_late:.4f}, "
    f"diff={(cart_early-cart_late)*100:.2f}pp, ratio={cart_ratio:.3f}")
log(f"Purchase rate: early={purch_early:.4f}, late={purch_late:.4f}, "
    f"diff={(purch_early-purch_late)*100:.2f}pp, ratio={purch_ratio:.3f}")
if cond_early and cond_late:
    log(f"Purchase|Cart: early={cond_early:.4f}, late={cond_late:.4f}, "
        f"diff={(cond_early-cond_late)*100:.2f}pp, ratio={cond_ratio:.3f}")

# ============================================================
# A6: Heterogeneity
# ============================================================
log("\n--- Heterogeneity: Price ---")
price_valid = df.dropna(subset=['price_bucket'])
if len(price_valid) > 0:
    median_p = price_valid['price_bucket'].median()
    for cond, label in [
        (price_valid['price_bucket'] > median_p, 'High price'),
        (price_valid['price_bucket'] <= median_p, 'Low price'),
    ]:
        sub = price_valid[cond]
        if len(sub) < 100: continue
        ce = sub.loc[sub['early_exposure']==1, 'was_carted'].mean()
        cl = sub.loc[sub['early_exposure']==0, 'was_carted'].mean()
        pe = sub.loc[sub['early_exposure']==1, 'was_purchased'].mean()
        pl = sub.loc[sub['early_exposure']==0, 'was_purchased'].mean()
        log(f"  {label} (N={len(sub):,}):")
        log(f"    Cart diff: {(ce-cl)*100:.2f}pp")
        log(f"    Purchase diff: {(pe-pl)*100:.2f}pp")

log("\n--- Heterogeneity: Session length ---")
median_len = df['session_length'].median()
for cond, label in [
    (df['session_length'] <= median_len, 'Short session (high intent)'),
    (df['session_length'] > median_len, 'Long session (low intent)'),
]:
    sub = df[cond]
    ce = sub.loc[sub['early_exposure']==1, 'was_carted'].mean()
    cl = sub.loc[sub['early_exposure']==0, 'was_carted'].mean()
    pe = sub.loc[sub['early_exposure']==1, 'was_purchased'].mean()
    pl = sub.loc[sub['early_exposure']==0, 'was_purchased'].mean()
    log(f"  {label} (N={len(sub):,}):")
    log(f"    Cart diff: {(ce-cl)*100:.2f}pp")
    log(f"    Purchase diff: {(pe-pl)*100:.2f}pp")

# Position gradient (top 15)
log("\n--- Position gradient (view position 1-15) ---")
pos_stats = df[df['view_position'] <= 15].groupby('view_position').agg(
    cart_rate=('was_carted', 'mean'),
    purchase_rate=('was_purchased', 'mean'),
    n=('was_carted', 'count'),
).reset_index()
pos_stats.to_csv(os.path.join(RESULTS_DIR, "coveo_browsing_position_gradient.csv"), index=False)
for _, row in pos_stats.iterrows():
    log(f"  Pos {int(row['view_position']):2d}: cart={row['cart_rate']:.4f}, "
        f"purchase={row['purchase_rate']:.5f}, N={int(row['n']):,}")

# Free browsing data
del df, carted

# ============================================================
# PART B: Search Rank Analysis (Like Diginetica)
# ============================================================
log("\n" + "=" * 60)
log("PART B: SEARCH RANK ANALYSIS")
log("=" * 60)

log("Loading search data...")
# search_train has: session_id_hash, query_vector, clicked_skus_hash, product_skus_hash, timestamp
# product_skus_hash = ranked list of results shown
# clicked_skus_hash = items clicked from results

search = pd.read_csv(os.path.join(RAW_DIR, "search_train.csv"),
                      usecols=['session_id_hash', 'clicked_skus_hash',
                               'product_skus_hash', 'server_timestamp_epoch_ms'])
log(f"Search events: {len(search):,}")

# Keep only rows that have ranked results
search = search.dropna(subset=['product_skus_hash']).copy()
log(f"Search events with results: {len(search):,}")

# Sample for memory
np.random.seed(42)
if len(search) > 100_000:
    search = search.sample(100_000, random_state=42)
    log(f"Sampled to {len(search):,}")

# Parse ranked item lists
import ast

rows = []
for idx, row in search.iterrows():
    try:
        items = ast.literal_eval(row['product_skus_hash'])
    except:
        continue
    if not isinstance(items, list) or len(items) < 2:
        continue

    # Parse clicked items
    clicked = set()
    if pd.notna(row['clicked_skus_hash']):
        try:
            c = ast.literal_eval(row['clicked_skus_hash'])
            if isinstance(c, list):
                clicked = set(c)
            elif isinstance(c, str):
                clicked = {c}
        except:
            pass

    sid = row['session_id_hash']
    for rank, sku in enumerate(items[:30], 1):  # Cap at 30
        rows.append((
            sid,
            sku,
            rank,
            min(len(items), 30),
            1 if sku in clicked else 0,
        ))

log(f"Search item-position rows: {len(rows):,}")
sdf = pd.DataFrame(rows, columns=['session_id', 'sku', 'rank_position', 'list_length', 'was_clicked'])
del rows

# Merge with product info for price
sdf = sdf.merge(products, on='sku', how='left')

# Normalized rank
sdf['norm_rank'] = (sdf['rank_position'] - 1) / (sdf['list_length'] - 1)
sdf.loc[sdf['list_length'] == 1, 'norm_rank'] = 0
sdf['early_exposure'] = (sdf['norm_rank'] <= 0.25).astype(int)
sdf['top5'] = (sdf['rank_position'] <= 5).astype(int)

log(f"Search dataset shape: {sdf.shape}")
log(f"Click rate: {sdf['was_clicked'].mean():.4f}")
log(f"Early exposure fraction: {sdf['early_exposure'].mean():.3f}")

# ============================================================
# B2: Regressions
# ============================================================
log("\n--- Search Rank: Click ~ Rank Position ---")
r = run_lpm('was_clicked', ['early_exposure'], sdf, 'C-B1a: Click ~ EarlyExposure (Search)')
if r: results_table.append(r)

r = run_lpm('was_clicked', ['top5'], sdf, 'C-B1b: Click ~ Top5 (Search)')
if r: results_table.append(r)

r = run_lpm('was_clicked', ['norm_rank'], sdf, 'C-B1c: Click ~ NormRank (Search)')
if r: results_table.append(r)

sdf['price_bucket'] = pd.to_numeric(sdf['price_bucket'], errors='coerce')
r = run_lpm('was_clicked', ['early_exposure'], sdf, 'C-B1d: Click ~ EarlyExposure + price (Search)',
            controls=['price_bucket'])
if r: results_table.append(r)

# Search position gradient
log("\n--- Search position gradient (rank 1-20) ---")
spos = sdf[sdf['rank_position'] <= 20].groupby('rank_position').agg(
    click_rate=('was_clicked', 'mean'),
    n=('was_clicked', 'count'),
).reset_index()
spos.to_csv(os.path.join(RESULTS_DIR, "coveo_search_position_gradient.csv"), index=False)
for _, row in spos.iterrows():
    log(f"  Rank {int(row['rank_position']):2d}: click={row['click_rate']:.4f}, N={int(row['n']):,}")

# Wedge for search
click_early = sdf.loc[sdf['early_exposure']==1, 'was_clicked'].mean()
click_late = sdf.loc[sdf['early_exposure']==0, 'was_clicked'].mean()
click_ratio = click_early / click_late if click_late > 0 else np.nan

wedge_b = {
    'source': 'coveo_search',
    'click_rate_early': click_early,
    'click_rate_late': click_late,
    'click_diff_pp': (click_early - click_late) * 100,
    'click_ratio': click_ratio,
}

log(f"\nSearch wedge: click early={click_early:.4f}, late={click_late:.4f}, "
    f"diff={(click_early-click_late)*100:.2f}pp, ratio={click_ratio:.3f}")

# ============================================================
# Save results
# ============================================================
results_df = pd.DataFrame(results_table)
results_df.to_csv(os.path.join(RESULTS_DIR, "coveo_regression_results.csv"), index=False)

with open(os.path.join(RESULTS_DIR, "coveo_wedge_summary.json"), 'w') as f:
    json.dump({'browsing': wedge_a, 'search': wedge_b}, f, indent=2, default=str)

with open(os.path.join(LOGS_DIR, "coveo_analysis.log"), 'w') as f:
    f.write('\n'.join(log_lines))

log("\nDone. Coveo results saved to results/ directory.")
