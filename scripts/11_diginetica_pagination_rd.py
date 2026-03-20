"""
Diginetica: Pagination Regression Discontinuity
================================================
Tests whether click rates exhibit sharp discontinuities at potential
page boundaries (rank 10, 20) while purchase rates remain smooth.

IDENTIFICATION: Products on either side of a page boundary have nearly
identical algorithmic scores z_j (the ranking algorithm places them
adjacent). But the page break creates a sharp drop in positional
salience b(r) — the user must scroll or click "next page" to see
items beyond the boundary. If clicks drop sharply at the boundary
but purchases don't, this isolates positional salience from quality.

This is the strongest identification strategy available without
randomized ranking experiments.
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

log(f"Train queries: {len(queries):,}")

# Use 20% sample for adequate power at each rank position
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

log(f"Dataset: {len(df):,} item-position pairs")
log(f"Click rate: {df['was_clicked'].mean():.4f}")
log(f"Purchase rate: {df['was_purchased'].mean():.6f}")

# ============================================================
# STEP B: Position gradient (rank 1-30) — detect discontinuities
# ============================================================
log("\n" + "=" * 60)
log("POSITION GRADIENT: DETECTING PAGE BOUNDARIES")
log("=" * 60)

# Full position gradient
pos_stats = df.groupby('rank_position').agg(
    click_rate=('was_clicked', 'mean'),
    purchase_rate=('was_purchased', 'mean'),
    n=('was_clicked', 'count'),
).reset_index()

log("\nRank-by-rank click and purchase rates:")
log(f"{'Rank':>4} {'Click%':>8} {'Purch%':>8} {'N':>10} {'Click_Δ':>8}")
prev_click = None
for _, row in pos_stats.iterrows():
    delta = f"{(row['click_rate'] - prev_click)*100:+.3f}" if prev_click is not None else "   ---"
    log(f"{int(row['rank_position']):4d} {row['click_rate']*100:8.3f} "
        f"{row['purchase_rate']*100:8.4f} {int(row['n']):10,} {delta:>8}")
    prev_click = row['click_rate']

# Compute rank-to-rank changes to find discontinuities
pos_stats['click_delta'] = pos_stats['click_rate'].diff()
pos_stats['purchase_delta'] = pos_stats['purchase_rate'].diff()
pos_stats['click_pct_change'] = pos_stats['click_rate'].pct_change()

log("\n--- Largest click rate drops (potential page boundaries) ---")
drops = pos_stats.dropna(subset=['click_delta']).nsmallest(5, 'click_delta')
for _, row in drops.iterrows():
    log(f"  Rank {int(row['rank_position'])}: click Δ = {row['click_delta']*100:.3f}pp "
        f"({row['click_pct_change']*100:.1f}%), purchase Δ = {row['purchase_delta']*100:.4f}pp")

# ============================================================
# STEP C: RD at candidate boundaries
# ============================================================
log("\n" + "=" * 60)
log("REGRESSION DISCONTINUITY ANALYSIS")
log("=" * 60)

results_table = []

# Test RD at multiple candidate boundaries
for boundary in [5, 10, 15, 20, 25]:
    log(f"\n--- RD at rank {boundary} (bandwidth ±3) ---")

    # Bandwidth: 3 positions on each side
    bw = 3
    rd_sample = df[
        (df['rank_position'] >= boundary - bw) &
        (df['rank_position'] <= boundary + bw)
    ].copy()

    # Only use queries with items on both sides of boundary
    rd_sample = rd_sample[rd_sample['list_length'] > boundary].copy()

    if len(rd_sample) < 1000:
        log(f"  Insufficient observations ({len(rd_sample)}), skipping")
        continue

    # Treatment: below boundary (rank > boundary)
    rd_sample['below_boundary'] = (rd_sample['rank_position'] > boundary).astype(int)
    # Running variable: distance from boundary
    rd_sample['distance'] = rd_sample['rank_position'] - boundary

    n_above = (rd_sample['below_boundary'] == 0).sum()
    n_below = (rd_sample['below_boundary'] == 1).sum()
    log(f"  N above boundary (ranks {boundary-bw}-{boundary}): {n_above:,}")
    log(f"  N below boundary (ranks {boundary+1}-{boundary+bw}): {n_below:,}")

    for outcome in ['was_clicked', 'was_purchased']:
        # Simple RD: outcome ~ below_boundary + distance + below_boundary*distance
        rd_sample['interaction'] = rd_sample['below_boundary'] * rd_sample['distance']
        X = rd_sample[['below_boundary', 'distance', 'interaction']].copy()
        X = sm.add_constant(X)
        y = rd_sample[outcome]

        model = sm.OLS(y, X).fit(cov_type='HC1')

        coef = model.params['below_boundary']
        se = model.bse['below_boundary']
        pval = model.pvalues['below_boundary']
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""

        # Mean rates on each side
        rate_above = rd_sample.loc[rd_sample['below_boundary'] == 0, outcome].mean()
        rate_below = rd_sample.loc[rd_sample['below_boundary'] == 1, outcome].mean()

        log(f"  {outcome}:")
        log(f"    Above boundary: {rate_above*100:.4f}%")
        log(f"    Below boundary: {rate_below*100:.4f}%")
        log(f"    RD estimate: {coef:.6f} (SE={se:.6f}, p={pval:.4f}) {sig}")

        results_table.append({
            'boundary': boundary,
            'bandwidth': bw,
            'outcome': outcome,
            'rate_above': rate_above,
            'rate_below': rate_below,
            'rd_estimate': coef,
            'se': se,
            'pval': pval,
            'N': len(rd_sample),
        })

# ============================================================
# STEP D: RD with item FE (strongest specification)
# ============================================================
log("\n" + "=" * 60)
log("RD WITH ITEM FIXED EFFECTS")
log("=" * 60)

# Find the strongest boundary from Step C
if results_table:
    click_results = [r for r in results_table if r['outcome'] == 'was_clicked']
    if click_results:
        best_boundary = min(click_results, key=lambda x: x['pval'])['boundary']
        log(f"Strongest click discontinuity at rank {best_boundary}")

        bw = 4  # Slightly wider for FE power
        rd_fe = df[
            (df['rank_position'] >= best_boundary - bw) &
            (df['rank_position'] <= best_boundary + bw) &
            (df['list_length'] > best_boundary)
        ].copy()

        rd_fe['below_boundary'] = (rd_fe['rank_position'] > best_boundary).astype(int)
        rd_fe['distance'] = rd_fe['rank_position'] - best_boundary

        # Item FE via demeaning
        item_counts = rd_fe.groupby('item_id').size()
        frequent = item_counts[item_counts >= 3].index
        rd_fe = rd_fe[rd_fe['item_id'].isin(frequent)].copy()
        log(f"Items with >=3 appearances near boundary: {len(frequent):,}")
        log(f"Observations: {len(rd_fe):,}")

        if len(rd_fe) > 1000:
            for var in ['was_clicked', 'was_purchased', 'below_boundary', 'distance']:
                item_means = rd_fe.groupby('item_id')[var].transform('mean')
                rd_fe[f'{var}_dm'] = rd_fe[var] - item_means

            for outcome in ['was_clicked', 'was_purchased']:
                X = rd_fe[['below_boundary_dm', 'distance_dm']].copy()
                X = sm.add_constant(X)
                model = sm.OLS(rd_fe[f'{outcome}_dm'], X).fit(cov_type='HC1')

                coef = model.params['below_boundary_dm']
                se = model.bse['below_boundary_dm']
                pval = model.pvalues['below_boundary_dm']
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""

                log(f"\n  {outcome} ~ below_boundary (RD + item FE):")
                log(f"    RD estimate: {coef:.6f} (SE={se:.6f}, p={pval:.4f}) {sig}")

                results_table.append({
                    'boundary': best_boundary,
                    'bandwidth': bw,
                    'outcome': f'{outcome}_item_fe',
                    'rate_above': None,
                    'rate_below': None,
                    'rd_estimate': coef,
                    'se': se,
                    'pval': pval,
                    'N': len(rd_fe),
                })

# ============================================================
# STEP E: Bandwidth sensitivity
# ============================================================
log("\n" + "=" * 60)
log("BANDWIDTH SENSITIVITY (at strongest boundary)")
log("=" * 60)

if results_table:
    for bw in [2, 3, 4, 5]:
        rd_bw = df[
            (df['rank_position'] >= best_boundary - bw) &
            (df['rank_position'] <= best_boundary + bw) &
            (df['list_length'] > best_boundary)
        ].copy()

        rd_bw['below_boundary'] = (rd_bw['rank_position'] > best_boundary).astype(int)
        rd_bw['distance'] = rd_bw['rank_position'] - best_boundary
        rd_bw['interaction'] = rd_bw['below_boundary'] * rd_bw['distance']

        X = rd_bw[['below_boundary', 'distance', 'interaction']].copy()
        X = sm.add_constant(X)

        for outcome in ['was_clicked', 'was_purchased']:
            model = sm.OLS(rd_bw[outcome], X).fit(cov_type='HC1')
            coef = model.params['below_boundary']
            se = model.bse['below_boundary']
            pval = model.pvalues['below_boundary']
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            log(f"  bw={bw}, {outcome}: RD={coef:.6f} (SE={se:.6f}, p={pval:.4f}) {sig}")

# ============================================================
# Save results
# ============================================================
pos_stats.to_csv(os.path.join(RESULTS_DIR, "diginetica_full_position_gradient.csv"), index=False)

results_df = pd.DataFrame(results_table)
results_df.to_csv(os.path.join(RESULTS_DIR, "diginetica_pagination_rd_results.csv"), index=False)

with open(os.path.join(LOGS_DIR, "diginetica_pagination_rd.log"), 'w') as f:
    f.write('\n'.join(log_lines))

log(f"\nResults saved.")
log("Done.")
