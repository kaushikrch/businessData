#!/usr/bin/env python3
"""
15_falsification_tests.py
=========================
Three falsification / placebo tests to strengthen causal claims
for the attention-value wedge.

Test 1: Permutation test on Diginetica (rank shuffling within queries)
Test 2: Reverse-causation check on REES46 (position of purchased items)
Test 3: Coveo formal falsification (high-ρ platform → no wedge predicted)

Outputs
-------
results/falsification_results.csv   – summary of all three tests
results/fig13_permutation_test.png  – histogram of placebo coefficients
"""

import warnings
warnings.filterwarnings('ignore')

import logging
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data_raw')
RESULTS = os.path.join(BASE, 'results')
os.makedirs(RESULTS, exist_ok=True)

all_results = []

# ─────────────────────────────────────────────────────────────
# TEST 1 – Permutation test on Diginetica
# ─────────────────────────────────────────────────────────────
log.info('=== TEST 1: Permutation test on Diginetica ===')

try:
    # Load queries
    queries = pd.read_csv(os.path.join(DATA, 'diginetica', 'train-queries.csv'), sep=';')
    queries = queries[queries['is.test'] == False].copy()
    log.info(f'Loaded {len(queries):,} training queries')

    # Sample 10 %
    np.random.seed(42)
    queries_sample = queries.sample(frac=0.10, random_state=42)
    log.info(f'Sampled {len(queries_sample):,} queries (10%%)')

    # Parse items column → item-position rows (rank 1-30)
    rows = []
    for _, row in queries_sample.iterrows():
        items = str(row['items']).split(',')
        qid = row['queryId']
        sid = row['sessionId']
        for rank, item in enumerate(items[:30], start=1):
            try:
                rows.append({'queryId': int(qid), 'sessionId': int(sid),
                             'itemId': int(item.strip()), 'rank_position': rank})
            except (ValueError, TypeError):
                continue
    df_items = pd.DataFrame(rows)
    log.info(f'Created {len(df_items):,} item-position rows')

    # Merge clicks
    clicks = pd.read_csv(os.path.join(DATA, 'diginetica', 'train-clicks.csv'), sep=';')
    clicks['was_clicked'] = 1
    df_items = df_items.merge(clicks[['queryId', 'itemId', 'was_clicked']].drop_duplicates(),
                              on=['queryId', 'itemId'], how='left')
    df_items['was_clicked'] = df_items['was_clicked'].fillna(0).astype(int)

    # Merge purchases
    purchases = pd.read_csv(os.path.join(DATA, 'diginetica', 'train-purchases.csv'), sep=';')
    purchases['was_purchased'] = 1
    df_items = df_items.merge(purchases[['sessionId', 'itemId', 'was_purchased']].drop_duplicates(),
                              on=['sessionId', 'itemId'], how='left')
    df_items['was_purchased'] = df_items['was_purchased'].fillna(0).astype(int)

    # Construct early_exposure from normalized rank
    max_rank = df_items.groupby('queryId')['rank_position'].transform('max')
    df_items['norm_rank'] = (df_items['rank_position'] - 1) / (max_rank - 1).replace(0, 1)
    df_items['early_exposure'] = (df_items['norm_rank'] <= 0.25).astype(int)

    log.info(f'Click rate: {df_items["was_clicked"].mean():.4f}')
    log.info(f'Early exposure share: {df_items["early_exposure"].mean():.3f}')

    # --- Actual OLS ---
    y = df_items['was_clicked'].values
    X = sm.add_constant(df_items['early_exposure'].values)
    model = sm.OLS(y, X).fit(cov_type='HC1')
    actual_coef = model.params[1]
    actual_pval = model.pvalues[1]
    log.info(f'ACTUAL coefficient: {actual_coef:.6f}  (p={actual_pval:.2e})')

    # --- 200 permutations ---
    N_PERMS = 200
    placebo_coefs = []
    rng = np.random.RandomState(42)

    # Pre-group for speed
    query_groups = df_items.groupby('queryId')['rank_position'].transform('count')
    group_indices = df_items.groupby('queryId').indices

    for i in range(N_PERMS):
        shuffled_ranks = df_items['rank_position'].values.copy()
        for _, idx in group_indices.items():
            rng.shuffle(shuffled_ranks[idx])
        max_r = np.ones(len(shuffled_ranks))
        for _, idx in group_indices.items():
            mr = shuffled_ranks[idx].max()
            max_r[idx] = mr
        norm_r = (shuffled_ranks - 1) / np.where(max_r - 1 == 0, 1, max_r - 1)
        ee = (norm_r <= 0.25).astype(int)
        X_perm = sm.add_constant(ee)
        m = sm.OLS(y, X_perm).fit(cov_type='HC1')
        placebo_coefs.append(m.params[1])
        if (i + 1) % 50 == 0:
            log.info(f'  Permutation {i+1}/{N_PERMS}')

    placebo_coefs = np.array(placebo_coefs)
    perm_pvalue = (np.abs(placebo_coefs) >= np.abs(actual_coef)).mean()
    log.info(f'Permutation p-value: {perm_pvalue:.4f}  '
             f'(|placebo| >= |actual| in {perm_pvalue*100:.1f}% of cases)')

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(placebo_coefs, bins=40, color='#7fafd4', edgecolor='white', alpha=0.85, label='Placebo coefficients')
    ax.axvline(actual_coef, color='#c0392b', linewidth=2.2, linestyle='--',
               label=f'Actual coeff = {actual_coef:.5f}')
    ax.set_xlabel('Coefficient on early_exposure → Click', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Permutation Test: Diginetica Click ~ Early Exposure\n'
                 f'(200 permutations, p = {perm_pvalue:.3f})', fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'fig13_permutation_test.png'), dpi=200)
    plt.close(fig)
    log.info('Saved fig13_permutation_test.png')

    all_results.append({
        'test': 'Permutation Test (Diginetica)',
        'description': 'Shuffle rank within query, re-estimate Click ~ early_exposure',
        'actual_coefficient': actual_coef,
        'actual_pvalue': actual_pval,
        'placebo_mean': placebo_coefs.mean(),
        'placebo_std': placebo_coefs.std(),
        'permutation_pvalue': perm_pvalue,
        'n_permutations': N_PERMS,
        'N': len(df_items),
        'conclusion': 'Actual coefficient far outside placebo distribution' if perm_pvalue < 0.05
                      else 'Cannot reject null – actual within placebo range',
    })

except Exception as e:
    log.error(f'Test 1 failed: {e}', exc_info=True)
    all_results.append({'test': 'Permutation Test (Diginetica)',
                        'conclusion': f'FAILED: {e}'})

# ─────────────────────────────────────────────────────────────
# TEST 2 – Reverse-causation check on REES46
# ─────────────────────────────────────────────────────────────
log.info('=== TEST 2: Reverse-causation check on REES46 ===')

try:
    shards = []
    for s in ['shard_0.parquet', 'shard_1.parquet']:
        p = os.path.join(DATA, 'rees46', s)
        if os.path.exists(p):
            shards.append(pd.read_parquet(p))
    rees = pd.concat(shards, ignore_index=True)
    log.info(f'Loaded {len(rees):,} REES46 events')

    # Filter to views
    views = rees[rees['event_type'] == 'view'].copy()
    log.info(f'{len(views):,} view events')

    # Session order
    views = views.sort_values(['user_session', 'event_time']).reset_index(drop=True)
    views['session_order'] = views.groupby('user_session').cumcount() + 1
    session_size = views.groupby('user_session')['session_order'].transform('max')
    views['norm_position'] = (views['session_order'] - 1) / (session_size - 1).replace(0, 1)

    # Identify purchased items from purchase events
    purchases_rees = rees[rees['event_type'] == 'purchase'][['user_session', 'product_id']].drop_duplicates()
    purchases_rees['was_purchased'] = 1
    views = views.merge(purchases_rees, on=['user_session', 'product_id'], how='left')
    views['was_purchased'] = views['was_purchased'].fillna(0).astype(int)

    log.info(f'Purchase rate among views: {views["was_purchased"].mean():.4f}')

    # Mean position comparison
    mean_pos_purchased = views.loc[views['was_purchased'] == 1, 'norm_position'].mean()
    mean_pos_not_purchased = views.loc[views['was_purchased'] == 0, 'norm_position'].mean()
    log.info(f'Mean norm_position — purchased: {mean_pos_purchased:.4f}, '
             f'not purchased: {mean_pos_not_purchased:.4f}')

    # Regression: norm_position ~ was_purchased
    y2 = views['norm_position'].values
    X2 = sm.add_constant(views['was_purchased'].values)
    model2 = sm.OLS(y2, X2).fit(cov_type='HC1')
    coef_purchased = model2.params[1]
    pval_purchased = model2.pvalues[1]
    log.info(f'Coefficient (norm_position ~ was_purchased): {coef_purchased:.6f}  (p={pval_purchased:.2e})')

    # Interpretation
    if coef_purchased < -0.05:
        conclusion2 = ('Purchased items viewed significantly EARLIER → possible reverse causation '
                       'concern (coefficient={:.4f})'.format(coef_purchased))
    elif coef_purchased > 0.05:
        conclusion2 = ('Purchased items viewed LATER → no reverse causation '
                       '(coefficient={:.4f})'.format(coef_purchased))
    else:
        conclusion2 = ('Near-zero coefficient ({:.4f}) → purchased items NOT systematically '
                       'earlier, supports our causal direction'.format(coef_purchased))

    log.info(f'Conclusion: {conclusion2}')

    all_results.append({
        'test': 'Reverse Causation Check (REES46)',
        'description': 'norm_position ~ was_purchased; checks if purchased items cluster early',
        'mean_pos_purchased': mean_pos_purchased,
        'mean_pos_not_purchased': mean_pos_not_purchased,
        'coefficient': coef_purchased,
        'pvalue': pval_purchased,
        'N': len(views),
        'conclusion': conclusion2,
    })

except Exception as e:
    log.error(f'Test 2 failed: {e}', exc_info=True)
    all_results.append({'test': 'Reverse Causation Check (REES46)',
                        'conclusion': f'FAILED: {e}'})

# ─────────────────────────────────────────────────────────────
# TEST 3 – Coveo formal falsification
# ─────────────────────────────────────────────────────────────
log.info('=== TEST 3: Coveo formal falsification ===')

try:
    browsing = pd.read_csv(os.path.join(DATA, 'coveo', 'train', 'browsing_train.csv'))
    log.info(f'Loaded {len(browsing):,} Coveo browsing events')

    # Build session order
    browsing = browsing.sort_values(['session_id_hash', 'server_timestamp_epoch_ms']).reset_index(drop=True)
    browsing['session_order'] = browsing.groupby('session_id_hash').cumcount() + 1
    sess_size = browsing.groupby('session_id_hash')['session_order'].transform('max')
    browsing['norm_position'] = (browsing['session_order'] - 1) / (sess_size - 1).replace(0, 1)
    browsing['early_exposure'] = (browsing['norm_position'] <= 0.25).astype(int)

    # Identify cart / purchase actions
    browsing['was_carted'] = (browsing['product_action'] == 'add').astype(int)
    browsing['was_purchased'] = (browsing['product_action'] == 'purchase').astype(int)

    # Filter to detail views only (like other analyses)
    details = browsing[browsing['product_action'] == 'detail'].copy()
    # Merge cart/purchase indicators per session-product
    cart_flags = browsing[browsing['was_carted'] == 1][['session_id_hash', 'product_sku_hash']].drop_duplicates()
    cart_flags['was_carted'] = 1
    purch_flags = browsing[browsing['was_purchased'] == 1][['session_id_hash', 'product_sku_hash']].drop_duplicates()
    purch_flags['was_purchased'] = 1

    details = details.merge(cart_flags, on=['session_id_hash', 'product_sku_hash'], how='left', suffixes=('_x', ''))
    details['was_carted'] = details['was_carted'].fillna(0).astype(int)
    details = details.merge(purch_flags, on=['session_id_hash', 'product_sku_hash'], how='left', suffixes=('_x', ''))
    details['was_purchased'] = details['was_purchased'].fillna(0).astype(int)

    # Drop any duplicated columns
    for c in list(details.columns):
        if c.endswith('_x'):
            details = details.drop(columns=[c])

    log.info(f'Detail views: {len(details):,}')
    log.info(f'Cart rate: {details["was_carted"].mean():.4f}, Purchase rate: {details["was_purchased"].mean():.4f}')

    # Rates by exposure
    early = details[details['early_exposure'] == 1]
    late = details[details['early_exposure'] == 0]

    cart_rate_early = early['was_carted'].mean()
    cart_rate_late = late['was_carted'].mean()
    purch_rate_early = early['was_purchased'].mean()
    purch_rate_late = late['was_purchased'].mean()

    cart_ratio = cart_rate_early / cart_rate_late if cart_rate_late > 0 else np.nan
    purch_ratio = purch_rate_early / purch_rate_late if purch_rate_late > 0 else np.nan

    log.info(f'Cart rate  — early: {cart_rate_early:.4f}, late: {cart_rate_late:.4f}, ratio: {cart_ratio:.3f}')
    log.info(f'Purch rate — early: {purch_rate_early:.4f}, late: {purch_rate_late:.4f}, ratio: {purch_ratio:.3f}')

    # Formal regression: Purchase ~ early_exposure
    y3 = details['was_purchased'].values
    X3 = sm.add_constant(details['early_exposure'].values)
    model3 = sm.OLS(y3, X3).fit(cov_type='HC1')
    coef3 = model3.params[1]
    pval3 = model3.pvalues[1]
    log.info(f'Purchase ~ early_exposure: coef={coef3:.6f}, p={pval3:.2e}')

    if purch_ratio >= 1.0:
        conclusion3 = (f'Coveo purchase ratio = {purch_ratio:.3f} (>= 1): early items are MORE likely '
                       f'purchased. No wedge present. This is consistent with the model prediction '
                       f'that high-rho platforms show no attention-value wedge.')
    else:
        conclusion3 = (f'Coveo purchase ratio = {purch_ratio:.3f} (< 1): some wedge may exist. '
                       f'Unexpected under high-rho assumption.')

    log.info(f'Conclusion: {conclusion3}')

    all_results.append({
        'test': 'Coveo Formal Falsification',
        'description': 'High-rho platform should show no wedge; purchase ratio >= 1 expected',
        'cart_rate_early': cart_rate_early,
        'cart_rate_late': cart_rate_late,
        'cart_ratio': cart_ratio,
        'purch_rate_early': purch_rate_early,
        'purch_rate_late': purch_rate_late,
        'purchase_ratio': purch_ratio,
        'purchase_coef': coef3,
        'purchase_pval': pval3,
        'N': len(details),
        'conclusion': conclusion3,
    })

except Exception as e:
    log.error(f'Test 3 failed: {e}', exc_info=True)
    all_results.append({'test': 'Coveo Formal Falsification',
                        'conclusion': f'FAILED: {e}'})

# ─────────────────────────────────────────────────────────────
# Save all results
# ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(RESULTS, 'falsification_results.csv'), index=False)
log.info(f'Saved falsification_results.csv ({len(results_df)} rows)')
log.info('Done.')
