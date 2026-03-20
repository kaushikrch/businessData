#!/usr/bin/env python3
"""
17_mht_correction.py
====================
Multiple Hypothesis Testing correction across all primary specifications
for the attention-value wedge paper.

Step 1: Collect p-values from result files and classify tests
Step 2: Apply Bonferroni (primary) and Benjamini-Hochberg FDR (all)

Output
------
results/mht_corrected_pvalues.csv
"""

import warnings
warnings.filterwarnings('ignore')

import logging
import os
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, 'results')

# ─────────────────────────────────────────────────────────────
# Step 1: Collect p-values
# ─────────────────────────────────────────────────────────────
log.info('=== Step 1: Collecting p-values from result files ===')

collected = []  # list of dicts with test_name, dataset, outcome, p_original, test_category


def _safe_read(fname):
    path = os.path.join(RESULTS, fname)
    if os.path.exists(path):
        log.info(f'  Reading {fname}')
        return pd.read_csv(path)
    log.warning(f'  File not found: {fname}')
    return None


# --- REES46 & Diginetica regression results ---
robust = _safe_read('robust_regression_results.csv')
if robust is not None:
    # Combined file – parse both datasets
    for _, row in robust.iterrows():
        spec = str(row.get('specification', ''))
        outcome = str(row.get('outcome', ''))
        # Determine dataset
        ds = 'REES46' if 'rees' in spec.lower() or 'M1' in spec or 'M2' in spec or 'M3' in spec else 'Diginetica'
        # Find p-value columns
        for col in robust.columns:
            if col.startswith('pval_'):
                pv = row[col]
                if pd.notna(pv) and pv != '' and float(pv) < 1.0:
                    varname = col.replace('pval_', '')
                    collected.append({
                        'test_name': f'{spec} [{varname}]',
                        'dataset': ds,
                        'outcome': outcome,
                        'p_original': float(pv),
                    })
else:
    # Read separate files
    for fname, dataset in [('rees46_regression_results.csv', 'REES46'),
                           ('diginetica_regression_results.csv', 'Diginetica')]:
        df = _safe_read(fname)
        if df is None:
            continue
        for _, row in df.iterrows():
            spec = str(row.get('specification', ''))
            outcome = str(row.get('outcome', ''))
            for col in df.columns:
                if col.startswith('pval_'):
                    pv = row[col]
                    if pd.notna(pv) and str(pv).strip() != '':
                        try:
                            pv = float(pv)
                        except (ValueError, TypeError):
                            continue
                        if pv < 1.0:
                            varname = col.replace('pval_', '')
                            collected.append({
                                'test_name': f'{spec} [{varname}]',
                                'dataset': dataset,
                                'outcome': outcome,
                                'p_original': pv,
                            })

# --- Coveo regression results ---
coveo = _safe_read('coveo_regression_results.csv')
if coveo is not None:
    for _, row in coveo.iterrows():
        spec = str(row.get('specification', ''))
        outcome = str(row.get('outcome', ''))
        for col in coveo.columns:
            if col.startswith('pval_'):
                pv = row[col]
                if pd.notna(pv) and str(pv).strip() != '':
                    try:
                        pv = float(pv)
                    except (ValueError, TypeError):
                        continue
                    if pv < 1.0:
                        varname = col.replace('pval_', '')
                        collected.append({
                            'test_name': f'{spec} [{varname}]',
                            'dataset': 'Coveo',
                            'outcome': outcome,
                            'p_original': pv,
                        })

# --- Diginetica item FE results ---
item_fe = _safe_read('diginetica_item_fe_results.csv')
if item_fe is not None:
    for _, row in item_fe.iterrows():
        spec = str(row.get('specification', ''))
        pv = row.get('pval')
        if pd.notna(pv):
            try:
                pv = float(pv)
            except (ValueError, TypeError):
                pv = None
            if pv is not None:
                # Determine outcome from spec
                outcome = ''
                if 'Click' in spec or 'click' in spec:
                    outcome = 'was_clicked'
                elif 'Purchase' in spec or 'purchase' in spec:
                    outcome = 'was_purchased'
                collected.append({
                    'test_name': spec,
                    'dataset': 'Diginetica',
                    'outcome': outcome,
                    'p_original': pv,
                })

# --- REES46 session FE / externality ---
sess_fe = _safe_read('rees46_session_fe_externality.csv')
if sess_fe is not None:
    for _, row in sess_fe.iterrows():
        spec = str(row.get('specification', row.get('test', '')))
        test_type = str(row.get('test', ''))
        pv = row.get('pval')
        if pd.notna(pv):
            try:
                pv = float(pv)
            except (ValueError, TypeError):
                pv = None
            if pv is not None:
                outcome = ''
                if 'cart' in spec.lower():
                    outcome = 'was_carted'
                elif 'purchase' in spec.lower():
                    outcome = 'was_purchased'
                collected.append({
                    'test_name': f'{test_type}: {spec}' if test_type != spec else spec,
                    'dataset': 'REES46',
                    'outcome': outcome,
                    'p_original': pv,
                })

log.info(f'Collected {len(collected)} p-values total')

if len(collected) == 0:
    log.error('No p-values collected. Check that result files exist.')
    import sys
    sys.exit(1)

pvals_df = pd.DataFrame(collected)

# ─────────────────────────────────────────────────────────────
# Classify tests
# ─────────────────────────────────────────────────────────────
log.info('=== Classifying tests ===')


def classify_test(row):
    """
    Primary: core wedge tests (H1-H3) on REES46 and Diginetica
      - Cart/Click ~ early_exposure (the main treatment variable)
      - Purchase ~ early_exposure
    Secondary: item FE, heterogeneity, robustness
    Exploratory: everything else
    """
    name = str(row['test_name']).lower()
    ds = str(row['dataset'])
    outcome = str(row['outcome']).lower()

    # Primary: core specifications on REES46 and Diginetica with early_exposure
    is_core_dataset = ds in ('REES46', 'Diginetica')
    is_early_exposure = 'early_exposure' in name or 'earlyexposure' in name
    is_core_spec = ('m1a' in name or 'm1b' in name or 'm2a' in name or 'm2b' in name
                    or 'm3a' in name or 'm3b' in name
                    or 'd-m1a' in name or 'd-m1b' in name or 'd-m2a' in name or 'd-m2b' in name)

    if is_core_dataset and is_early_exposure and is_core_spec:
        return 'Primary'

    # Secondary: item FE, session FE, controls, top5/norm_rank variants
    if 'item fe' in name or 'item_fe' in name or 'demeaned' in name:
        return 'Secondary'
    if 'session fe' in name or 'session_fe' in name:
        return 'Secondary'
    if is_core_dataset and is_early_exposure:
        return 'Secondary'

    # Coveo tests are secondary (robustness / falsification)
    if ds == 'Coveo':
        return 'Secondary'

    return 'Exploratory'


pvals_df['test_category'] = pvals_df.apply(classify_test, axis=1)

for cat in ['Primary', 'Secondary', 'Exploratory']:
    n = (pvals_df['test_category'] == cat).sum()
    log.info(f'  {cat}: {n} tests')

# ─────────────────────────────────────────────────────────────
# Step 2: Apply corrections
# ─────────────────────────────────────────────────────────────
log.info('=== Step 2: Applying MHT corrections ===')

# --- Bonferroni on primary tests ---
primary_mask = pvals_df['test_category'] == 'Primary'
n_primary = primary_mask.sum()
log.info(f'Bonferroni correction over {n_primary} primary tests')

pvals_df['p_bonferroni'] = np.nan
if n_primary > 0:
    pvals_df.loc[primary_mask, 'p_bonferroni'] = np.minimum(
        pvals_df.loc[primary_mask, 'p_original'] * n_primary, 1.0
    )

# --- Benjamini-Hochberg FDR on ALL tests ---
n_all = len(pvals_df)
log.info(f'Benjamini-Hochberg FDR correction over {n_all} tests (q=0.05)')

sorted_idx = pvals_df['p_original'].values.argsort()
sorted_pvals = pvals_df['p_original'].values[sorted_idx]

# BH adjusted p-values
bh_adj = np.zeros(n_all)
ranks = np.arange(1, n_all + 1)
bh_raw = sorted_pvals * n_all / ranks

# Enforce monotonicity (step-up): from largest rank down, p_adj_i = min(p_adj_i, p_adj_{i+1})
bh_adj = bh_raw.copy()
for i in range(n_all - 2, -1, -1):
    bh_adj[i] = min(bh_adj[i], bh_adj[i + 1])
bh_adj = np.minimum(bh_adj, 1.0)

# Map back to original order
p_bh = np.zeros(n_all)
p_bh[sorted_idx] = bh_adj
pvals_df['p_bh'] = p_bh

# --- Significance flags ---
alpha = 0.05
pvals_df['significant_original'] = pvals_df['p_original'] < alpha
pvals_df['significant_bonferroni'] = pvals_df['p_bonferroni'].apply(
    lambda x: x < alpha if pd.notna(x) else np.nan
)
pvals_df['significant_bh'] = pvals_df['p_bh'] < alpha

# ─────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────
output_cols = ['test_name', 'dataset', 'outcome', 'p_original', 'p_bonferroni',
               'p_bh', 'significant_original', 'significant_bonferroni',
               'significant_bh', 'test_category']
out = pvals_df[output_cols].sort_values(['test_category', 'dataset', 'p_original'])
out.to_csv(os.path.join(RESULTS, 'mht_corrected_pvalues.csv'), index=False)
log.info(f'Saved mht_corrected_pvalues.csv ({len(out)} rows)')

# Summary
log.info('=== Summary ===')
for cat in ['Primary', 'Secondary', 'Exploratory']:
    subset = out[out['test_category'] == cat]
    if len(subset) == 0:
        continue
    n_sig_orig = subset['significant_original'].sum()
    if cat == 'Primary':
        n_sig_bonf = subset['significant_bonferroni'].sum()
        log.info(f'{cat}: {len(subset)} tests, {n_sig_orig} sig at 0.05, '
                 f'{n_sig_bonf} sig after Bonferroni')
    n_sig_bh = subset['significant_bh'].sum()
    log.info(f'{cat}: {len(subset)} tests, {n_sig_orig} sig at 0.05, '
             f'{n_sig_bh} sig after BH-FDR')

log.info('Done.')
