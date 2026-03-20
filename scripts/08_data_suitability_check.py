"""
Step 1: Verify Data Suitability for Structural Estimation
==========================================================
Key question: Do Diginetica and Coveo have sufficient WITHIN-PRODUCT
rank variation across sessions? This is critical for identifying the
rank effect function f(r) separately from product fixed effects.

If product j always appears at rank 3, we cannot distinguish:
  "rank 3 gets more clicks" from "product j is inherently popular"

We need the SAME product appearing at DIFFERENT ranks across sessions.
"""

import pandas as pd
import numpy as np
import os
import ast

RESULTS_DIR = "/home/user/businessData/results"

print("=" * 70)
print("DIGINETICA: Within-Product Rank Variation")
print("=" * 70)

# Load Diginetica queries with item rankings
digi_dir = "/home/user/businessData/data_raw/diginetica"
queries = pd.read_csv(os.path.join(digi_dir, "train-queries.csv"), sep=";")
print(f"Total queries: {len(queries):,}")

# Parse items column: list of product IDs in ranked order
np.random.seed(42)
sample = queries.sample(min(50000, len(queries)), random_state=42)

rows = []
for _, row in sample.iterrows():
    try:
        items = ast.literal_eval(str(row['items']))
    except:
        continue
    if not isinstance(items, list) or len(items) < 2:
        continue
    sid = row['session_id']
    for rank, pid in enumerate(items[:30], 1):
        rows.append((sid, pid, rank))

ddf = pd.DataFrame(rows, columns=['session_id', 'product_id', 'rank'])
print(f"Item-position pairs: {len(ddf):,}")
print(f"Unique products: {ddf['product_id'].nunique():,}")
print(f"Unique sessions: {ddf['session_id'].nunique():,}")

# Within-product rank variation
product_rank_stats = ddf.groupby('product_id').agg(
    n_appearances=('rank', 'count'),
    n_unique_ranks=('rank', 'nunique'),
    mean_rank=('rank', 'mean'),
    std_rank=('rank', 'std'),
    min_rank=('rank', 'min'),
    max_rank=('rank', 'max'),
).reset_index()

print(f"\n--- Products by number of appearances ---")
for thresh in [1, 2, 5, 10, 20, 50, 100]:
    n = (product_rank_stats['n_appearances'] >= thresh).sum()
    pct = n / len(product_rank_stats) * 100
    print(f"  Appear >= {thresh:3d} times: {n:,} products ({pct:.1f}%)")

multi = product_rank_stats[product_rank_stats['n_appearances'] >= 2]
print(f"\n--- Within-product rank variation (products appearing >= 2 times) ---")
print(f"  N products: {len(multi):,}")
print(f"  Mean unique ranks per product: {multi['n_unique_ranks'].mean():.2f}")
print(f"  Median unique ranks per product: {multi['n_unique_ranks'].median():.0f}")
print(f"  Mean rank std: {multi['std_rank'].mean():.2f}")
print(f"  Median rank range (max-min): {(multi['max_rank'] - multi['min_rank']).median():.0f}")

# Products with GOOD variation (appear 5+ times, 3+ unique ranks)
good = product_rank_stats[(product_rank_stats['n_appearances'] >= 5) &
                           (product_rank_stats['n_unique_ranks'] >= 3)]
print(f"\n--- Products with GOOD variation (>=5 appearances, >=3 unique ranks) ---")
print(f"  N products: {len(good):,}")
print(f"  Mean rank std: {good['std_rank'].mean():.2f}")
print(f"  These products appear in {ddf[ddf['product_id'].isin(good['product_id'])].shape[0]:,} item-position pairs")

# Example: show a few products with variation
print(f"\n--- Example products with rank variation ---")
examples = good.nlargest(5, 'n_appearances')
for _, p in examples.iterrows():
    pid = p['product_id']
    ranks = ddf[ddf['product_id'] == pid]['rank'].values
    print(f"  Product {pid}: appears {p['n_appearances']} times, "
          f"ranks={sorted(set(ranks))[:15]}{'...' if len(set(ranks))>15 else ''}, "
          f"mean={p['mean_rank']:.1f}, std={p['std_rank']:.1f}")

# Save
product_rank_stats.to_csv(os.path.join(RESULTS_DIR, "diginetica_product_rank_variation.csv"), index=False)

print("\n" + "=" * 70)
print("COVEO SEARCH: Within-Product Rank Variation")
print("=" * 70)

coveo_dir = "/home/user/businessData/data_raw/coveo/train"
search = pd.read_csv(os.path.join(coveo_dir, "search_train.csv"),
                      usecols=['session_id_hash', 'clicked_skus_hash',
                               'product_skus_hash', 'server_timestamp_epoch_ms'])

search = search.dropna(subset=['product_skus_hash']).copy()
np.random.seed(42)
if len(search) > 50000:
    search = search.sample(50000, random_state=42)

print(f"Search events sampled: {len(search):,}")

rows = []
for _, row in search.iterrows():
    try:
        items = ast.literal_eval(row['product_skus_hash'])
    except:
        continue
    if not isinstance(items, list) or len(items) < 2:
        continue
    sid = row['session_id_hash']
    for rank, sku in enumerate(items[:30], 1):
        rows.append((sid, sku, rank))

cdf = pd.DataFrame(rows, columns=['session_id', 'product_id', 'rank'])
print(f"Item-position pairs: {len(cdf):,}")
print(f"Unique products: {cdf['product_id'].nunique():,}")
print(f"Unique sessions: {cdf['session_id'].nunique():,}")

coveo_stats = cdf.groupby('product_id').agg(
    n_appearances=('rank', 'count'),
    n_unique_ranks=('rank', 'nunique'),
    mean_rank=('rank', 'mean'),
    std_rank=('rank', 'std'),
    min_rank=('rank', 'min'),
    max_rank=('rank', 'max'),
).reset_index()

print(f"\n--- Products by number of appearances ---")
for thresh in [1, 2, 5, 10, 20, 50, 100]:
    n = (coveo_stats['n_appearances'] >= thresh).sum()
    pct = n / len(coveo_stats) * 100
    print(f"  Appear >= {thresh:3d} times: {n:,} products ({pct:.1f}%)")

multi_c = coveo_stats[coveo_stats['n_appearances'] >= 2]
print(f"\n--- Within-product rank variation (products appearing >= 2 times) ---")
print(f"  N products: {len(multi_c):,}")
print(f"  Mean unique ranks per product: {multi_c['n_unique_ranks'].mean():.2f}")
print(f"  Median unique ranks per product: {multi_c['n_unique_ranks'].median():.0f}")
print(f"  Mean rank std: {multi_c['std_rank'].mean():.2f}")

good_c = coveo_stats[(coveo_stats['n_appearances'] >= 5) &
                      (coveo_stats['n_unique_ranks'] >= 3)]
print(f"\n--- Products with GOOD variation (>=5 appearances, >=3 unique ranks) ---")
print(f"  N products: {len(good_c):,}")
if len(good_c) > 0:
    print(f"  Mean rank std: {good_c['std_rank'].mean():.2f}")
    print(f"  These products appear in {cdf[cdf['product_id'].isin(good_c['product_id'])].shape[0]:,} item-position pairs")

    print(f"\n--- Example products with rank variation ---")
    examples_c = good_c.nlargest(5, 'n_appearances')
    for _, p in examples_c.iterrows():
        pid = p['product_id']
        ranks = cdf[cdf['product_id'] == pid]['rank'].values
        print(f"  Product {pid[:20]}...: appears {p['n_appearances']} times, "
              f"{p['n_unique_ranks']} unique ranks, "
              f"mean={p['mean_rank']:.1f}, std={p['std_rank']:.1f}")

coveo_stats.to_csv(os.path.join(RESULTS_DIR, "coveo_product_rank_variation.csv"), index=False)

print("\n" + "=" * 70)
print("VERDICT: Data Suitability for Structural Estimation")
print("=" * 70)

digi_good_n = len(good)
digi_good_obs = ddf[ddf['product_id'].isin(good['product_id'])].shape[0]
coveo_good_n = len(good_c)
coveo_good_obs = cdf[cdf['product_id'].isin(good_c['product_id'])].shape[0] if len(good_c) > 0 else 0

print(f"""
DIGINETICA:
  Products with good rank variation: {digi_good_n:,}
  Item-position pairs from those products: {digi_good_obs:,}
  VERDICT: {'SUFFICIENT' if digi_good_n > 500 else 'INSUFFICIENT'} for structural estimation

COVEO SEARCH:
  Products with good rank variation: {coveo_good_n:,}
  Item-position pairs from those products: {coveo_good_obs:,}
  VERDICT: {'SUFFICIENT' if coveo_good_n > 500 else 'INSUFFICIENT'} for structural estimation
""")
