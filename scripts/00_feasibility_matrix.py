"""
Dataset Feasibility Matrix for Attention-Value Wedge Paper
============================================================
Generates a structured feasibility table for all candidate datasets.
"""

import pandas as pd
import os

# Define feasibility matrix
datasets = [
    {
        "dataset": "YOOCHOOSE (RecSys 2015)",
        "source": "https://s3-eu-west-1.amazonaws.com/yc-rdata/yoochoose-data.7z",
        "public_access": "YES - direct download, no auth",
        "click_info": "YES - 33M click events with session ID + timestamp + item ID + category",
        "purchase_info": "YES - 1.15M buy events with session ID + timestamp + item ID + price + quantity",
        "add_to_cart": "NO",
        "timestamps": "YES - millisecond precision",
        "session_ids": "YES",
        "exposure_order": "Can construct within-session click order from timestamps",
        "hypothesis_coverage": "H1-H3 (no uncertainty/intent proxies beyond category)",
        "key_limitations": "No explicit page position/rank; no search queries; no add-to-cart; category coding is opaque (numbers + special offers); no user IDs across sessions",
        "overall_rating": "HIGH - best fully-public dataset with click+purchase",
        "downloaded": True,
    },
    {
        "dataset": "REES46 Multi-Category Store (via HuggingFace)",
        "source": "https://huggingface.co/datasets/kevykibbz/ecommerce-behavior-data-from-multi-category-store_oct-nov_2019",
        "public_access": "YES - HuggingFace parquet, no auth required",
        "click_info": "YES - millions of 'view' events with timestamps",
        "purchase_info": "YES - purchase events with price",
        "add_to_cart": "YES - cart events present",
        "timestamps": "YES - second precision",
        "session_ids": "YES - user_session field",
        "exposure_order": "Can construct within-session view order from timestamps",
        "hypothesis_coverage": "H1-H5 (has category, brand, price for uncertainty/intent proxies)",
        "key_limitations": "No explicit page position/rank; no search queries; session = user_session UUID (may span long periods); timestamps at second granularity",
        "overall_rating": "HIGHEST - view+cart+purchase funnel with rich item metadata",
        "downloaded": True,
    },
    {
        "dataset": "UCI Clickstream (e-shop clothing 2008)",
        "source": "https://archive.ics.uci.edu/static/public/553/clickstream+data+for+online+shopping.zip",
        "public_access": "YES - direct download",
        "click_info": "YES - 165K clicks with session ID + order + page location",
        "purchase_info": "NO - all rows are clicks, no purchase outcome",
        "add_to_cart": "NO",
        "timestamps": "NO - only year/month/day, no within-day time",
        "session_ids": "YES",
        "exposure_order": "YES - explicit 'order' field (click sequence within session) + page location",
        "hypothesis_coverage": "H1 ONLY - no downstream purchase/conversion data",
        "key_limitations": "CRITICAL: No purchase events. Cannot test H2-H5. Only useful for attention/click-order analysis.",
        "overall_rating": "LOW - click-only, no downstream outcomes",
        "downloaded": True,
    },
    {
        "dataset": "Retail Rocket / Retailrocket",
        "source": "https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset",
        "public_access": "BLOCKED - Kaggle login required, no public mirror with data files",
        "click_info": "YES (if accessible) - view events with visitor ID + timestamp + item ID",
        "purchase_info": "YES (if accessible) - transaction events",
        "add_to_cart": "YES (if accessible) - addtocart events",
        "timestamps": "YES - Unix timestamps",
        "session_ids": "Visitor ID (not session ID)",
        "exposure_order": "Can construct from timestamps within visitor sessions",
        "hypothesis_coverage": "H1-H3 potentially, H4-H5 limited (item properties available but sparse)",
        "key_limitations": "BLOCKED: requires Kaggle authentication. No clean public mirror found. GitHub repos contain notebooks but not data files.",
        "overall_rating": "BLOCKED - worth manual Kaggle download later",
        "downloaded": False,
    },
    {
        "dataset": "Diginetica (CIKM Cup 2016)",
        "source": "https://drive.google.com/drive/folders/0B7XZSACQf0KdXzZFS21DblRxQ3c",
        "public_access": "PARTIAL - Google Drive folder (needs gdown), Kaggle mirror, GitHub LFS mirror",
        "click_info": "YES - train-item-views.csv (product page views) + SERP click grades",
        "purchase_info": "YES - train-purchases.csv with ordernumber + itemId",
        "add_to_cart": "NO",
        "timestamps": "RELATIVE - timeframe (ms since session start) + eventdate (calendar date)",
        "session_ids": "YES - sessionId in all files",
        "exposure_order": "BEST: train-queries.csv 'items' column lists products in DEFAULT RANKED ORDER from search engine",
        "hypothesis_coverage": "H1-H5 (search rank = near-exogenous exposure order; hashed query tokens for intent; categories)",
        "key_limitations": "No absolute timestamps. Requires gdown or Kaggle auth. No explicit license. Query tokens are hashed. 574K sessions, 134M impressions.",
        "overall_rating": "HIGH - search engine ranking provides closest-to-exogenous exposure order.",
        "downloaded": True,
    },
    {
        "dataset": "Coveo SIGIR eCom 2021",
        "source": "https://www.coveo.com/en/resources/datasets/sigir-ecom-2021-data-challenge-dataset",
        "public_access": "FORM REQUIRED - free for research, must submit name/institution and agree to T&C",
        "click_info": "YES - pageview + click events with session_id_hash + ms timestamps",
        "purchase_info": "YES - purchase events (product_action='purchase')",
        "add_to_cart": "YES - add events (product_action='add')",
        "timestamps": "YES - millisecond precision (server_timestamp_epoch_ms)",
        "session_ids": "YES - native session_id_hash",
        "exposure_order": "BEST: has search result impressions (items shown but not clicked) + click/no-click labels",
        "hypothesis_coverage": "H1-H5 (impression data enables true exposure-order analysis; query vectors for intent)",
        "key_limitations": "Requires form submission (not wget-able). ~30M events, 5M sessions, 57K products. Product IDs hashed.",
        "overall_rating": "HIGHEST POTENTIAL - only dataset with impression-level exposure data. Manual form required.",
        "downloaded": False,
    },
    {
        "dataset": "Taobao User Behavior (Alibaba)",
        "source": "https://tianchi.aliyun.com/dataset/649",
        "public_access": "BLOCKED - requires Alibaba Tianchi account",
        "click_info": "YES (if accessible) - page view events",
        "purchase_info": "YES (if accessible) - buy events",
        "add_to_cart": "YES (if accessible) - cart + fav events",
        "timestamps": "YES - Unix timestamps",
        "session_ids": "User ID (sessions must be constructed)",
        "exposure_order": "Must construct sessions from timestamp sequences",
        "hypothesis_coverage": "H1-H3 potentially",
        "key_limitations": "BLOCKED: requires Chinese platform account. Some mirrors exist but quality uncertain.",
        "overall_rating": "BLOCKED - large and useful if accessible",
        "downloaded": False,
    },
]

df = pd.DataFrame(datasets)

# Save to CSV
os.makedirs("/home/user/businessData/results", exist_ok=True)
df.to_csv("/home/user/businessData/results/dataset_feasibility_matrix.csv", index=False)

# Print summary
print("=" * 80)
print("DATASET FEASIBILITY MATRIX")
print("=" * 80)
for _, row in df.iterrows():
    print(f"\n--- {row['dataset']} ---")
    print(f"  Access: {row['public_access']}")
    print(f"  Downloaded: {row['downloaded']}")
    print(f"  Hypothesis coverage: {row['hypothesis_coverage']}")
    print(f"  Rating: {row['overall_rating']}")
    print(f"  Key limitation: {row['key_limitations'][:100]}")

print("\n" + "=" * 80)
print("USABLE DATASETS FOR EMPIRICAL ANALYSIS:")
print("  1. REES46 (HuggingFace) - H1-H5 - view/cart/purchase with metadata")
print("  2. YOOCHOOSE - H1-H3 - click/buy with timestamps")
print("  3. UCI Clickstream - H1 only - click sequence only, no purchases")
print("=" * 80)
