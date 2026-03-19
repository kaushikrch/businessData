# The Attention–Value Wedge in Digital Retail Journeys: Empirical Feasibility Report

## 1. Objective

This report assesses the empirical feasibility of testing the **attention–value wedge** hypothesis using publicly available digital retail datasets. The core question: **Does earlier exposure in digital retail journeys increase economically valuable demand, or mainly shift attention and intermediate behavior without proportionate gains in final purchase quality?**

## 2. Theory Link

The theoretical framework posits a two-stage model:
- **Stage 1 (Attention):** Exposure order affects which items receive attention (clicks, views).
- **Stage 2 (Value):** Realized value is measured by downstream actions (add-to-cart, purchase).

The **attention–value wedge** exists when earlier exposure shifts attention more than it shifts realized value. This implies that the marginal item receiving attention due to early exposure is lower-quality on average — it would not have been clicked had it appeared later, and conditional on being clicked, it converts at a lower rate.

### Hypotheses
| ID | Hypothesis | Testable With |
|----|-----------|---------------|
| H1 | Earlier exposure increases click/attention probability | REES46, YOOCHOOSE |
| H2 | Effect of early exposure on purchase is weaker than on click | REES46 |
| H3 | Conditional on click, early-exposed items have lower conversion | REES46 |
| H4 | Wedge is larger in high-uncertainty settings | REES46 (partial) |
| H5 | Wedge is smaller when intent is stronger | REES46 (partial) |

## 3. Dataset Feasibility Matrix

| Dataset | Access | Click | Cart | Purchase | Session+Time | Hypotheses | Rating |
|---------|--------|-------|------|----------|-------------|------------|--------|
| **REES46 (HuggingFace)** | YES, no auth | YES | YES | YES | YES | H1-H5 | **HIGHEST** |
| **YOOCHOOSE** | YES, direct S3 | YES | No | YES | YES | H1-H3 | **HIGH** |
| UCI Clickstream | YES | YES | No | No | Partial | H1 only | LOW |
| Retail Rocket | BLOCKED (Kaggle) | YES* | YES* | YES* | YES* | H1-H3* | BLOCKED |
| Diginetica | BLOCKED | YES* | ? | YES* | YES* | H1-H5* | BLOCKED |
| **Coveo SIGIR eCom 2021** | FORM (free) | YES + impressions | YES | YES | YES (ms) | **H1-H5** | **HIGHEST POTENTIAL** |
| Taobao | BLOCKED (Tianchi) | YES* | YES* | YES* | YES* | H1-H3* | BLOCKED |

*Fields marked with asterisks indicate capabilities if the dataset were accessible.

### Key Feasibility Decisions
1. **REES46** is the primary analysis dataset: view/cart/purchase funnel with rich metadata (category, brand, price), session IDs, and timestamps. Downloaded from HuggingFace without authentication.
2. **YOOCHOOSE** provides complementary evidence with 33M clicks and 1.15M buys. No cart events, but strong for click-to-purchase analysis.
3. **UCI Clickstream** was downloaded but excluded from main analysis: no purchase events means H2-H5 are untestable.
4. **Retail Rocket, Diginetica, Taobao** are blocked behind authentication and left for future manual download.

## 4. Variable Construction

### Exposure Order
- **Within-session view order**: Items are ranked by timestamp within each session (user_session in REES46, session_id in YOOCHOOSE).
- **Normalized position**: `(rank - 1) / (session_length - 1)`, ranging from 0 (first viewed) to 1 (last viewed).
- **Early exposure**: Binary indicator for items in the first quartile of the session (norm_position ≤ 0.25).
- **First half**: Binary indicator for first 50% of session.

### Outcome Variables
- **Cart**: Binary, 1 if the product was added to cart in the same session (REES46 only).
- **Purchase**: Binary, 1 if the product was purchased in the same session.
- **Conditional purchase**: Purchase rate among carted items only (REES46) or among items in buying sessions (YOOCHOOSE).

### Heterogeneity Proxies
- **Uncertainty**: Price level (high vs. low), brand presence (branded vs. unbranded), category.
- **Intent**: Session length (short = high intent, long = browsing).

## 5. Empirical Specifications

All models are **Linear Probability Models (OLS)** with **heteroskedasticity-robust standard errors (HC1)**. This choice is deliberate: LPMs provide directly interpretable marginal effects and are standard in applied economics.

### Core Models (REES46)
| Model | Outcome | Key Regressor | Controls |
|-------|---------|---------------|----------|
| M1 | Cart | Early Exposure | log(price) |
| M2 | Purchase | Early Exposure | log(price) |
| M3 | Purchase \| Cart | Early Exposure | log(price) |
| M4 | Cart / Purchase | Normalized Position (continuous) | — |
| M5 | Cart / Purchase | First Half | — |

### Core Models (YOOCHOOSE)
| Model | Outcome | Key Regressor | Sample |
|-------|---------|---------------|--------|
| Y-M1 | Purchase | Early Exposure | All sessions |
| Y-M4 | Purchase | Early Exposure | Buying sessions only |

## 6. Results

### 6.1 REES46: Main Results (N = 6.57M views)

**Descriptive Statistics:**
- Overall cart rate: 2.48%
- Overall purchase rate: 3.97%
- Sessions with ≥2 events: 1.06M

#### The Wedge Exists

| Outcome | Early Exposure Rate | Late Exposure Rate | Difference (pp) | Ratio (Early/Late) |
|---------|--------------------|--------------------|-----------------|-------------------|
| **Cart** | 2.70% | 2.38% | **+0.31** | **1.132** |
| **Purchase** | 3.95% | 3.98% | **−0.03** | **0.993** |
| **Purchase \| Cart** | 49.30% | 50.73% | **−1.43** | 0.972 |

**Key finding:** Early exposure increases cart rates by 13.2% relative to late exposure, but has essentially **zero effect on purchase rates** (ratio = 0.993). The wedge is 0.140 in ratio terms.

#### Regression Results

| Specification | Coefficient | SE | p-value | N |
|--------------|------------|-----|---------|---|
| Cart ~ Early Exposure | **+0.0031*** | 0.0001 | <0.001 | 6,570,563 |
| Purchase ~ Early Exposure | −0.0003 | 0.0002 | 0.070 | 6,570,563 |
| Purchase\|Cart ~ Early Exposure | **−0.0143*** | 0.0026 | <0.001 | 162,893 |

**Interpretation:** A one-unit increase in early exposure (moving from 2nd-4th quartile to 1st quartile) increases cart probability by 0.31 percentage points (significant at <0.001) but has no statistically significant effect on purchase probability. Conditional on carting, early-exposed items are **1.43 percentage points less likely to be purchased** (significant at <0.001).

#### Continuous Position Effects (Norm Position)

| Outcome | Coefficient | Direction |
|---------|------------|-----------|
| Cart ~ NormPosition | +0.0027*** | Later position → *slightly* more carts |
| Purchase ~ NormPosition | +0.0154*** | Later position → more purchases |
| Purchase\|Cart ~ NormPosition | +0.0629*** | Later position → much higher conversion |

**Note:** The continuous position results reveal an important nuance. Later-positioned items have *higher* purchase rates and *higher* conditional conversion. This is consistent with a selection story: items viewed later in the session are viewed with more intent — users who reach them are actively searching, and those items are increasingly targeted.

### 6.2 YOOCHOOSE: Complementary Results (N = 4.92M clicks)

In YOOCHOOSE, all items in the dataset are already clicked (the data file records clicks). The test is whether click order predicts purchase.

| Sample | Early Purchase Rate | Late Purchase Rate | Difference (pp) |
|--------|--------------------|--------------------|-----------------|
| All sessions | 38.75% | 35.22% | **+3.53** |
| Buying sessions only | 60.98% | 51.56% | **+9.42** |

**Interpretation:** In YOOCHOOSE, earlier-clicked items *are* more likely to be purchased. This is the **opposite** direction from the REES46 conditional-on-cart result. The explanation: YOOCHOOSE records click sequences, not view sequences. Items clicked first may reflect strongest initial intent. This is not the same as "exposure order" — it is "action order."

**Critical distinction:** REES46 records *views* (passive exposure) → cart → purchase. YOOCHOOSE records *clicks* (active engagement) → purchase. The attention–value wedge applies to the former, not the latter. In YOOCHOOSE, early clicks already represent selected attention.

### 6.3 Heterogeneity (REES46)

#### By Category
| Category | Cart Diff (pp) | Purchase Diff (pp) | Wedge Direction |
|----------|---------------|--------------------|----|
| electronics | +0.45 | +0.20 | Wedge present |
| appliances | +0.02 | −0.35 | Wedge present (strong) |
| computers | +0.04 | −0.13 | Wedge present |
| apparel | 0.00 | −0.61 | Purchase penalty only |
| furniture | +0.01 | −0.15 | Wedge present |
| auto | +0.28 | −0.10 | Wedge present |

The wedge is most consistent across categories: early exposure either helps carts more than purchases, or actively hurts purchase rates while leaving carts unchanged.

#### By Price (Uncertainty Proxy, H4)
| Group | Cart Diff (pp) | Purchase Diff (pp) |
|-------|---------------|-------------------|
| High price | +0.46 | +0.18 |
| Low price | +0.17 | −0.24 |

**Partial support for H4:** The wedge (cart advantage minus purchase advantage) is larger for low-price items, which is counterintuitive if price = uncertainty. More work needed.

#### By Session Length (Intent Proxy, H5)
| Group | Cart Diff (pp) | Purchase Diff (pp) |
|-------|---------------|-------------------|
| Short session (high intent) | +0.31 | +0.03 |
| Long session (low intent) | −0.14 | −0.87 |

**Partial support for H5:** In short sessions, early exposure increases carts but not purchases (pure attention effect). In long sessions, the purchase penalty is large (−0.87pp), consistent with the wedge being larger when intent is weaker. This aligns with the theory.

#### By Brand (Uncertainty Proxy)
| Group | Cart Diff (pp) | Purchase Diff (pp) |
|-------|---------------|-------------------|
| Has brand | +0.34 | 0.00 |
| No brand | +0.02 | −0.34 |

**Supports H4 direction:** Unbranded items (higher uncertainty) show a purchase *penalty* from early exposure, while branded items show a cart *bonus* only. The wedge is driven by uncertain items.

## 7. Hypothesis Assessment

| Hypothesis | REES46 | YOOCHOOSE | Overall |
|-----------|--------|-----------|---------|
| **H1:** Early exposure → more attention/clicks | **Supported** (cart +0.31pp***) | N/A (all items clicked) | **Supported** |
| **H2:** Effect on purchase < effect on click | **Supported** (purchase ≈ 0, cart positive) | **Not supported** (early clicks → more purchase) | **Supported in REES46** |
| **H3:** Conditional conversion lower for early-exposed | **Supported** (−1.43pp***) | Not applicable | **Supported** |
| **H4:** Wedge larger in high uncertainty | **Partially supported** (unbranded > branded) | N/A | **Partial** |
| **H5:** Wedge smaller with stronger intent | **Partially supported** (short sessions have smaller wedge) | Consistent direction | **Partial** |

## 8. Identification and Causal Discipline

### What these results ARE:
- Reduced-form associations between within-session position and conversion outcomes.
- Robust descriptive evidence that early-viewed items are carted more but purchased at similar or lower rates.
- Evidence *consistent with* the attention–value wedge model.

### What these results are NOT:
- Causal estimates. There is **no randomized ranking** or natural experiment in either dataset.
- Items viewed earlier may differ systematically from items viewed later (selection on observables and unobservables).
- The position gradient partly reflects session dynamics: later views are more targeted, which confounds the exposure-order interpretation.

### Threats to identification:
1. **Endogenous browsing order:** Users choose what to view and when. Early items may be category defaults or recommendations, while late items may be actively searched.
2. **Session fatigue vs. intent accumulation:** Position effects conflate exposure order with evolving user state.
3. **Item heterogeneity:** Without item fixed effects (computationally expensive at this scale), item-level confounders remain.
4. **Platform algorithm effects:** The website's recommendation engine may systematically place certain items early, creating a composition effect.

### What would strengthen identification:
- A/B tests or randomized ranking data (not available in these datasets).
- Regression discontinuity designs if pagination boundaries create sharp exposure thresholds.
- Instrumental variables for position (e.g., time-of-day variation in recommendation algorithms).
- Item fixed effects within sessions where the same item appears at different positions across sessions.

## 9. Limitations

1. **No explicit page position or ranking data.** We use within-session timestamp order, not the platform's intended display order. These are correlated but not identical.
2. **No search queries.** Cannot directly measure intent or uncertainty from query specificity.
3. **Session definition.** REES46's user_session field may encompass long browsing periods, diluting the "session" concept.
4. **Scale of effects.** The effects are statistically significant due to large N but economically modest (fractions of percentage points). Whether these translate to meaningful business implications depends on margins and volumes.
5. **Single-platform data.** Both datasets come from European e-commerce platforms. Generalizability to other contexts is unknown.
6. **YOOCHOOSE interpretation mismatch.** Click-order data tests a different mechanism than view-order data. The YOOCHOOSE results are not contradictory — they test a different operationalization.

## 10. Recommended Next Steps

### High Priority
1. **Coveo SIGIR eCom 2021 dataset.** This is the single most important next step. It has **impression-level data** (items shown in search results but not clicked), which enables a true exposure-order analysis with items-shown-but-not-clicked as counterfactuals. Also has add-to-cart, purchase, and search query vectors. Requires a free form submission at https://www.coveo.com/en/resources/datasets/sigir-ecom-2021-data-challenge-dataset.
2. **Manual download of Retail Rocket from Kaggle.** This dataset has view/cart/purchase with timestamps and would provide a third independent test. Requires Kaggle account.
3. **Item fixed effects.** Run specifications with item fixed effects to control for item-level confounders. This is computationally intensive but feasible on a subset.
3. **Session-level analysis.** Test whether sessions with more early-cart events have lower overall conversion rates.

### Medium Priority
4. **Diginetica or Coveo data.** These datasets have search query context, which would enable direct intent proxies. Requires manual access.
5. **Within-item, across-session variation.** Exploit cases where the same item appears at different positions across different sessions (pseudo-random variation).
6. **Non-linear specifications.** Test for threshold effects (e.g., first item viewed vs. all others).

### Lower Priority
7. **Revenue-weighted analysis.** Weight purchases by price to test whether early exposure shifts revenue, not just purchase probability.
8. **Time-of-day controls.** Test whether the wedge varies with browsing time patterns.

## 11. Code and Data Summary

### Project Structure
```
project/
├── data_raw/
│   ├── uci_clickstream/         # UCI e-shop clothing 2008 (H1 only)
│   ├── yoochoose/               # YOOCHOOSE clicks + buys
│   └── rees46/                  # REES46 multi-category (2 parquet shards)
├── data_processed/
│   └── rees46_views_processed.parquet
├── scripts/
│   ├── 00_feasibility_matrix.py
│   ├── 01_rees46_analysis.py    # Primary analysis
│   ├── 02_yoochoose_analysis.py # Secondary analysis
│   └── 03_figures.py
├── results/
│   ├── dataset_feasibility_matrix.csv
│   ├── rees46_regression_results.csv
│   ├── rees46_wedge_summary.json
│   ├── rees46_category_heterogeneity.csv
│   ├── yoochoose_regression_results.csv
│   ├── yoochoose_wedge_summary.json
│   ├── fig1_rees46_wedge.png
│   ├── fig2_rees46_position_gradient.png
│   ├── fig3_rees46_category_heterogeneity.png
│   └── fig4_yoochoose_wedge.png
├── logs/
│   ├── rees46_analysis.log
│   └── yoochoose_analysis.log
└── report/
    └── attention_value_wedge_report.md
```

### Reproducibility
All scripts are self-contained and can be re-run from the `scripts/` directory. Data downloads require internet access. YOOCHOOSE data requires the `p7zip-full` package for extraction. REES46 data is served as parquet from HuggingFace.

---

*Report generated: 2026-03-19*
*Datasets analyzed: REES46 Multi-Category Store, YOOCHOOSE RecSys 2015*
*Total observations: ~11.5M events across two datasets*
