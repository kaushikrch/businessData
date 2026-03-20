# Research Feasibility Assessment: Attention Budget Competition and the Attention-Value Wedge

## 1. Executive Summary

**Verdict: The project is feasible and well-positioned for a top economics or marketing journal.**

The theoretical framework (formalized in `model.tex`) produces seven propositions, each with a clear empirical mapping to the four datasets already analyzed (REES46, YOOCHOOSE, Diginetica, Coveo). The cross-platform variation provides natural tests that no single-dataset study can achieve. The theory makes genuine contributions beyond existing models (Compiani et al. 2024, Cattaneo et al. 2023, Bloedel-Segal 2023, Weitzman 1979) by endogenizing the attention-value wedge as a function of assortment size, ranking calibration, and attention budgets.

## 2. Theory Assessment

### 2.1 Core Intellectual Contribution

The model unifies three previously separate ideas:
1. **Finite attention budgets** (Cattaneo et al., Bloedel-Segal)
2. **Platform-controlled rankings** (Compiani et al.)
3. **Multi-stage search funnels** (Weitzman)

The key novel object is **W(J, ρ, A)**: the wedge as an endogenous, closed-form function of assortment size J, ranking calibration ρ, and attention budget A. No existing paper produces this object.

### 2.2 What Distinguishes This from Existing Work

| Paper | What They Have | What They Lack | What We Add |
|-------|---------------|----------------|-------------|
| Compiani et al. (2024) | Two indices (search, utility) | No attention budget, no overload, wedge is a parameter | Wedge as endogenous function of (J, ρ, A); overload results |
| Cattaneo et al. (2023) | Attention overload axiom | No platform, no ranking, no funnel | Specific mechanism (ranking + budget + concave φ); purchase stage |
| Bloedel-Segal (2023) | Sender competition for attention | No search, no consideration set, no purchase | Consumer actively allocates attention; consideration-to-purchase funnel |
| Weitzman (1979) | Sequential search with per-box costs | No global budget, no platform, no rivalry | Rival attention resource; attention externalities across products |
| Gabaix (2019) | Attention weights on attributes | Operates within-product, not across-product | Product-level attention allocation for platform design |

### 2.3 Proposition-by-Proposition Feasibility

| Proposition | Statement | Empirical Support | Data Quality | Testability |
|------------|-----------|-------------------|-------------|-------------|
| P1 (Overload) | W increases in J | **Strong**: Diginetica (J≈30, W=0.234) > REES46 (W=0.140) > Coveo (J≈2, no wedge) | High | **High** |
| P2 (Calibration) | W decreases in ρ | **Strong**: Coveo (high ρ) = no wedge; REES46/Diginetica (low ρ) = wedge | High | **High** |
| P3 (Interaction) | ∂²W/∂J∂ρ < 0 | **Moderate**: Cross-platform pattern consistent; within-platform tests need more power | Medium | **Medium** |
| P4 (Externality) | Promoting j hurts k | **Directional**: Within-session substitution patterns available but not yet tested | Medium | **Medium** |
| P5 (Welfare Gap) | CS(welfare) > CS(click) | **Structural**: Requires counterfactual estimation on Diginetica | Low (needs estimation) | **Low-Medium** |
| P6 (Expertise) | W decreases in τ_c | **Strong**: Diginetica text search (no rank effect) vs. browse (full wedge); REES46 short vs. long sessions | High | **High** |
| P7 (U-Shape) | Non-monotone attention in rank | **Strong**: Coveo browsing cart rates U-shaped (4.34% → 3.24% → 4.68%) | High | **High** |

### 2.4 Novel Predictions Not in Any Existing Paper

1. **W(J, ρ, A) as a closed-form function** — no paper has this object
2. **The interaction ∂²W/∂J∂ρ < 0** — overload is worse with bad rankings
3. **Attention externality through the budget constraint** in product search
4. **The U-shape prediction** from positional salience × private information
5. **Welfare loss scaling as O(J^{1-γ}·(1-ρ))** — relates assortment size to welfare

## 3. Empirical Assessment

### 3.1 Cross-Dataset Pattern Summary

| Dataset | Effective J | Estimated ρ | Wedge Magnitude | Wedge Direction |
|---------|------------|-------------|-----------------|-----------------|
| REES46 | Medium (~50-100 per category) | Low (browsing order ≈ weak signal) | 0.140 | Click > Purchase |
| Diginetica | 30 (top SERP results) | Low-Medium (search algorithm) | 0.234 | Click > Purchase |
| Coveo Browse | Low (~2 median per session) | High (optimized recommendations) | **None** | Click ≈ Purchase |
| Coveo Search | 20 (ranked results) | High (relevance ranking) | Large click effect, no purchase data | Click >> (purchase N/A) |

**This cross-platform pattern is the paper's strongest empirical asset.** It simultaneously tests P1 (J variation) and P2 (ρ variation) and is hard to explain by any single confound.

### 3.2 Identification Strengths and Weaknesses

**Strengths:**
- **Diginetica provides quasi-exogenous variation**: Rankings are algorithmically assigned, not user-chosen. The negative conditional-conversion result rules out pure quality-ranking confounding.
- **Cross-platform variation**: Comparing Coveo (no wedge) to REES46/Diginetica (wedge) across different ρ levels is a strong falsification test.
- **Within-Diginetica item variation**: Same product at different ranks across queries enables item fixed effects.
- **Large N throughout**: 6.57M (REES46), 4.92M (YOOCHOOSE), 1.79M (Diginetica), 2.54M (Coveo).

**Weaknesses:**
- **No randomized ranking experiment** in any dataset.
- **REES46 browsing order is endogenous** — users choose what to view.
- **Limited price/product attributes** in Diginetica and Coveo for heterogeneity analysis.
- **Cross-platform comparisons assume comparable consumer populations** — maintained assumption.

### 3.3 What Would Strengthen the Paper

| Enhancement | Difficulty | Payoff | Priority |
|------------|-----------|--------|----------|
| Item fixed effects in Diginetica (same product, different ranks) | Medium | High | **1** |
| Pagination RD in Diginetica (page 1 vs. page 2 boundary) | Medium | High | **2** |
| Within-session attention externality test (P4) | Medium | Medium | **3** |
| Structural estimation of (A, λ, β, τ) on Diginetica | High | Very High | **4** |
| Revenue-weighted wedge analysis | Low | Medium | **5** |

## 4. Publication Venue Assessment

### 4.1 Target Venues (ranked)

1. **Marketing Science** — Best fit. Compiani et al. (2024) is there. The click-vs-purchase framing is core marketing science. Cross-platform empirics with structural model is the journal's bread and butter.

2. **Management Science** — Strong fit. Ghose et al. (2014), Derakhshan et al. (2022), and Dinerstein et al. (2018) are in related territory. The platform optimization angle fits well.

3. **Quantitative Marketing and Economics** — Good fit for the structural estimation version. Allows longer papers with detailed appendices.

4. **Econometrica** — Possible if the theoretical contribution (nesting Weitzman/Cattaneo/Compiani) is developed as the primary contribution with empirical illustration. Requires full proofs and more generality.

### 4.2 Paper Structure Recommendation

1. **Introduction** — Motivate with the cross-platform empirical pattern
2. **Model** — Sections 1-3 of model.tex (environment, attention, consideration, purchase)
3. **Equilibrium Analysis** — Optimal attention allocation, consideration sets
4. **The Attention-Value Wedge** — Define W(J, ρ, A); prove Propositions 1-3
5. **Attention Externalities** — Proposition 4
6. **Platform Design** — Ranking optimization; Proposition 5
7. **Consumer Heterogeneity** — Propositions 6-7 (expertise, U-shape)
8. **Empirical Evidence** — Cross-platform tests, position gradients, heterogeneity
9. **Structural Estimation** (if targeting Marketing Science/QME)
10. **Conclusion**

## 5. New Empirical Results (March 20, 2026)

### 5.1 Diginetica Item Fixed Effects (Script: `09_diginetica_item_fe.py`)

**Design:** The same product appears at different rank positions across different search queries. Within-item demeaning absorbs time-invariant product quality q_j, isolating the pure positional effect.

**Sample:** 3.53M item-position observations, 37,633 items with ≥5 query appearances. Mean rank standard deviation within item: 2.53 positions. Mean rank range: 8.0 positions.

**Key Results:**

| Specification | Coefficient | SE | p-value | Interpretation |
|--------------|------------|-----|---------|----------------|
| Click ~ rank_position (item FE) | **-0.000766*** | 0.000032 | ≈0 | Each rank position reduces click prob by 0.077pp |
| Purchase ~ rank_position (item FE) | +0.000007 | 0.000004 | 0.110 | **Zero effect** on purchases |
| Click ~ early_exposure (item FE) | **+0.005308*** | 0.000466 | ≈0 | Top quartile: +0.53pp click advantage |
| Purchase ~ early_exposure (item FE) | -0.000071 | 0.000059 | 0.231 | **Zero effect** on purchases |
| Click ~ top5 (item FE) | **+0.010503*** | 0.000583 | ≈0 | Top 5: +1.05pp click advantage |
| Purchase ~ top5 (item FE) | -0.000056 | 0.000081 | 0.490 | **Zero effect** on purchases |
| Purchase\|Click ~ early_exposure (item FE) | **-0.004091*** | 0.002047 | 0.046 | Early-clicked items convert 0.41pp *worse* |

**Comparison to no-FE baseline:** The click coefficient is *larger* with item FE (-0.000766 vs. -0.000502 without FE), suggesting quality sorting actually *understates* the positional effect. The purchase coefficient remains zero in both specifications.

**Assessment:** This is the paper's strongest causal evidence. Holding product quality fixed, rank shifts attention dramatically but has zero effect on purchases. The wedge is driven by positional salience, not quality sorting. **Proposition 2 (calibration) strongly supported.**

### 5.2 REES46 Within-Session Attention Externality (Script: `10_rees46_externality.py`)

**Design:** Tests whether acting on item j (carting it) affects engagement with subsequent items in the session.

**Sample:** 5.21M item-position observations across 744,693 sessions with ≥3 items.

**Results:**

| Test | Specification | Coefficient | p-value | Direction |
|------|--------------|------------|---------|-----------|
| Cart spillover | was_carted ~ any_cart_before | **+0.535*** | ≈0 | Positive (unexpected) |
| Cart spillover | was_purchased ~ any_cart_before | **+0.261*** | ≈0 | Positive (unexpected) |
| Saturation | was_carted ~ cum_carts_before | **+0.091*** | ≈0 | Positive |
| Early commitment | 2nd-half cart ~ early_cart_in_session | **+0.488*** | ≈0 | Positive |

**Interpretation:** The positive coefficients reflect **session-level selection**, not the attention externality predicted by Proposition 4. Users who cart one item are high-intent shoppers who cart/purchase many items in the same session. The between-session heterogeneity (high-intent vs. low-intent) dominates the within-session rivalry effect.

**Heterogeneity confirms selection:** Short sessions show even larger positive spillovers (72pp cart diff) than long sessions (35pp), consistent with high-intent users having concentrated shopping behavior.

**Assessment:** The current test does **not** identify the attention externality because it confounds within-session rivalry with between-session heterogeneity. **Proposition 4 requires session fixed effects or an instrumental variable design** — e.g., exploiting exogenous variation in whether a specific product is carted (perhaps from price shocks or stock-out events). This is a genuine empirical limitation. Reclassified from "Medium testability" to "Low testability with current data."

### 5.3 Updated Proposition Testability

| Proposition | Previous Rating | New Rating | Change Reason |
|------------|----------------|------------|---------------|
| P1 (Overload) | High | **High** | Unchanged |
| P2 (Calibration) | High | **Very High** | Item FE results confirm wedge survives quality controls |
| P3 (Interaction) | Medium | **Medium** | Unchanged |
| P4 (Externality) | Medium | **Low** | Selection confound in REES46; needs session FE or IV |
| P5 (Welfare Gap) | Low-Medium | **Low-Medium** | Unchanged |
| P6 (Expertise) | High | **High** | Unchanged |
| P7 (U-Shape) | High | **High** | Unchanged |

## 6. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Compiani et al. extend their model to include attention budgets | Low-Medium | High | Move quickly; our cross-platform empirics are distinctive |
| Referee demands randomized experiment | Medium | Medium | Diginetica quasi-exogeneity + Coveo falsification test |
| Structural estimation doesn't converge | Medium | Medium | Can publish reduced-form version first (Marketing Science allows this) |
| Cross-platform comparison rejected as apples-to-oranges | Medium | Medium | Emphasize within-Diginetica and within-Coveo tests as primary; cross-platform as illustration |
| Concurrent paper on attention overload in e-commerce | Low | Medium | Literature search shows no close competitor as of March 2026 |

## 6. Conclusion

The project has:
- **A clear theoretical contribution**: The wedge as an endogenous function W(J, ρ, A) that nests three canonical models
- **Seven testable propositions** with sharp empirical predictions
- **Strong empirical support** from four datasets totaling ~15.8M observations
- **A natural falsification test**: Coveo (no wedge) vs. REES46/Diginetica (wedge)
- **Multiple credible identification strategies**: Diginetica quasi-exogeneity, within-item variation, pagination RD

**Recommended next steps:**
1. Finalize model.tex proofs (convert proof sketches to full proofs)
2. ~~Run item fixed effects specifications on Diginetica~~ **DONE** — wedge confirmed with item FE
3. ~~Test within-session attention externality (Proposition 4)~~ **DONE** — selection confound identified; needs redesign
4. Redesign Proposition 4 test with session fixed effects or IV approach
5. Run pagination RD on Diginetica (page 1 vs. page 2 boundary)
6. Draft paper targeting Marketing Science submission

---
*Assessment date: 2026-03-20 (updated with item FE and externality results)*
*Based on: 22-paper literature collection, 4 analyzed datasets, formal theoretical model, 2 additional empirical analyses*
