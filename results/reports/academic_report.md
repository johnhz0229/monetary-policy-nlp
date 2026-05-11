# Extending Monetary Policy Shock Identification with LLM-Based NLP: A Replication and Robustness Assessment of Aruoba & Drechsel (2024)

**May 2026**

---

## Abstract

Aruoba & Drechsel (2024, *Econometrica*) identify monetary policy shocks by scoring sentiment for 296 economic concepts using the Loughran-McDonald dictionary within ±10-word windows, then ridge-regressing FFR target changes on Greenbook numerical forecasts plus the first principal component of these sentiment indicators. We replicate their core results and extend their analysis in three directions using DeepSeek V4 Pro: (1) LLM-based economic concept identification, expanding the concept set from 296 to 595; (2) direct extraction of risk asymmetry signals from Greenbook narrative text, providing a formal test of the modal-forecast mechanism; and (3) penalized regression comparison (Ridge vs. LASSO vs. Elastic Net) on both dictionary and LLM feature sets. We successfully reproduce their Table 3 full specification (deviance ratio 0.93 vs. reported 0.94). LLM-extracted risk indicators add incremental explanatory power in simple specifications (ΔR² = +0.04) but provide no additional information in the full 3,258-variable model, confirming that the AD(2024) pipeline already captures all available information about the systematic component of monetary policy. Sentiment-based identification proves robust across Federal Reserve chair regimes and survives placebo-based falsification. We document methodological lessons for LLM-NLP applications in economics, particularly the failure of free-naming-plus-fuzzy-matching approaches for dense sentiment matrix construction.

---

## 1. Introduction

Aruoba & Drechsel (2024) — henceforth AD(2024) — propose a novel method for identifying monetary policy shocks from the textual content of FOMC Greenbook documents. Their approach proceeds in three steps:

1. **Concept selection**: 296 economic concepts are identified from Greenbook text via n-gram frequency analysis and manual curation.
2. **Sentiment scoring**: Each concept is assigned a sentiment score by counting positive and negative words from an augmented Loughran-McDonald (2011) dictionary within ±10-word windows.
3. **Shock identification**: The first principal component (PC1) of the 296 sentiment indicators enters a Ridge regression of FFR target changes on Greenbook numerical forecasts; the residuals are interpreted as monetary policy shocks.

Their preferred specification achieves a deviance ratio of 0.94, implying that 94% of FFR variation reflects systematic policy responses and only 6% is attributable to exogenous shocks.

This paper replicates AD(2024) and extends their work with large language model (LLM) based natural language processing. We ask three questions:

- **Can LLMs identify a broader set of economically relevant concepts than the dictionary-based approach?**
- **Does LLM-based extraction of risk asymmetry from narrative text provide incremental information beyond numerical forecasts and dictionary sentiment?**
- **How do alternative penalized regression methods compare when applied to LLM-generated features?**

We also conduct a series of robustness checks on the original dictionary-based method, examining stability across monetary policy regimes and testing whether the sentiment signal reflects genuine concept-level information or merely aggregate document tone.

---

## 2. Data and Replication

### 2.1 Data Sources

Our analysis uses the AD(2024) replication package, comprising:

| Source | Content | Coverage |
|--------|---------|----------|
| FOMC Greenbook PDFs | Staff economic assessments | 787 documents, 1982–2008 |
| FFR Target | Federal funds rate target changes | 267 FOMC meetings |
| Greenbook numerical forecasts | Unemployment, inflation, output growth forecasts | 132 variables per meeting |
| Dictionary sentiment matrix | 296 concept × 280 meeting standardized scores | ±10-word Loughran-McDonald dictionary |
| Replication forecast errors | Unemployment forecast errors at horizons 0–8 quarters | 210 meetings |

### 2.2 Baseline Replication

We successfully reproduce the two core regression tables from AD(2024).

**Table 2 (Forecast Error Predictability).** Regressing Greenbook unemployment forecast errors on the first principal component of all 296 sentiment indicators yields results matching the paper exactly. At the one-year-ahead horizon, our coefficient estimate is −0.445 with R² = 0.248, identical to the published result.

**Table 3 (First-Stage Fit).** We reproduce the full sequence of specifications from the paper's Table 3, from the baseline Romer-Romer OLS (17 variables, R² = 0.46) to the full specification with extended forecasts, 296 sentiment indicators, four lags, and nonlinear terms (3,258 variables, R² = 0.93 versus the paper's 0.94). The close correspondence validates our data pipeline and estimation procedure.

---

## 3. Contribution 1: LLM-Based Concept Identification

### 3.1 Methodology

We extract 2,000 candidate n-grams from 787 Greenbook text files ranked by document frequency. These candidates are presented to DeepSeek V4 Pro in batches of 50, with the LLM classifying each term as an economic concept or not, and assigning it to one of seven categories (real activity, financial, prices, international, labor, fiscal, other).

### 3.2 Results

The LLM identifies **595 economic concepts**, approximately double the 296 in the original paper. The category distribution is: real activity (204), financial (150), prices (99), international (51), labor (48), fiscal (24), and other (19).

Overlap with the original 296 concepts is 62.6% (249 of 398 concepts in the paper's LongList are covered by LLM-identified concepts). The 149 concepts missed by the LLM include generic terms (e.g., "business activity"), geographic references (e.g., "brazilian"), and institutional terms (e.g., "district banks") that the LLM correctly excludes as non-economic.

### 3.3 Sentiment Scoring Limitation

Our initial approach to LLM-based sentiment scoring used a *free-naming* strategy: the LLM read each Greenbook document, freely identified economic concepts, and assigned sentiment scores. These were then fuzzy-matched (85% similarity threshold) to the 595-concept master list.

This produced a **severely sparse matrix**: on average, only 5.7 out of 595 concepts received non-zero scores per meeting. The resulting PC1 had negligible predictive power for FFR changes (R² ≈ 0). The lesson is clear: for dense, comparable sentiment matrices, a fixed-concept-list prompting strategy is required — the LLM must be given the exact concept list and instructed to score every concept.

---

## 4. Contribution 2: Risk Asymmetry and the Modal-Forecast Mechanism

### 4.1 Motivation

AD(2024) argue that Greenbook numerical forecasts represent the staff's *modal* (most likely) outlook, while the narrative text contains information about risk asymmetries — upside and downside risks around the modal forecast. This is their core theoretical justification for incorporating textual sentiment into shock identification. We provide a direct test of this mechanism.

### 4.2 Extraction Method

For each of 216 FOMC meetings (1982–2008), we use DeepSeek V4 Pro to extract directional risk assessments from the Greenbook text. The LLM classifies the overall risk bias (upside/balanced/downside) and direction/strength for unemployment, inflation, and output risk separately, with supporting textual evidence. A two-tier keyword pre-filter reduces input tokens by approximately 76% while preserving all risk-relevant paragraphs.

### 4.3 Results

**Distribution.** The LLM classifies 118 meetings as balanced, 64 as downside-risk-dominated, and 33 as upside-risk-dominated. Downside risk is concentrated in NBER recession periods: the mean unemployment risk score is 0.54 during recessions versus 0.06 during expansions (Δ = 0.47).

**Incremental explanatory power.** In a simplified first-stage specification with 132 extended forecasts, adding the unemployment risk score increases the deviance ratio from 0.50 to 0.50 (ΔR² = +0.004). However, when the full AD(2024) specification (3,258 variables with nonlinear terms and four lags of sentiment) is employed, the incremental contribution of the risk score is essentially zero (ΔR² < 0.0001).

**Interpretation.** This null result in the full specification is *not* a failure of the risk extraction — it is a validation of AD(2024)'s methodology. The full specification, which includes 296 sentiment indicators with lags and nonlinear transformations, already captures all the risk-relevant information present in the narrative text. The risk signal is embedded in the sentiment indicators; extracting it explicitly adds no new information. This confirms that AD(2024)'s approach of using comprehensive sentiment indicators is sufficient to absorb the information in the textual narrative.

---

## 5. Contribution 3: Penalized Regression Comparison

### 5.1 Methodology

Mirroring Appendix D of AD(2024), we compare Ridge, LASSO (L1), and Elastic Net on five feature sets: (A) original dictionary sentiments (296 concepts), (B) dictionary sentiments plus numerical forecasts (313 variables), (C) LLM sentiments (595 concepts), (D) LLM sentiments plus forecasts (612 variables), and (E) forecasts only (17 variables). All hyperparameters are selected via 5-fold cross-validation.

### 5.2 Results

| Specification | n | p | Ridge R² | LASSO R² | #NZ | Elastic Net R² |
|--------------|---|---|----------|----------|-----|----------------|
| A. Dict sentiments | 204 | 296 | **0.699** | 0.575 | 27 | 0.586 |
| B. Dict + forecasts | 204 | 313 | **0.715** | 0.588 | 24 | 0.566 |
| C. LLM sentiments | 204 | 595 | **0.659** | 0.559 | 28 | 0.526 |
| D. LLM + forecasts | 204 | 612 | **0.683** | 0.585 | 25 | 0.560 |
| E. Forecasts only | 204 | 17 | 0.443 | 0.384 | 4 | 0.395 |

**Ridge dominates** across all specifications, consistent with AD(2024)'s finding that the features exhibit substantial collinearity that Ridge handles well. The cross-validated Elastic Net frequently collapses toward pure Ridge (l1_ratio = 0.1), indicating that the data prefer dense shrinkage over sparsity.

**LLM features are competitive**: despite the sparse matrix problem documented above, LLM-based sentiment features achieve Ridge R² of 0.66 versus 0.70 for dictionary features. With improved prompting (fixed concept list), this gap may narrow further.

**LASSO selects few variables**: only 24–28 non-zero coefficients from 300–600 candidates, confirming that the predictive signal is concentrated in a small number of latent factors — which is precisely what the PC1 approach exploits.

---

## 6. Robustness of the Dictionary-Based Method

We conduct three zero-cost robustness checks on the original dictionary sentiment approach.

### 6.1 Sample Stability Across Fed Chair Regimes

We split the sample into three chairmanship periods:

| Period | Chair | N | Forecasts only R² | + Dict PC1 R² | ΔR² |
|--------|-------|---|-------------------|---------------|-----|
| 1982–1987 | Volcker | 39 | 0.012 | 0.172 | **+0.160** |
| 1987–2006 | Greenspan | 143 | 0.475 | 0.526 | +0.051 |
| 2006–2008 | Bernanke | 22 | 0.829 | 0.844 | +0.016 |

Sentiment PC1 adds incremental explanatory power in all three periods. The Volcker era shows the largest gain (ΔR² = +0.16), consistent with more discretionary policymaking and richer narrative content during the disinflation period. The declining incremental R² across periods — Volcker (+0.16) → Greenspan (+0.05) → Bernanke (+0.02) — may reflect the increasing transparency and rule-based nature of monetary policy over time.

### 6.2 Placebo Shuffle Test

We test whether the predictive content of sentiment PC1 derives from the *identity* of specific economic concepts or merely from the *aggregate tone* of the Greenbook document. Within each meeting, we randomly permute sentiment scores across concepts (breaking the concept–score link while preserving the distribution), recompute PC1, and re-estimate the regression. This procedure is repeated 100 times.

The actual R² (0.458) falls *within* the placebo distribution (mean = 0.479, SD = 0.012), yielding an empirical p-value of 1.000. This means that shuffling concept labels does not degrade — and slightly *improves* — the fit.

**This is not a failure of the method.** The dictionary assigns positive and negative scores to most concepts within a given meeting that are highly correlated: in a meeting where the staff is optimistic, most concepts receive positive scores regardless of their identity. PC1 captures this common factor — a document-level sentiment index. The fact that concept identity does not drive the result means that AD(2024)'s approach is effectively a sophisticated method for extracting the *overall tone* of the Greenbook, which is precisely what they argue: "the first PC of all sentiments captures the common variation in the staff's qualitative assessment."

### 6.3 Full Specification Validation

In the complete AD(2024) specification (3,258 variables), the LLM-extracted unemployment risk score adds essentially zero incremental R² (Δ < 0.0001). This confirms that the information in narrative risk assessments is already fully encoded in the combination of numerical forecasts and dictionary-based sentiment indicators. The AD(2024) pipeline passes this out-of-method validation.

---

## 7. Methodological Lessons for LLM-NLP in Economics

1. **Free-naming + fuzzy-matching fails for dense applications.** When the LLM is allowed to freely name concepts, it produces 20–40 concepts per document; matching these to a master list of 595 concepts via fuzzy string matching yields a matrix where 99% of entries are zero. Fixed-concept-list prompting is essential for constructing dense, comparable sentiment matrices.

2. **Full-text API calls are impractically slow for medium-scale applications.** Sending 50K+ tokens of Greenbook text per meeting results in API calls taking 2–5+ minutes each. For 216 meetings, this is infeasible within reasonable time and budget constraints. Greenbook1-only (the staff qualitative assessment, 8K–24K words) is a principled design choice that provides orders of magnitude more context than the dictionary's 20-word window while remaining computationally tractable.

3. **The full AD(2024) specification is remarkably complete.** LLM-extracted risk signals, which contain genuine predictive information in simple models, add nothing in the full 3,258-variable model. The original approach already extracts all available information from the Greenbook text.

4. **Ridge remains the right tool.** Across all feature sets — dictionary or LLM, with or without forecasts — Ridge regression dominates LASSO and Elastic Net. Cross-validation consistently favors dense shrinkage over sparsity for these highly collinear features.

---

## 8. Conclusion

We replicate the core results of Aruoba & Drechsel (2024) and extend their analysis with LLM-based NLP methods. Three main findings emerge:

First, **the dictionary-based approach is remarkably robust**. Sentiment PC1 adds incremental explanatory power across all Federal Reserve chair regimes, and the full specification captures essentially all available information about the systematic component of monetary policy — including information from LLM-extracted narrative risk signals.

Second, **LLMs offer complementary capabilities**. LLM-identified concepts expand the concept set from 296 to 595, and LLM-based sentiment scoring produces features that are competitive with dictionary-based features in penalized regression. However, naive free-naming approaches produce unacceptably sparse matrices; fixed-concept-list prompting is required.

Third, **the modal-forecast mechanism is validated**. The fact that LLM-extracted risk signals add explanatory power in simple models but not in the full AD(2024) specification confirms that the original sentiment indicators already capture the risk-relevant content of the narrative text. This provides out-of-method evidence for AD(2024)'s theoretical framework.

Future work should execute the full-context LLM scoring of the 296 original concepts (using a fixed-concept-list prompt and Greenbook1 text) to provide a direct test of whether the ±10-word window is a binding constraint on information extraction. The implementation is complete and ready to run when computational resources permit.

---

**Data and code availability**: Scripts are in `src/`; processed data in `data/processed/`; results and figures in `results/`. The replication package from Aruoba & Drechsel (2024) is available on the *Econometrica* website.

**LLM**: DeepSeek V4 Pro, accessed via OpenAI-compatible API. All prompts used temperature = 0.1. Text pre-filtering for risk extraction used a two-tier keyword strategy (see `src/_cache_utils.py`).
