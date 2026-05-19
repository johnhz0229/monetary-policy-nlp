# Does Economic Concept Selection Matter for Monetary Policy Shock Identification?

**Evidence and Mathematical Analysis from Aruoba & Drechsel (2024)**

---

## Overview

This project replicates and extends Aruoba & Drechsel (2024), who identify monetary policy shocks by scoring sentiment for 296 hand-picked economic concepts using the Loughran-McDonald dictionary within $\pm$10-word windows.

We ask: **how robust is this method to the specific choice of concepts?**

### Key Findings

1. **Replication is exact.** All Table 2 horizons match the original. Table 3 specifications reproduced within $\pm$0.01 (CV lambda grid variation).
2. **Sentiment adds value across all Fed chairs.** $\Delta$R$^2$: Volcker +0.16, Greenspan +0.05, Bernanke +0.02.
3. **Concept selection is subjective.** Human vs. LLM curation: only 62.6% overlap (249/296).
4. **Labels barely matter for the fit.** Shuffling all 296 concept labels within each meeting drops R$^2$ by only 0.03 (from 0.96). Shock correlations remain $>$ 0.99.
5. **This holds for an LLM-generated list too.** A 595-concept LLM list with more balanced composition (higher within-meeting variance) produces the same result: shock correlations $>$ 0.99 after shuffling. **Labels are incidental; the number of sampling points is what matters.**
6. **Why?** Within any meeting, most concept scores point the same direction. The common signal dominates regardless of which label is attached to which score. The concept set provides dense temporal coverage of the document — the specific identity of each sampling point is secondary.
7. **The model does not overfit** despite $p/n \approx 16$. Ridge ($\lambda = 924$) successfully shrinks $\sim$3,224 variables spanning $\sim$155 effective dimensions.

---

## Repository Structure

```
├── papers/                              # Reference papers
│   ├── Original_paper_Aruoba_Drechsel.pdf   # AD(2024) Econometrica paper
│   └── Devesh_Thavalam_PeerReview_AD2024.pdf # Peer review (May 2026)
│
├── src/                                 # Python source code
│   ├── 01_extract_text.py               # PDF → TXT extraction (787 Greenbooks)
│   ├── 01a_llm_sentiment_v2.py          # LLM sentiment scoring (296 concepts, full Greenbook1)
│   ├── 02_concept_identification.py     # LLM concept identification (2,000 n-grams → 595 concepts)
│   ├── 03_risk_extraction.py            # LLM risk signal extraction (upside/downside/balanced)
│   ├── 03_sentiment_scoring_llm.py      # LLM-based sentiment scoring for all concepts
│   ├── 04_replicate_baseline.py         # Table 2 replication (forecast error predictability)
│   ├── 05_evaluate_improvements.py      # Early evaluation framework
│   ├── 06_evaluate_improvements.py      # Refined evaluation with LLM concepts vs. original
│   ├── 07_ml_comparison.py              # Table 3 replication (7 Ridge specifications)
│   ├── run_robustness_checks.py         # Check 1-3: Sample split, Placebo shuffle, Risk score
│   ├── run_llm_shuffle_test.py          # Check 4: LLM list shuffle test (Devesh Rec. #3)
│   └── _cache_utils.py                  # DeepSeek prefix cache monitoring
│
├── discussion/                          # Supplementary discussion notes (LaTeX)
│   ├── main.tex                         # Q1–Q4 responses with mathematical proofs
│   └── main.pdf                         # Compiled PDF (10 pages)
│
├── results/                             # Outputs
│   ├── beamer_presentation.tex          # Beamer source (27 slides)
│   ├── beamer_presentation.pdf          # Compiled presentation
│   ├── robustness_checks.ipynb          # Robustness checks notebook
│   ├── tables/                          # CSV/JSON result tables
│   │   ├── check1_sample_split.csv      # Sample split by Fed chair
│   │   ├── check2_placebo.csv           # Dict PC1 placebo shuffle
│   │   ├── check3_risk_score.csv        # Risk score incremental R²
│   │   ├── check4a_composition.csv      # Dict vs. LLM composition
│   │   ├── check4b_baseline.csv         # Dict PC1 vs. LLM PC1 baseline
│   │   ├── check4c_shuffle_pc1.csv      # PC1 shuffle (100 perms × 2 lists)
│   │   ├── check4c_shuffle_summary.csv   # PC1 shuffle summary stats
│   │   ├── check4d_shuffle_full.csv     # Full-column shuffle (100 perms × 2 lists)
│   │   ├── check4d_shuffle_full_summary.csv
│   │   ├── check4e_shock_corr.csv       # Shock correlation distributions
│   │   └── ...                          # Additional replication tables
│   ├── figures/                         # PNG figures
│   │   ├── check4a_composition.png      # Dict vs. LLM composition histograms
│   │   ├── check4c_shuffle_pc1.png      # PC1 shuffle test distributions
│   │   ├── check4d_shuffle_full.png     # Full-column shuffle distributions
│   │   ├── check4e_shock_corr.png       # Shock correlation distributions
│   │   └── ...                          # Additional figures
│   └── reports/                         # Markdown/HTML reports
│
├── data/                                # EXCLUDED from git (see .gitignore)
│   ├── create_sentiments/               # FOMC PDFs, Greenbook forecasts, L-M dictionary
│   ├── replication_files/               # AD(2024) replication package outputs
│   ├── processed/                       # Extracted texts, LLM outputs, concept lists
│   └── VAR_models/                      # BVAR replication code (MATLAB)
│
├── CLAUDE.md                            # Project instructions for AI assistant
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

---

## Quick Start

### Dependencies

```bash
pip install -r requirements.txt
```

Core packages: `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `matplotlib`, `openai`, `python-dotenv`, `pymupdf`

### Reproduce Results (No API Calls)

| Script | Description | Output |
|--------|-------------|--------|
| `src/04_replicate_baseline.py` | Table 2 — forecast error predictability | `results/tables/table2_baseline_replication.csv` |
| `src/07_ml_comparison.py` | Table 3 — 7 first-stage specifications | `results/tables/ml_comparison.csv` |
| `src/run_robustness_checks.py` | Checks 1–3: sample split, placebo, risk score | `results/tables/check1–3_*.csv`, `results/figures/check1–3_*.png` |
| `src/run_llm_shuffle_test.py` | Check 4: LLM list shuffle test | `results/tables/check4a–e_*.csv`, `results/figures/check4a–e_*.png` |

### Full Pipeline (Requires DeepSeek API)

| Script | Description | API Calls |
|--------|-------------|-----------|
| `src/01_extract_text.py` | Extract text from 787 FOMC PDFs | 0 |
| `src/02_concept_identification.py` | LLM concept identification from 2,000 n-grams | $\sim$250 |
| `src/01a_llm_sentiment_v2.py` | LLM sentiment scoring for 296 concepts | $\sim$216 |
| `src/03_risk_extraction.py` | LLM risk signal extraction | $\sim$216 |

---

## Presentation & Discussion

- **Slides:** `results/beamer_presentation.pdf` (27 slides)
- **Discussion notes:** `discussion/main.pdf` (10 pages, Q&A format)
  - **Q1:** Does shuffling concept labels change the Ridge fit? → **No.**
  - **Q2:** Do shuffled shocks produce the same IRFs? → **Yes, mathematically and empirically.**
  - **Q3:** Does the model overfit? → **No.** 5 diagnostic tests confirm.
  - **Q4:** Does the shuffle test hold for an LLM-generated concept list? → **Yes.** Both lists invariant to labels.

---

## Citation

Aruoba, S. B. and Drechsel, T. (2024). "Identifying Monetary Policy Shocks: A Natural Language Approach." *Econometrica*.
