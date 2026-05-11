# Does Economic Concept Selection Matter for Monetary Policy Shock Identification?

**Evidence and Mathematical Analysis from Aruoba & Drechsel (2024, *Econometrica*)**

---

## Overview

This project replicates and extends Aruoba & Drechsel (2024), who identify monetary policy shocks by scoring sentiment for 296 hand-picked economic concepts using the Loughran-McDonald dictionary within ±10-word windows.

We ask: **how robust is this method to the specific choice of concepts?**

### Key Findings

1. **Concept selection is subjective.** Human vs. LLM curation: only 62.6% overlap.
2. **Labels barely matter for the original list.** Shuffling all 296 concept labels drops R² by only 0.03 out of 0.96.
3. **This is because 86% of the original concepts are pro-cyclical** — they point the same direction within any meeting.
4. **List composition determines regularization.** A balanced (50/50) concept list causes Ridge cross-validation to select near-zero penalty, leading to overfitting. The method's robustness is *conditional* on the list being dominated by a single common factor.
5. **Mathematical proofs** show that under a strong common factor, shuffled-label shocks are nearly identical to original shocks, and the CV-optimal penalty is a function of the singular value spectrum.

---

## Repository Structure

```
├── paper/                          # LaTeX article
│   ├── main.tex                    # Main article with mathematical proofs
│   └── main.pdf                    # Compiled PDF
│
├── src/                            # Python source code
│   ├── 02_concept_identification.py # N-gram + LLM concept classification
│   ├── 03_risk_extraction.py       # LLM risk signal extraction
│   ├── 04_replicate_baseline.py    # Table 2 replication
│   ├── 07_ml_comparison.py         # Ridge/LASSO/Elastic Net comparison
│   ├── 01a_llm_sentiment_v2.py     # Fixed-concept-list LLM scoring (design only)
│   ├── run_robustness_checks.py    # Placebo + sample split + risk score tests
│   └── _cache_utils.py             # DeepSeek cache monitoring utilities
│
├── results/                        # Output files
│   ├── beamer_presentation.pdf     # 21-slide presentation
│   ├── beamer_presentation.tex     # Beamer source
│   ├── robustness_checks.ipynb     # Robustness checks notebook
│   ├── tables/                     # CSV and JSON result tables
│   ├── figures/                    # PNG figures
│   └── reports/                    # Markdown and HTML reports
│
├── CLAUDE.md
├── requirements.txt
└── README.md
```

**Note:** The `data/` directory (787 Greenbook PDFs, replication package, extracted texts) is excluded from version control via `.gitignore`.

---

## Reproduction

### Dependencies

```bash
pip install -r requirements.txt
```

Core packages: `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `matplotlib`, `openai`, `python-dotenv`

### Key Scripts

| Script | Description | API Calls |
|--------|-------------|-----------|
| `src/04_replicate_baseline.py` | Replicate Table 2 | 0 |
| `src/07_ml_comparison.py` | Replicate Table 3 (7 specs) | 0 |
| `src/run_robustness_checks.py` | Placebo shuffle + sample split + risk score | 0 |
| `src/02_concept_identification.py` | LLM concept identification | ~250 |
| `src/03_risk_extraction.py` | LLM risk signal extraction | ~216 |

---

## Citation

Aruoba, S. B. and Drechsel, T. (2024). "Identifying Monetary Policy Shocks: A Natural Language Approach." *Econometrica*.
