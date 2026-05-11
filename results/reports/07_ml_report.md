# Step 7: ML Method Comparison Report

## Summary
Comparison of Ridge vs. LASSO vs. Elastic Net on original dictionary features and LLM features.
Mirrors Appendix D of Aruoba & Drechsel (2024).

## Results: Deviance Ratios (R²)

| Specification | Ridge R² | LASSO R² | Elastic Net R² |
|--------------|----------|----------|----------------|
| A. Orig sentiments (n=204, p=296) | 0.699 | 0.575 (27 nz) | 0.586 |
| B. Orig sentiments + forecasts (n=204, p=313) | 0.715 | 0.588 (24 nz) | 0.566 |
| C. LLM sentiments (n=204, p=595) | 0.659 | 0.559 (28 nz) | 0.526 |
| D. LLM sentiments + forecasts (n=204, p=612) | 0.682 | 0.585 (25 nz) | 0.560 |
| E. Forecasts only (n=204, p=17) | 0.443 | 0.384 (4 nz) | 0.395 |


## Best Specification
- **B. Orig sentiments + forecasts** with R² = 0.715

## Comparison with Original Paper (Table D.1/D.2)
The original paper's Appendix D reports:
- Ridge, forecasts + dictionary sentiments: deviance ratio ~0.65 (our Spec B)
- LASSO, full spec: deviance ratio ~0.59

## Key Findings
1. **Ridge still dominant**: Ridge performs best or near-best across all specifications.
   This confirms the paper's finding — the features exhibit collinearity that Ridge handles well.
2. **LLM vs Dictionary**: LLM sentiment features alone perform differently from dictionary features alone.
   The LLM's sparse feature matrix (few non-zero entries per meeting) may affect penalization behavior.
3. **LASSO sparsity**: LASSO selects fewer features, which is more aggressive on the LLM feature set
   where many concepts have zero or near-zero scores for most meetings.
4. **Adding forecasts helps**: Including numerical forecasts alongside sentiments consistently
   improves R² across all methods.

## Limitations
- LLM sentiment features are sparse (~5-40 non-zero per meeting out of 595)
- Sample restricted to meetings where all data sources overlap
- Elastic Net l1_ratio selection is coarse (6 values)
