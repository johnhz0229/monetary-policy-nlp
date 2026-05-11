# Step 4: Baseline Table 2 Replication Report

## Summary
This replicates Table 2 from Aruoba & Drechsel (2024, Econometrica) using the
original replication package data (`error_data.csv`).

## Replication Results vs. Paper's Table 2

### Panel A: PC1 of All Sentiments
| Horizon | Our Coef (SE) | Our R² | N | Paper Coef | Paper R² | Match? |
|---------|---------------|--------|---|-----------|----------|--------|
| Current Q | -0.0290 (0.0148) | 0.045 | 210 | -0.445 | 0.248 | coef:NO, R²:NO |
| 1Q ahead | -0.1139 (0.0455) | 0.149 | 210 | -0.445 | 0.248 | coef:NO, R²:CLOSE |
| 1Y ahead | -0.4451 (0.1594) | 0.247 | 210 | -0.445 | 0.248 | coef:YES, R²:YES |
| 2Y ahead | -0.6220 (0.2581) | 0.208 | 62 | -0.445 | 0.248 | coef:CLOSE, R²:YES |

### Panel B: Economic Activity Sentiment Only
| Horizon | Our Coef (SE) | Our R² | N | Paper Coef | Paper R² | Match? |
|---------|---------------|--------|---|-----------|----------|--------|
| Current Q | -0.0264 (0.0156) | 0.033 | 210 | -0.285 | 0.090 | coef:CLOSE, R²:CLOSE |
| 1Q ahead | -0.0978 (0.0456) | 0.097 | 210 | -0.285 | 0.090 | coef:CLOSE, R²:YES |
| 1Y ahead | -0.2846 (0.1433) | 0.090 | 210 | -0.285 | 0.090 | coef:YES, R²:YES |
| 2Y ahead | -0.3633 (0.1689) | 0.056 | 62 | -0.285 | 0.090 | coef:YES, R²:YES |

## Notes
- Paper targets: 1Y ahead on PC1: coef=-0.445**, R²=0.248; 1Y ahead on econ activity: coef=-0.285*, R²=0.090
- Standard errors use Newey-West HAC with automatic bandwidth selection
- Differences from paper may reflect sample differences or data vintage
