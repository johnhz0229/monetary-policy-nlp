"""
Step 4: Replicate Table 2 (forecast error predictability) using original sentiment indicators.
Uses error_data.csv from the replication package which has pre-computed forecast errors,
PC1 of sentiments, and business activity sentiment.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import warnings

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Load data ---
error_path = PROJECT_ROOT / "data" / "replication_tables_forecast_error" / "error_data.csv"
df = pd.read_csv(error_path)
print(f"Loaded error_data: {df.shape[0]} observations")

# Define forecast horizons matching Table 2
HORIZONS = {
    "Current Q": 0,
    "1Q ahead": 1,
    "1Y ahead": 4,
    "2Y ahead": 8,
}

def newey_west_se(X, y):
    """Run OLS with Newey-West HAC standard errors (optimal bandwidth)."""
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const, missing="drop")
    # Use 'HAC' covariance estimator with automatic lag selection
    results = model.fit(cov_type="HAC", cov_kwds={"maxlags": None}, use_t=True)
    return results

# Prepare results table
regression_results = []

print("\n=== Table 2 Replication ===\n")

for label, h in HORIZONS.items():
    error_col = f"error_unemp_{h}"

    # Filter non-missing
    data_subset = df[[error_col, "pc1_std", "businessactivity"]].dropna()

    if len(data_subset) < 20:
        print(f"  {label} (h={h}): insufficient data (n={len(data_subset)})")
        continue

    y = data_subset[error_col].values
    n = len(y)

    # Regression 1: PC1 of all sentiments
    X_pc1 = data_subset[["pc1_std"]].values
    r1 = newey_west_se(X_pc1, y)
    coef_pc1 = float(r1.params[1])
    se_pc1 = float(r1.bse[1])
    r2_pc1 = float(r1.rsquared)
    pval1 = float(r1.pvalues[1])
    stars_pc1 = "***" if pval1 < 0.01 else ("**" if pval1 < 0.05 else ("*" if pval1 < 0.1 else ""))
    print(f"  {label} on PC1: coef={coef_pc1:.4f}{stars_pc1}, se={se_pc1:.4f}, R²={r2_pc1:.3f}, N={n}")

    regression_results.append({
        "horizon": label,
        "predictor": "PC1 (all sentiments)",
        "coefficient": round(coef_pc1, 4),
        "se_newy_west": round(se_pc1, 4),
        "r_squared": round(r2_pc1, 4),
        "n": n,
        "significance": stars_pc1,
        "paper_coefficient": "-0.445",
        "paper_r2": "0.248",
    })

    # Regression 2: Economic activity sentiment alone
    X_econ = data_subset[["businessactivity"]].values
    r2_econ = newey_west_se(X_econ, y)
    coef_econ = float(r2_econ.params[1])
    se_econ = float(r2_econ.bse[1])
    r2_econ_val = float(r2_econ.rsquared)
    pval_econ = float(r2_econ.pvalues[1])
    stars_econ = "***" if pval_econ < 0.01 else ("**" if pval_econ < 0.05 else ("*" if pval_econ < 0.1 else ""))
    print(f"  {label} on EconActivity: coef={coef_econ:.4f}{stars_econ}, se={se_econ:.4f}, R²={r2_econ_val:.3f}, N={n}")

    regression_results.append({
        "horizon": label,
        "predictor": "Economic activity sentiment",
        "coefficient": round(coef_econ, 4),
        "se_newy_west": round(se_econ, 4),
        "r_squared": round(r2_econ_val, 4),
        "n": n,
        "significance": stars_econ,
        "paper_coefficient": "-0.285",
        "paper_r2": "0.090",
    })

print()

# Save results table
df_results = pd.DataFrame(regression_results)
out_path = PROJECT_ROOT / "results" / "tables" / "table2_baseline_replication.csv"
df_results.to_csv(out_path, index=False)
print(f"Saved to {out_path}")

# Generate report
report = f"""# Step 4: Baseline Table 2 Replication Report

## Summary
This replicates Table 2 from Aruoba & Drechsel (2024, Econometrica) using the
original replication package data (`error_data.csv`).

## Replication Results vs. Paper's Table 2

### Panel A: PC1 of All Sentiments
| Horizon | Our Coef (SE) | Our R² | N | Paper Coef | Paper R² | Match? |
|---------|---------------|--------|---|-----------|----------|--------|
"""
for _, row in df_results[df_results["predictor"].str.contains("PC1")].iterrows():
    paper_coef = float(row["paper_coefficient"])
    paper_r2 = float(row["paper_r2"])
    coef_match = "YES" if abs(row["coefficient"] - paper_coef) < 0.1 else "CLOSE" if abs(row["coefficient"] - paper_coef) < 0.3 else "NO"
    r2_match = "YES" if abs(row["r_squared"] - paper_r2) < 0.05 else "CLOSE" if abs(row["r_squared"] - paper_r2) < 0.1 else "NO"
    report += f"| {row['horizon']} | {row['coefficient']:.4f} ({row['se_newy_west']:.4f}) | {row['r_squared']:.3f} | {int(row['n'])} | {paper_coef:.3f} | {paper_r2:.3f} | coef:{coef_match}, R²:{r2_match} |\n"

report += """
### Panel B: Economic Activity Sentiment Only
| Horizon | Our Coef (SE) | Our R² | N | Paper Coef | Paper R² | Match? |
|---------|---------------|--------|---|-----------|----------|--------|
"""
for _, row in df_results[df_results["predictor"].str.contains("Economic")].iterrows():
    paper_coef = float(row["paper_coefficient"])
    paper_r2 = float(row["paper_r2"])
    coef_match = "YES" if abs(row["coefficient"] - paper_coef) < 0.1 else "CLOSE" if abs(row["coefficient"] - paper_coef) < 0.3 else "NO"
    r2_match = "YES" if abs(row["r_squared"] - paper_r2) < 0.05 else "CLOSE" if abs(row["r_squared"] - paper_r2) < 0.1 else "NO"
    report += f"| {row['horizon']} | {row['coefficient']:.4f} ({row['se_newy_west']:.4f}) | {row['r_squared']:.3f} | {int(row['n'])} | {paper_coef:.3f} | {paper_r2:.3f} | coef:{coef_match}, R²:{r2_match} |\n"

report += """
## Notes
- Paper targets: 1Y ahead on PC1: coef=-0.445**, R²=0.248; 1Y ahead on econ activity: coef=-0.285*, R²=0.090
- Standard errors use Newey-West HAC with automatic bandwidth selection
- Differences from paper may reflect sample differences or data vintage
"""

report_path = PROJECT_ROOT / "results" / "reports" / "04_baseline_report.md"
report_path.write_text(report)
print(f"Report saved to {report_path}")
print("Step 4 complete.")
