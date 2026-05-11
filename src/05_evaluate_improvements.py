"""
Step 5: Evaluate LLM-based improvements against the original dictionary-based approach.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Load data ---

# 1. Risk indicators from Step 3
risk_path = PROJECT_ROOT / "data" / "processed" / "risk_indicators.csv"
df_risk = pd.read_csv(risk_path)
df_risk["meeting_date_str"] = df_risk["meeting_date"].astype(str)
print(f"Loaded risk indicators: {len(df_risk)} meetings")

# 2. Baseline replication from Step 4
baseline_path = PROJECT_ROOT / "results" / "tables" / "table2_baseline_replication.csv"
df_baseline = pd.read_csv(baseline_path)
print(f"Loaded baseline results")

# 3. Forecast errors from original data
error_path = PROJECT_ROOT / "data" / "replication_tables_forecast_error" / "error_data.csv"
df_error = pd.read_csv(error_path)

# Convert greenbook_date to meeting_date for merging
# greenbook_date format: e.g. 19820929 (integer YYYYMMDD)
def gb_date_to_meeting(gb_date):
    s = str(int(gb_date))
    if len(s) == 8:
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return None

df_error["meeting_date"] = df_error["greenbook_date"].apply(gb_date_to_meeting)

# Merge risk indicators with forecast errors
df_merged = df_error.merge(df_risk, on="meeting_date", how="inner")
print(f"Merged error + risk: {len(df_merged)} observations")

if len(df_merged) < 10:
    print("WARNING: Insufficient merged data. Creating synthetic merge for testing...")
    # Fallback: merge on greenbook_date integer format
    df_risk["gb_date_int"] = df_risk["meeting_date"].str.replace("-", "").astype(int)
    df_merged = df_error.merge(df_risk, left_on="greenbook_date", right_on="gb_date_int", how="inner")
    print(f"  Merged via greenbook_date integer: {len(df_merged)} observations")

# --- Helper function ---
def newey_west_regression(X, y):
    """Run OLS with Newey-West HAC SEs."""
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_clean = X[mask]
    y_clean = y[mask]
    if len(y_clean) < 20:
        return None, len(y_clean)
    X_with_const = sm.add_constant(X_clean)
    model = sm.OLS(y_clean, X_with_const)
    results = model.fit(cov_type="HAC", cov_kwds={"maxlags": None}, use_t=True)
    return results, len(y_clean)

# --- Analysis ---
HORIZONS = {"Current Q": 0, "1Q ahead": 1, "1Y ahead": 4, "2Y ahead": 8}

improvement_results = []

print("\n=== Improvement #2: LLM Risk Scores ===\n")

# Check which risk columns we have
risk_col = None
for col in ["unemployment_risk_score", "unemployment_risk"]:
    if col in df_merged.columns:
        risk_col = col
        break

if risk_col:
    print(f"Using risk column: {risk_col}")
    # If categorical, encode for regression
    if df_merged[risk_col].dtype == object:
        direction_map = {"upside": 1, "balanced": 0, "downside": -1}
        df_merged["risk_score_encoded"] = df_merged[risk_col].map(direction_map)
        # Apply strength modifier if available
        if "unemployment_strength" in df_merged.columns:
            strength_map = {"high": 1.0, "medium": 0.5, "low": 0.25}
            str_vals = df_merged["unemployment_strength"].map(strength_map).fillna(0.5)
            df_merged["risk_score_encoded"] = df_merged["risk_score_encoded"] * str_vals
        risk_var = "risk_score_encoded"
    else:
        risk_var = risk_col
else:
    print("No unemployment risk column found — using overall_bias as proxy")
    # Fallback: encode overall_bias
    direction_map = {"upside": 1, "balanced": 0, "downside": -1}
    df_merged["risk_score_encoded"] = df_merged["overall_bias"].map(direction_map)
    risk_var = "risk_score_encoded"

for label, h in HORIZONS.items():
    error_col = f"error_unemp_{h}"

    # Analysis A: Risk score alone
    X_risk = df_merged[[risk_var]].values
    y = df_merged[error_col].values

    r_risk, n = newey_west_regression(X_risk, y)
    if r_risk is not None:
        coef = float(r_risk.params[1])
        se = float(r_risk.bse[1])
        r2 = float(r_risk.rsquared)
        pval = float(r_risk.pvalues[1])
        stars = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
        print(f"  {label} on RiskScore: coef={coef:.4f}{stars}, se={se:.4f}, R²={r2:.3f}, N={n}")
        improvement_results.append({
            "horizon": label,
            "method": "LLM risk score (Improvement 2)",
            "r_squared": round(r2, 4),
            "coefficient": round(coef, 4),
            "se_newy_west": round(se, 4),
            "n": n,
            "significance": stars
        })

    # Analysis B: Combined model (PC1 + risk score)
    X_combined = df_merged[["pc1_std", risk_var]].values
    r_comb, n_comb = newey_west_regression(X_combined, y)
    if r_comb is not None:
        r2_comb = float(r_comb.rsquared)
        coef_pc1 = float(r_comb.params[1])
        coef_risk = float(r_comb.params[2])
        print(f"  {label} Combined: PC1 coef={coef_pc1:.4f}, Risk coef={coef_risk:.4f}, R²={r2_comb:.3f}, N={n_comb}")
        improvement_results.append({
            "horizon": label,
            "method": "Combined (PC1 + LLM risk)",
            "r_squared": round(r2_comb, 4),
            "coefficient": round(coef_risk, 4),
            "se_newy_west": round(float(r_comb.bse[2]), 4),
            "n": n_comb,
            "significance": ""
        })

print()

# Add baseline PC1 results for comparison
for _, row in df_baseline[df_baseline["predictor"].str.contains("PC1")].iterrows():
    improvement_results.append({
        "horizon": row["horizon"],
        "method": "Original PC1 (paper Table 2)",
        "r_squared": row["r_squared"],
        "coefficient": row["coefficient"],
        "se_newy_west": row["se_newy_west"],
        "n": row["n"],
        "significance": row["significance"] if "significance" in row else ""
    })

print()

# --- Save results ---
df_improve = pd.DataFrame(improvement_results)
improve_path = PROJECT_ROOT / "results" / "tables" / "improvement_comparison.csv"
df_improve.to_csv(improve_path, index=False)
print(f"Saved improvement results to {improve_path}")

# --- Building clean summary table ---
summary_rows = []
for h in ["Current Q", "1Q ahead", "1Y ahead", "2Y ahead"]:
    for method in ["Original PC1 (paper Table 2)", "LLM risk score (Improvement 2)", "Combined (PC1 + LLM risk)"]:
        match = df_improve[(df_improve["horizon"] == h) & (df_improve["method"] == method)]
        if len(match) > 0:
            row = match.iloc[0]
            summary_rows.append({
                "Horizon": h,
                "Method": method,
                "R_squared": f"{row['r_squared']:.3f}",
                "Coefficient": f"{row['coefficient']:.4f}",
                "N": row["n"]
            })

df_summary = pd.DataFrame(summary_rows)
# Pivot for cleaner presentation
df_pivot = df_summary.pivot(index="Method", columns="Horizon", values="R_squared")
df_pivot.to_csv(PROJECT_ROOT / "results" / "tables" / "final_summary_table.csv")
print(f"Saved final summary to {PROJECT_ROOT / 'results' / 'tables' / 'final_summary_table.csv'}")

# --- Figure 1: Risk score over time ---
fig, ax = plt.subplots(figsize=(12, 5))

df_risk_plot = df_risk.copy()
df_risk_plot["date_dt"] = pd.to_datetime(df_risk_plot["meeting_date"])

# Plot risk score
if risk_col and risk_col in df_risk_plot.columns:
    scores = df_risk_plot[risk_col]
elif "risk_score_encoded" in df_merged.columns:
    df_risk_plot = df_risk_plot.merge(
        df_merged[["meeting_date", "risk_score_encoded"]].drop_duplicates(),
        on="meeting_date", how="left"
    )
    scores = df_risk_plot["risk_score_encoded"]
else:
    direction_map = {"upside": 1, "balanced": 0, "downside": -1}
    scores = df_risk_plot["overall_bias"].map(direction_map).fillna(0)

ax.plot(df_risk_plot["date_dt"], scores, 'b-', alpha=0.7, linewidth=0.8)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# NBER recession shading
recession_periods = [
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
]
for start, end in recession_periods:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color='gray', alpha=0.2)

ax.set_xlabel("Meeting Date")
ax.set_ylabel("Unemployment Risk Score\n(+ = upside risk to unemployment)")
ax.set_title("LLM-Extracted Unemployment Risk Score (1982–2008)")
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.autofmt_xdate()
fig.tight_layout()

fig_path = PROJECT_ROOT / "results" / "figures" / "risk_timeseries.png"
fig.savefig(fig_path, dpi=150)
plt.close(fig)
print(f"Saved {fig_path}")

# --- Figure 2: R² comparison bar chart ---
fig, ax = plt.subplots(figsize=(10, 5))

methods_ordered = ["Original PC1 (paper Table 2)", "LLM risk score (Improvement 2)", "Combined (PC1 + LLM risk)"]
horizons_ordered = ["Current Q", "1Q ahead", "1Y ahead", "2Y ahead"]

x = np.arange(len(horizons_ordered))
width = 0.25
colors = ['#4472C4', '#ED7D31', '#A5A5A5']

for i, method in enumerate(methods_ordered):
    r2_vals = []
    for h in horizons_ordered:
        match = df_improve[(df_improve["horizon"] == h) & (df_improve["method"] == method)]
        if len(match) > 0:
            r2_vals.append(match.iloc[0]["r_squared"])
        else:
            r2_vals.append(0)
    bars = ax.bar(x + i * width, r2_vals, width, label=method, color=colors[i])
    for bar, val in zip(bars, r2_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

ax.set_xlabel("Forecast Horizon")
ax.set_ylabel("R²")
ax.set_title("Forecast Error Predictability: Original vs. LLM-Enhanced")
ax.set_xticks(x + width)
ax.set_xticklabels(horizons_ordered)
ax.legend(loc="upper left")
fig.tight_layout()

fig_path2 = PROJECT_ROOT / "results" / "figures" / "r2_comparison.png"
fig.savefig(fig_path2, dpi=150)
plt.close(fig)
print(f"Saved {fig_path2}")

# --- Generate report ---
# Sample evidence quotes
evidence_quotes_str = "No evidence quotes available"
if "unemployment_evidence" in df_risk.columns:
    evidence_col = df_risk["unemployment_evidence"].dropna()
    evidence_col = evidence_col[evidence_col != "null"]
    if len(evidence_col) > 0:
        sample = evidence_col.sample(min(5, len(evidence_col)), random_state=1)
        evidence_quotes_str = "\n".join(f"- \"{q}\"" for q in sample)

# Find best improvement
baseline_r2 = {}
for _, row in df_improve[df_improve["method"] == "Original PC1 (paper Table 2)"].iterrows():
    baseline_r2[row["horizon"]] = row["r_squared"]

improvement_summary = ""
for _, row in df_improve[df_improve["method"] == "LLM risk score (Improvement 2)"].iterrows():
    h = row["horizon"]
    baseline = baseline_r2.get(h, 0)
    delta = row["r_squared"] - baseline
    direction = "improvement" if delta > 0 else "decline"
    improvement_summary += f"- **{h}**: LLM risk score R²={row['r_squared']:.3f} vs. baseline R²={baseline:.3f} ({delta:+.3f} {direction})\n"

report = f"""# Step 5: Evaluation Report — LLM Improvements vs. Baseline

## Analysis A: Improvement #2 — LLM Risk Score Validation

{improvement_summary}

### Key Results
The regression of unemployment forecast error on LLM-extracted risk scores
tests whether the LLM can identify Greenbook modal-vs-mean risk asymmetry
that the dictionary-based approach misses.

### Sample Evidence Quotes
{evidence_quotes_str}

## Analysis B: Combined Model
Adding LLM risk scores to the original PC1 sentiment indicator shows whether
the LLM captures complementary information about forecast risk.

## Analysis C: Improvement #3 — LLM Concept Identification
Results in `results/tables/concept_comparison.csv` from Step 2.

## Economic Interpretation
- The LLM risk score captures directional risk asymmetry around Greenbook modal forecasts
- Positive values indicate upside risk to unemployment (staff worried about higher unemployment)
- Correlations with NBER recession dates are reported in Step 3's report

## Limitations
- Sample size limitations (276 meetings, 1982–2008)
- LLM extraction quality depends on prompt design and text quality
- Risk score encoding scheme (categorical → continuous) involves judgment calls
- DeepSeek model may have different biases than human readers

## Suggested Talking Points
1. The LLM-based approach to risk extraction is fully automated and reproducible
2. Evidence quotes make results auditable — unlike the dictionary approach
3. The categorical rating scheme (high/medium/low) is more interpretable than continuous scores
4. Combined models show whether LLM indicators add value beyond traditional sentiment
"""

report_path = PROJECT_ROOT / "results" / "reports" / "05_evaluation_report.md"
report_path.write_text(report)
print(f"Report saved to {report_path}")
print("Step 5 complete.")
