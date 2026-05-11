"""
Step 6: Improvement Evaluation — compare all contributions vs. baseline.
Matches CLAUDE.md spec:
  Baseline:   PC1 of original 296 LM-dictionary sentiments | Ridge
  C1-LayerA:  PC1 of LLM-scored sentiments (overlap w/ original 296) | Ridge
  C1-LayerB:  PC1 of LLM-scored sentiments (all 595 LLM concepts) | Ridge
  C2-Risk:    unemployment_risk_score alone | OLS
  C2-Combined: unemployment_risk_score + baseline PC1 | OLS + F-test
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- 1. Load error data + Greenbook bridge (maps greenbook_date → meeting_date) ---
error_path = PROJECT_ROOT / "data" / "replication_tables_forecast_error" / "error_data.csv"
df_error = pd.read_csv(error_path)
print(f"Error data: {len(df_error)} rows")

# Bridge: Greenbook regression input has both greenbook_date and meeting_date
bridge_path = PROJECT_ROOT / "data" / "create_sentiments" / "Input_files" / "Greenbook_Regressioninput_Original.csv"
df_bridge = pd.read_csv(bridge_path)
df_bridge = df_bridge[["meeting_date", "greenbook_date"]].drop_duplicates()
# Convert meeting_date from "1982_10_05" to "1982-10-05"
df_bridge["meeting_date"] = df_bridge["meeting_date"].str.replace("_", "-")
print(f"Bridge: {len(df_bridge)} greenbook→meeting mappings")

# Merge to get meeting_date for each error observation
df_error = df_error.merge(df_bridge, on="greenbook_date", how="left")
n_mapped = df_error["meeting_date"].notna().sum()
print(f"Error rows with meeting_date: {n_mapped}/{len(df_error)}")

# --- 2. Load LLM sentiment data (595 concepts) ---
sent_path = PROJECT_ROOT / "data" / "processed" / "sentiment_llm.csv"
df_sent = pd.read_csv(sent_path)
sent_cols_all = [c for c in df_sent.columns if c != "meeting_date"]
print(f"LLM sentiment: {len(df_sent)} meetings, {len(sent_cols_all)} concepts")

# Identify overlap with original 296 concepts for Layer 1A
orig_path = PROJECT_ROOT / "data" / "create_sentiments" / "Input_files" / "Concepts_LongList.csv"
df_orig_concepts = pd.read_csv(orig_path)
orig_set = set(df_orig_concepts.iloc[:, 0].str.lower().str.strip())
overlap_cols = [c for c in sent_cols_all if c.lower().strip() in orig_set]
print(f"Layer 1A (overlap): {len(overlap_cols)} concepts (of {len(orig_set)} original)")
print(f"Layer 1B (all LLM): {len(sent_cols_all)} concepts")

# Compute PC1 for both layers
def compute_pc1(df, cols):
    X = df[cols].values
    X = np.nan_to_num(X, 0)
    X_scaled = StandardScaler().fit_transform(X)
    pc1 = PCA(n_components=1).fit_transform(X_scaled).flatten()
    return pc1

df_sent["pc1_layerA"] = compute_pc1(df_sent, overlap_cols)
df_sent["pc1_layerB"] = compute_pc1(df_sent, sent_cols_all)
print(f"PC1 explained variance: LayerA={PCA(n_components=1).fit(StandardScaler().fit_transform(np.nan_to_num(df_sent[overlap_cols].values,0))).explained_variance_ratio_[0]:.3f}, LayerB={PCA(n_components=1).fit(StandardScaler().fit_transform(np.nan_to_num(df_sent[sent_cols_all].values,0))).explained_variance_ratio_[0]:.3f}")

# Merge sentiment PC1s into error data
df = df_error.merge(df_sent[["meeting_date", "pc1_layerA", "pc1_layerB"]],
                     on="meeting_date", how="left")

# --- 3. Load risk indicators ---
risk_path = PROJECT_ROOT / "data" / "processed" / "risk_indicators.csv"
df_risk = pd.read_csv(risk_path)
print(f"Risk data: {len(df_risk)} meetings")

# Encode risk score
direction_map = {"upside": 1, "balanced": 0, "downside": -1, "not_mentioned": None}
strength_map = {"high": 1.0, "medium": 0.5, "low": 0.25}

df_risk["risk_score"] = df_risk["unemployment_risk"].map(direction_map).astype(float)
if "unemployment_strength" in df_risk.columns:
    str_vals = df_risk["unemployment_strength"].map(strength_map).fillna(0.0)
    df_risk["risk_score"] = df_risk["risk_score"] * str_vals

df = df.merge(df_risk[["meeting_date", "risk_score",
                        "unemployment_risk", "unemployment_strength",
                        "unemployment_evidence"]],
              on="meeting_date", how="left", suffixes=("", "_risk"))

print(f"Final merged: {len(df)} observations")

# --- 4. Ridge regression helper (for PC1-based specs) ---
def ridge_regress(X, y):
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_clean = X[mask]
    y_clean = y[mask]
    if len(y_clean) < 20:
        return {"r2": None, "coef": None, "se": None, "n": len(y_clean)}
    # Standardize for Ridge
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 20))
    ridge.fit(X_scaled, y_clean)
    r2 = ridge.score(X_scaled, y_clean)
    # Get coefficient on original scale
    coef = ridge.coef_[0] / scaler.scale_[0] if X_clean.shape[1] == 1 else ridge.coef_
    return {"r2": round(float(r2), 4), "coef": round(float(coef) if np.isscalar(coef) else coef[0], 4),
            "se": None, "n": len(y_clean), "alpha": round(float(ridge.alpha_), 4)}

# --- 5. OLS + Newey-West helper (for risk specs) ---
def ols_regress(X, y):
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_clean = X[mask]
    y_clean = y[mask]
    if len(y_clean) < 20:
        return {"r2": None, "coef": None, "se": None, "n": len(y_clean), "pval": None}
    X_c = sm.add_constant(X_clean)
    r = sm.OLS(y_clean, X_c).fit(cov_type="HAC", cov_kwds={"maxlags": 4}, use_t=True)
    idx = 1  # coefficient of interest
    return {
        "r2": round(float(r.rsquared), 4),
        "coef": round(float(r.params[idx]), 4),
        "se": round(float(r.bse[idx]), 4),
        "n": len(y_clean),
        "pval": float(r.pvalues[idx])
    }

# F-test for incremental R² (C2-Combined vs Baseline)
def f_test_incremental(X_small, X_large, y):
    """F-test: does adding risk_score to baseline PC1 improve fit?"""
    mask = ~(np.isnan(X_small).any(axis=1) | np.isnan(X_large).any(axis=1) | np.isnan(y))
    X_s = sm.add_constant(X_small[mask])
    X_l = sm.add_constant(X_large[mask])
    y_c = y[mask]
    r_small = sm.OLS(y_c, X_s).fit()
    r_large = sm.OLS(y_c, X_l).fit()
    rss_s = r_small.ssr
    rss_l = r_large.ssr
    df_s = r_small.df_resid
    df_l = r_large.df_resid
    if df_l < df_s and rss_s > 0:
        f_stat = ((rss_s - rss_l) / (df_s - df_l)) / (rss_l / df_l)
        # Use statsmodels to get p-value
        from scipy.stats import f as f_dist
        p_val = 1 - f_dist.cdf(f_stat, df_s - df_l, df_l)
        return round(float(f_stat), 4), round(float(r_large.rsquared - r_small.rsquared), 4), round(float(p_val), 4)
    return None, None, None

# --- 6. Run comparisons ---
HORIZONS = {"Current Q": 0, "1Q ahead": 1, "1Y ahead": 4, "2Y ahead": 8}
all_results = []

for label, h in HORIZONS.items():
    err_col = f"error_unemp_{h}"
    y = df[err_col].values

    # Spec 1: Baseline (original PC1 from paper)
    X_base = df[["pc1_std"]].values
    r = ridge_regress(X_base, y)
    if r["r2"] is not None:
        all_results.append({"horizon": label, "spec": "Baseline (orig PC1)", **r})

    # Spec 2: C1-LayerA (LLM scoring, overlap concepts)
    if "pc1_layerA" in df.columns:
        X_a = df[["pc1_layerA"]].values
        r = ridge_regress(X_a, y)
        if r["r2"] is not None:
            all_results.append({"horizon": label, "spec": "C1-LayerA (LLM score, 217 concepts)", **r})

    # Spec 3: C1-LayerB (LLM scoring, all 595 concepts)
    if "pc1_layerB" in df.columns:
        X_b = df[["pc1_layerB"]].values
        r = ridge_regress(X_b, y)
        if r["r2"] is not None:
            all_results.append({"horizon": label, "spec": "C1-LayerB (LLM score, 595 concepts)", **r})

    # Spec 4: C2-Risk (risk score alone)
    if "risk_score" in df.columns:
        X_risk = df[["risk_score"]].values
        r = ols_regress(X_risk, y)
        if r["r2"] is not None:
            all_results.append({"horizon": label, "spec": "C2-Risk (unemp risk score)", **r})

    # Spec 5: C2-Combined (baseline PC1 + risk score, OLS + F-test)
    if "risk_score" in df.columns:
        X_comb = df[["pc1_std", "risk_score"]].values
        r = ols_regress(X_comb, y)
        if r["r2"] is not None:
            f_stat, delta_r2, f_pval = f_test_incremental(
                df[["pc1_std"]].values, X_comb, y)
            r["f_stat"] = f_stat
            r["delta_r2"] = delta_r2
            r["f_pval"] = f_pval
            all_results.append({"horizon": label, "spec": "C2-Combined (PC1 + risk)", **r})

# --- 7. Save results ---
df_improve = pd.DataFrame(all_results)
print(f"\nResults: {len(df_improve)} rows")

# Final summary table (pivoted R²)
df_pivot = df_improve.pivot_table(
    index="spec", columns="horizon", values="r2", aggfunc="first"
)
horizon_order = ["Current Q", "1Q ahead", "1Y ahead", "2Y ahead"]
df_pivot = df_pivot[horizon_order]

summary_path = PROJECT_ROOT / "results" / "tables" / "final_summary_table.csv"
df_pivot.to_csv(summary_path)
print(f"\nFinal Summary Table (R²):")
print(df_pivot.to_string())
print(f"\nSaved: {summary_path}")

# Also save full detailed results
detail_path = PROJECT_ROOT / "results" / "tables" / "improvement_comparison.csv"
df_improve.to_csv(detail_path, index=False)
print(f"Saved: {detail_path}")

# --- 8. Figures ---
# Figure 1: Risk score timeseries
fig, ax = plt.subplots(figsize=(12, 5))
df_risk_plot = df_risk.copy()
df_risk_plot["date_dt"] = pd.to_datetime(df_risk_plot["meeting_date"])
ax.plot(df_risk_plot["date_dt"], df_risk_plot["risk_score"], 'b-', alpha=0.7, linewidth=0.8)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
recession_periods = [
    ("1990-07-01", "1991-03-01"), ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
]
for start, end in recession_periods:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color='gray', alpha=0.2)
ax.set_xlabel("Meeting Date")
ax.set_ylabel("Unemployment Risk Score\n(+ = upside risk to unemployment)")
ax.set_title("LLM-Extracted Unemployment Risk Score (1982–2008)")
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.autofmt_xdate(); fig.tight_layout()
fig_path1 = PROJECT_ROOT / "results" / "figures" / "risk_timeseries.png"
fig.savefig(fig_path1, dpi=150)
plt.close(fig)
print(f"Saved: {fig_path1}")

# Figure 2: R² comparison bar chart
fig, ax = plt.subplots(figsize=(12, 5))
specs_ordered = [s for s in df_pivot.index]
x = np.arange(len(horizon_order))
width = 0.8 / max(len(specs_ordered), 1)
colors = plt.cm.Set2(np.linspace(0, 1, len(specs_ordered)))
for i, spec in enumerate(specs_ordered):
    r2s = []
    for h in horizon_order:
        match = df_improve[(df_improve["spec"] == spec) & (df_improve["horizon"] == h)]
        r2s.append(match.iloc[0]["r2"] if len(match) > 0 else 0)
    bars = ax.bar(x + i * width, r2s, width, label=spec, color=colors[i])
    for bar, val in zip(bars, r2s):
        if val and val > 0.001:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)
ax.set_xlabel("Forecast Horizon"); ax.set_ylabel("R²")
ax.set_title("Forecast Error Predictability Across Methods")
ax.set_xticks(x + width * (len(specs_ordered) - 1) / 2)
ax.set_xticklabels(horizon_order)
ax.legend(loc="upper left", fontsize=7, ncol=2)
fig.tight_layout()
fig_path2 = PROJECT_ROOT / "results" / "figures" / "r2_comparison.png"
fig.savefig(fig_path2, dpi=150)
plt.close(fig)
print(f"Saved: {fig_path2}")

# --- 9. Report ---
# Check baseline match
baseline_1y = df_improve[(df_improve["horizon"] == "1Y ahead") &
                          (df_improve["spec"] == "Baseline (orig PC1)")]
baseline_1y_r2 = baseline_1y.iloc[0]["r2"] if len(baseline_1y) > 0 else "N/A"
paper_r2 = 0.248
print(f"Baseline 1Y R² check: ours={baseline_1y_r2}, paper={paper_r2}")

# Sample evidence quotes
evidence_sample = df[df["unemployment_evidence"].notna() &
                      (df["unemployment_evidence"] != "") &
                      (df["unemployment_evidence"] != "null")]
sample_quotes = ""
if len(evidence_sample) > 0:
    sample = evidence_sample.sample(min(5, len(evidence_sample)), random_state=42)
    sample_quotes = "\n".join(
        f"- **{row['meeting_date']}** ({row['unemployment_risk']}, {row['unemployment_strength']}): "
        f"\"{row['unemployment_evidence']}\""
        for _, row in sample.iterrows()
    )

# F-test significance
f_test_lines = ""
for _, row in df_improve[df_improve["spec"] == "C2-Combined (PC1 + risk)"].iterrows():
    stars = "***" if (row.get("f_pval") or 1) < 0.01 else ("**" if (row.get("f_pval") or 1) < 0.05 else ("*" if (row.get("f_pval") or 1) < 0.1 else ""))
    f_test_lines += f"- **{row['horizon']}**: ΔR²={row.get('delta_r2', 'N/A')}, F={row.get('f_stat', 'N/A')}{stars}, p={row.get('f_pval', 'N/A')}\n"

# Find best method per horizon
best_summary = ""
for h in horizon_order:
    sub = df_improve[df_improve["horizon"] == h]
    if len(sub) > 0:
        best = sub.loc[sub["r2"].idxmax()]
        best_summary += f"- **{h}**: Best = {best['spec']} (R²={best['r2']:.3f})\n"

report = f"""# Step 6: Improvement Evaluation Report

## Final Comparison Table (R²)
{df_pivot.to_string()}

## Best Method by Horizon
{best_summary}

## Baseline Verification
- Our baseline 1Y R²: **{baseline_1y_r2}** (paper: {paper_r2})

## Contribution 1: LLM NLP Features
### Layer 1A — LLM Scoring on Original Concepts
Uses LLM sentiment scoring on the {len(overlap_cols)} concepts that overlap with the original 296.
Compares directly to baseline (same concepts, different scoring method).

### Layer 1B — LLM Scoring on LLM-Identified Concepts
Uses LLM sentiment scoring on all {len(sent_cols_all)} LLM-identified concepts.
Tests whether LLM concept identification adds value beyond LLM scoring alone.

## Contribution 2: Risk Asymmetry (Direct Test of Modal-Forecast Mechanism)
### F-test for Incremental R²
{f_test_lines}

## Sample Unemployment Risk Evidence Quotes
{sample_quotes}

## Limitations
- Layer 1A only covers {len(overlap_cols)} of the ~296 original concepts (LLM free-naming + fuzzy matching missed some)
- LLM sentiment matrix is sparse ({len(sent_cols_all)} concepts but only ~5-40 non-zero per meeting)
- Risk score encoding (categorical→continuous) involves judgment calls
- Sample limited to 1982-2008 ({len(df)} observations with pc1_std)
"""

report_path = PROJECT_ROOT / "results" / "reports" / "06_evaluation_report.md"
report_path.write_text(report)
print(f"Report saved: {report_path}")
print("Step 6 complete.")
