"""
LLM List Shuffle Test — Devesh Thavalam Recommendation #3.
Tests whether shuffling concept labels changes Ridge fit more for the LLM-derived
595-concept list (balanced composition) than for the human 296-concept list.

0 API calls. Output: CSV tables + PNG figures.
"""
import pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "results" / "figures"
TBL_DIR = PROJECT_ROOT / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# 0. Load & Merge Data
# ===========================================================================

# FFR changes
df_ffr = pd.read_csv(PROJECT_ROOT / "data/create_sentiments/Input_files/FFR_Target.csv")
df_ffr["meeting_date"] = df_ffr["Meeting_Date"].apply(
    lambda s: (datetime(1899, 12, 30) + timedelta(days=int(s))).strftime("%Y-%m-%d"))

# Greenbook numerical forecasts
df_reg = pd.read_csv(PROJECT_ROOT / "data/create_sentiments/Input_files/Greenbook_Regressioninput_Original.csv")
df_reg["meeting_date"] = df_reg["meeting_date"].str.replace("_", "-")
meta_cols = ["Unnamed: 0", "meeting_date", "greenbook_date", "year_gb", "quarter_gb"]
fc_cols = [c for c in df_reg.columns if c not in meta_cols]

# Dictionary sentiment (296 concepts, dense)
df_ds = pd.read_csv(PROJECT_ROOT / "data/replication_files/Sentiment_Final/total_sentiment_file_std10.csv")
idc = df_ds.columns[0]
df_ds["meeting_date"] = df_ds[idc].astype(str).str.replace("_", "-")
dc_cols = [c for c in df_ds.columns if c not in (idc, "meeting_date")]

# LLM sentiment (595 concepts, sparse) — rename with prefix to avoid merge conflicts
df_llm = pd.read_csv(PROJECT_ROOT / "data/processed/sentiment_llm.csv")
llm_cols_orig = [c for c in df_llm.columns if c != "meeting_date"]
llm_rename = {c: f"llm_{c}" for c in llm_cols_orig}
df_llm = df_llm.rename(columns=llm_rename)
llm_cols = list(llm_rename.values())

# Merge — LLM columns now have llm_ prefix, no conflicts with dict columns
df = df_ffr[["meeting_date", "FFR_change"]].merge(
    df_reg[["meeting_date"] + fc_cols], on="meeting_date", how="inner").merge(
    df_ds[["meeting_date"] + dc_cols], on="meeting_date", how="inner").merge(
    df_llm[["meeting_date"] + llm_cols], on="meeting_date", how="inner")
df = df[(df["meeting_date"] >= "1982-10-01") & (df["meeting_date"] <= "2008-12-31")].copy()

# Build matrices
y = np.array(df["FFR_change"].values, dtype=float)
X_fc = np.nan_to_num(np.array(df[fc_cols].values, dtype=float), nan=0.0)

# Dict sentiment (dense — zeros are genuine sentiment scores)
X_dict = np.nan_to_num(np.array(df[dc_cols].values, dtype=float), nan=0.0)

# LLM sentiment (sparse — zeros mean concept not mentioned)
X_llm_raw = np.nan_to_num(np.array(df[llm_cols].values, dtype=float), nan=0.0)

# Compute PC1s
scaler = StandardScaler()
X_dict_scaled = scaler.fit_transform(X_dict)
dict_pc1 = PCA(n_components=1).fit_transform(X_dict_scaled).flatten()
dict_evr = PCA(n_components=1).fit(X_dict_scaled).explained_variance_ratio_[0]

X_llm_scaled = scaler.fit_transform(X_llm_raw)
llm_pc1 = PCA(n_components=1).fit_transform(X_llm_scaled).flatten()
llm_evr = PCA(n_components=1).fit(X_llm_scaled).explained_variance_ratio_[0]

print(f"Data: {len(df)} meetings, {len(fc_cols)} forecasts, {len(dc_cols)} dict concepts, {len(llm_cols)} LLM concepts")
print(f"Dict PC1 variance explained: {dict_evr:.1%}")
print(f"LLM PC1 variance explained: {llm_evr:.1%}")
print()

# ===========================================================================
# Ridge helper
# ===========================================================================
def ridge_r2(X, y):
    X = np.array(X, dtype=float, copy=True)
    y = np.array(y, dtype=float, copy=True)
    X = np.nan_to_num(X, nan=0.0)
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_c, y_c = X[mask], y[mask]
    if len(y_c) < 20:
        return None, len(y_c)
    X_s = StandardScaler().fit_transform(X_c)
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 30), cv=5)
    ridge.fit(X_s, y_c)
    return round(float(ridge.score(X_s, y_c)), 4), len(y_c)

# ===========================================================================
# CHECK 4a: Composition Comparison — Dict vs LLM Lists
# ===========================================================================
print("=" * 60)
print("CHECK 4a: Sentiment Composition — Dict vs LLM")
print("=" * 60)

# Per-meeting % positive
dict_pos_fracs = np.array([np.sum(row > 0) / max(np.sum(row != 0), 1) for row in X_dict])
llm_pos_fracs = np.array([np.sum(row > 0) / max(np.sum(row != 0), 1)
                          if np.sum(row != 0) > 0 else np.nan for row in X_llm_raw])

# Per-meeting non-zero count
dict_nz = np.array([np.sum(row != 0) for row in X_dict])
llm_nz = np.array([np.sum(row != 0) for row in X_llm_raw])

# By-concept average
dict_concept_avg = np.mean(X_dict, axis=0)
llm_concept_mask = np.sum(X_llm_raw != 0, axis=0) > 0
llm_concept_avg = np.array([
    np.mean(X_llm_raw[:, j][X_llm_raw[:, j] != 0])
    if np.sum(X_llm_raw[:, j] != 0) > 0 else 0
    for j in range(X_llm_raw.shape[1])
])

comp_rows = [
    {"Metric": "Concepts",          "Dict (296)": "296",     "LLM (595)": "595"},
    {"Metric": "Density (non-zero %)", "Dict (296)": f"{100 - np.mean(dict_nz==0)*100:.1f}%",
                                       "LLM (595)": f"{np.mean(llm_nz>0)*100:.1f}%"},
    {"Metric": "Non-zeros per meeting (mean)", "Dict (296)": f"{np.mean(dict_nz):.1f}",
                                                "LLM (595)": f"{np.mean(llm_nz):.1f}"},
    {"Metric": "% Positive per meeting (mean)", "Dict (296)": f"{np.mean(dict_pos_fracs):.1%}",
                                                 "LLM (595)": f"{np.nanmean(llm_pos_fracs):.1%}"},
    {"Metric": "% Positive per meeting (std)",  "Dict (296)": f"{np.std(dict_pos_fracs):.4f}",
                                                 "LLM (595)": f"{np.nanstd(llm_pos_fracs):.4f}"},
    {"Metric": "% Positive per meeting (min)",  "Dict (296)": f"{np.min(dict_pos_fracs):.1%}",
                                                 "LLM (595)": f"{np.nanmin(llm_pos_fracs):.1%}"},
    {"Metric": "% Positive per meeting (max)",  "Dict (296)": f"{np.max(dict_pos_fracs):.1%}",
                                                 "LLM (595)": f"{np.nanmax(llm_pos_fracs):.1%}"},
    {"Metric": "Concepts with mean > 0", "Dict (296)": f"{np.sum(dict_concept_avg > 0)} ({np.sum(dict_concept_avg > 0)/len(dict_concept_avg):.1%})",
                                          "LLM (595)": f"{np.sum(llm_concept_avg > 0)} ({np.sum(llm_concept_avg > 0)/len(llm_concept_avg):.1%})"},
    {"Metric": "Concepts with mean < 0", "Dict (296)": f"{np.sum(dict_concept_avg < 0)} ({np.sum(dict_concept_avg < 0)/len(dict_concept_avg):.1%})",
                                          "LLM (595)": f"{np.sum(llm_concept_avg < 0)} ({np.sum(llm_concept_avg < 0)/len(llm_concept_avg):.1%})"},
]
df_comp = pd.DataFrame(comp_rows)
df_comp.to_csv(TBL_DIR / "check4a_composition.csv", index=False)
for r in comp_rows:
    print(f"  {r['Metric']:<35} {r['Dict (296)']:<20} {r['LLM (595)']:<20}")
print(f"  Saved: {TBL_DIR / 'check4a_composition.csv'}\n")

# Figure: per-meeting % positive distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax1, ax2 = axes

# Panel A: Dict
ax1.hist(dict_pos_fracs, bins=25, color="#4472C4", alpha=0.7, edgecolor="white")
ax1.axvline(np.mean(dict_pos_fracs), color="#C00000", linewidth=2, linestyle="--",
            label=f"Mean = {np.mean(dict_pos_fracs):.1%}")
ax1.set_xlabel("% Positive per Meeting", fontsize=10)
ax1.set_ylabel("Frequency", fontsize=10)
ax1.set_title(f"Dict List (296 concepts, dense)", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(axis="y", alpha=0.3)

# Panel B: LLM
valid_llm = llm_pos_fracs[~np.isnan(llm_pos_fracs)]
ax2.hist(valid_llm, bins=25, color="#ED7D31", alpha=0.7, edgecolor="white")
ax2.axvline(np.nanmean(llm_pos_fracs), color="#C00000", linewidth=2, linestyle="--",
            label=f"Mean = {np.nanmean(llm_pos_fracs):.1%}")
ax2.set_xlabel("% Positive per Meeting", fontsize=10)
ax2.set_title(f"LLM List (595 concepts, sparse)", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)

fig.suptitle("Within-Meeting Sentiment Composition: Dict vs LLM", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "check4a_composition.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Figure saved: {FIG_DIR / 'check4a_composition.png'}\n")

# ===========================================================================
# CHECK 4b: Baseline First-Stage — Dict PC1 vs LLM PC1
# ===========================================================================
print("=" * 60)
print("CHECK 4b: First-Stage R² — Dict PC1 vs LLM PC1")
print("=" * 60)

r2_fc, _ = ridge_r2(X_fc, y)
r2_fc_dict_pc1, _ = ridge_r2(np.column_stack([X_fc, dict_pc1]), y)
r2_fc_llm_pc1, _ = ridge_r2(np.column_stack([X_fc, llm_pc1]), y)

print(f"  Forecasts only:               R² = {r2_fc:.4f}")
print(f"  Forecasts + Dict PC1:         R² = {r2_fc_dict_pc1:.4f}  (Δ = {r2_fc_dict_pc1 - r2_fc:+.4f})")
print(f"  Forecasts + LLM PC1:          R² = {r2_fc_llm_pc1:.4f}  (Δ = {r2_fc_llm_pc1 - r2_fc:+.4f})")

# Also test: LLM all sentiments directly (no PC1) — sparse but informative
X_fc_llm_all = np.column_stack([X_fc, X_llm_raw])
r2_fc_llm_all, _ = ridge_r2(X_fc_llm_all, y)
print(f"  Forecasts + LLM all (595):    R² = {r2_fc_llm_all:.4f}  (Δ = {r2_fc_llm_all - r2_fc:+.4f})")

pd.DataFrame([
    {"Specification": "Forecasts only",           "R²": r2_fc,       "Δ R²": 0.0},
    {"Specification": "Forecasts + Dict PC1",     "R²": r2_fc_dict_pc1, "Δ R²": round(r2_fc_dict_pc1 - r2_fc, 4)},
    {"Specification": "Forecasts + LLM PC1",      "R²": r2_fc_llm_pc1,  "Δ R²": round(r2_fc_llm_pc1 - r2_fc, 4)},
    {"Specification": "Forecasts + LLM all (595)","R²": r2_fc_llm_all,  "Δ R²": round(r2_fc_llm_all - r2_fc, 4)},
]).to_csv(TBL_DIR / "check4b_baseline.csv", index=False)
print(f"\n  Saved: {TBL_DIR / 'check4b_baseline.csv'}\n")

# ===========================================================================
# CHECK 4c: Shuffle Test — LLM List (PC1-Based, Same Method as Dict)
# ===========================================================================
print("=" * 60)
print("CHECK 4c: LLM Shuffle Test (PC1-Based)")
print("=" * 60)

N_SHUFFLE = 100
np.random.seed(42)

# Pre-compute dict shuffle results for comparison
dict_shuffle_r2s = []
print("  Dict shuffle: running...", end=" ", flush=True)
for seed in range(N_SHUFFLE):
    rng = np.random.RandomState(seed)
    X_shuf = np.zeros_like(X_dict)
    for i in range(X_dict.shape[0]):
        X_shuf[i, :] = rng.permutation(X_dict[i, :])
    X_s = StandardScaler().fit_transform(X_shuf)
    pc1_s = PCA(n_components=1).fit_transform(X_s).flatten()
    r2_s, _ = ridge_r2(np.column_stack([X_fc, pc1_s]), y)
    dict_shuffle_r2s.append(r2_s)
dict_shuffle_r2s = np.array(dict_shuffle_r2s)
print(f"done. Mean R² = {dict_shuffle_r2s.mean():.4f}, std = {dict_shuffle_r2s.std():.4f}, "
      f"range = [{dict_shuffle_r2s.min():.4f}, {dict_shuffle_r2s.max():.4f}]")

# LLM shuffle
llm_shuffle_r2s = []
n_meetings_skipped = []
print("  LLM shuffle: running...", end=" ", flush=True)
for seed in range(N_SHUFFLE):
    rng = np.random.RandomState(seed)
    X_shuf = np.zeros_like(X_llm_raw)
    skipped = 0
    for i in range(X_llm_raw.shape[0]):
        row = X_llm_raw[i, :].copy()
        if np.sum(row != 0) <= 1:
            # Meeting with 0 or 1 non-zero scores — shuffle is identity or trivial
            X_shuf[i, :] = row
            if np.sum(row != 0) == 0:
                skipped += 1
        else:
            X_shuf[i, :] = rng.permutation(row)
    n_meetings_skipped.append(skipped)
    X_s = StandardScaler().fit_transform(X_shuf)
    pc1_s = PCA(n_components=1).fit_transform(X_s).flatten()
    r2_s, _ = ridge_r2(np.column_stack([X_fc, pc1_s]), y)
    llm_shuffle_r2s.append(r2_s)
llm_shuffle_r2s = np.array(llm_shuffle_r2s)
print(f"done. Mean R² = {llm_shuffle_r2s.mean():.4f}, std = {llm_shuffle_r2s.std():.4f}, "
      f"range = [{llm_shuffle_r2s.min():.4f}, {llm_shuffle_r2s.max():.4f}]")
print(f"  Meetings with 0 non-zeros (per shuffle): mean={np.mean(n_meetings_skipped):.1f}")

# Statistics
dict_p_value = (np.sum(dict_shuffle_r2s >= r2_fc_dict_pc1) + 1) / (N_SHUFFLE + 1)
llm_p_value = (np.sum(llm_shuffle_r2s >= r2_fc_llm_pc1) + 1) / (N_SHUFFLE + 1)

print(f"\n  Dict: Actual R² = {r2_fc_dict_pc1:.4f}, Shuffle mean = {dict_shuffle_r2s.mean():.4f} ± {dict_shuffle_r2s.std():.4f}, p = {dict_p_value:.3f}")
print(f"  LLM:  Actual R² = {r2_fc_llm_pc1:.4f}, Shuffle mean = {llm_shuffle_r2s.mean():.4f} ± {llm_shuffle_r2s.std():.4f}, p = {llm_p_value:.3f}")

# Save
pd.DataFrame({
    "list": ["dict"] * N_SHUFFLE + ["llm"] * N_SHUFFLE,
    "shuffle_r2": np.concatenate([dict_shuffle_r2s, llm_shuffle_r2s]),
    "actual_r2": [r2_fc_dict_pc1] * N_SHUFFLE + [r2_fc_llm_pc1] * N_SHUFFLE,
}).to_csv(TBL_DIR / "check4c_shuffle_pc1.csv", index=False)

summary_rows = [
    {"List": "Dict (296)", "Actual R²": r2_fc_dict_pc1,
     "Shuffle Mean R²": round(float(dict_shuffle_r2s.mean()), 4),
     "Shuffle Std": round(float(dict_shuffle_r2s.std()), 4),
     "Shuffle Min": round(float(dict_shuffle_r2s.min()), 4),
     "Shuffle Max": round(float(dict_shuffle_r2s.max()), 4),
     "p-value": dict_p_value},
    {"List": "LLM (595)", "Actual R²": r2_fc_llm_pc1,
     "Shuffle Mean R²": round(float(llm_shuffle_r2s.mean()), 4),
     "Shuffle Std": round(float(llm_shuffle_r2s.std()), 4),
     "Shuffle Min": round(float(llm_shuffle_r2s.min()), 4),
     "Shuffle Max": round(float(llm_shuffle_r2s.max()), 4),
     "p-value": llm_p_value},
]
pd.DataFrame(summary_rows).to_csv(TBL_DIR / "check4c_shuffle_summary.csv", index=False)
print(f"\n  Saved: {TBL_DIR / 'check4c_shuffle_pc1.csv'}")
print(f"  Saved: {TBL_DIR / 'check4c_shuffle_summary.csv'}\n")

# Figure: side-by-side shuffle distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Dict
ax = axes[0]
ax.hist(dict_shuffle_r2s, bins=20, color="#4472C4", alpha=0.7, edgecolor="white",
        label=f"Shuffled PC1 (n={N_SHUFFLE})")
ax.axvline(r2_fc_dict_pc1, color="#C00000", linewidth=2.5, linestyle="--",
           label=f"Actual R² = {r2_fc_dict_pc1:.4f}")
ax.set_xlabel("R²", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.set_title(f"Dict List (296 concepts)\np = {dict_p_value:.3f}", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# LLM
ax = axes[1]
ax.hist(llm_shuffle_r2s, bins=20, color="#ED7D31", alpha=0.7, edgecolor="white",
        label=f"Shuffled PC1 (n={N_SHUFFLE})")
ax.axvline(r2_fc_llm_pc1, color="#C00000", linewidth=2.5, linestyle="--",
           label=f"Actual R² = {r2_fc_llm_pc1:.4f}")
ax.set_xlabel("R²", fontsize=11)
ax.set_title(f"LLM List (595 concepts)\np = {llm_p_value:.3f}", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

fig.suptitle("Label Shuffle Test: PC1 from Permuted Sentiment Columns", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "check4c_shuffle_pc1.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Figure saved: {FIG_DIR / 'check4c_shuffle_pc1.png'}\n")

# ===========================================================================
# CHECK 4d: Shuffle Test — Full Sentiment Matrix (No PC1, All Columns)
# ===========================================================================
print("=" * 60)
print("CHECK 4d: LLM Shuffle Test — Full Sentiment Columns (No PC1)")
print("=" * 60)

# Baseline: forecasts + all LLM columns
r2_fc_llm, _ = ridge_r2(np.column_stack([X_fc, X_llm_raw]), y)
print(f"  Baseline: Fc + LLM all (595) R² = {r2_fc_llm:.4f}")

# Shuffle LLM columns within each meeting, re-run with all columns
llm_full_shuffle_r2s = []
print(f"  Running {N_SHUFFLE} shuffles (full columns)...", end=" ", flush=True)
for seed in range(N_SHUFFLE):
    rng = np.random.RandomState(seed)
    X_shuf = np.zeros_like(X_llm_raw)
    for i in range(X_llm_raw.shape[0]):
        row = X_llm_raw[i, :].copy()
        if np.sum(row != 0) <= 1:
            X_shuf[i, :] = row
        else:
            X_shuf[i, :] = rng.permutation(row)
    X_full = np.column_stack([X_fc, X_shuf])
    r2_s, _ = ridge_r2(X_full, y)
    llm_full_shuffle_r2s.append(r2_s)
llm_full_shuffle_r2s = np.array(llm_full_shuffle_r2s)
print(f"done.")

llm_full_p = (np.sum(llm_full_shuffle_r2s >= r2_fc_llm) + 1) / (N_SHUFFLE + 1)

print(f"  LLM full: Actual R² = {r2_fc_llm:.4f}, Shuffle mean = {llm_full_shuffle_r2s.mean():.4f} ± {llm_full_shuffle_r2s.std():.4f}")
print(f"            min = {llm_full_shuffle_r2s.min():.4f}, max = {llm_full_shuffle_r2s.max():.4f}, p = {llm_full_p:.3f}")

# Also do dict full for direct comparison
r2_fc_dict, _ = ridge_r2(np.column_stack([X_fc, X_dict]), y)
dict_full_shuffle_r2s = []
print(f"  Dict baseline: Fc + Dict all (296) R² = {r2_fc_dict:.4f}")
print(f"  Running {N_SHUFFLE} shuffles (full columns)...", end=" ", flush=True)
for seed in range(N_SHUFFLE):
    rng = np.random.RandomState(seed)
    X_shuf = np.zeros_like(X_dict)
    for i in range(X_dict.shape[0]):
        X_shuf[i, :] = rng.permutation(X_dict[i, :])
    X_full = np.column_stack([X_fc, X_shuf])
    r2_s, _ = ridge_r2(X_full, y)
    dict_full_shuffle_r2s.append(r2_s)
dict_full_shuffle_r2s = np.array(dict_full_shuffle_r2s)
dict_full_p = (np.sum(dict_full_shuffle_r2s >= r2_fc_dict) + 1) / (N_SHUFFLE + 1)
print(f"done.")
print(f"  Dict full: Actual R² = {r2_fc_dict:.4f}, Shuffle mean = {dict_full_shuffle_r2s.mean():.4f} ± {dict_full_shuffle_r2s.std():.4f}")
print(f"            min = {dict_full_shuffle_r2s.min():.4f}, max = {dict_full_shuffle_r2s.max():.4f}, p = {dict_full_p:.3f}")

# Save full-model results
pd.DataFrame({
    "list": ["dict"] * N_SHUFFLE + ["llm"] * N_SHUFFLE,
    "shuffle_r2": np.concatenate([dict_full_shuffle_r2s, llm_full_shuffle_r2s]),
    "actual_r2": [r2_fc_dict] * N_SHUFFLE + [r2_fc_llm] * N_SHUFFLE,
}).to_csv(TBL_DIR / "check4d_shuffle_full.csv", index=False)

full_summary = [
    {"List": "Dict (296)", "Actual R²": r2_fc_dict,
     "Shuffle Mean R²": round(float(dict_full_shuffle_r2s.mean()), 4),
     "Shuffle Std": round(float(dict_full_shuffle_r2s.std()), 4),
     "Shuffle Min": round(float(dict_full_shuffle_r2s.min()), 4),
     "Shuffle Max": round(float(dict_full_shuffle_r2s.max()), 4),
     "p-value": dict_full_p},
    {"List": "LLM (595)", "Actual R²": r2_fc_llm,
     "Shuffle Mean R²": round(float(llm_full_shuffle_r2s.mean()), 4),
     "Shuffle Std": round(float(llm_full_shuffle_r2s.std()), 4),
     "Shuffle Min": round(float(llm_full_shuffle_r2s.min()), 4),
     "Shuffle Max": round(float(llm_full_shuffle_r2s.max()), 4),
     "p-value": llm_full_p},
]
pd.DataFrame(full_summary).to_csv(TBL_DIR / "check4d_shuffle_full_summary.csv", index=False)
print(f"\n  Saved: {TBL_DIR / 'check4d_shuffle_full.csv'}")
print(f"  Saved: {TBL_DIR / 'check4d_shuffle_full_summary.csv'}\n")

# Figure: full-model shuffle distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Dict full
ax = axes[0]
ax.hist(dict_full_shuffle_r2s, bins=20, color="#4472C4", alpha=0.7, edgecolor="white",
        label=f"Shuffled (n={N_SHUFFLE})")
ax.axvline(r2_fc_dict, color="#C00000", linewidth=2.5, linestyle="--",
           label=f"Actual R² = {r2_fc_dict:.4f}")
ax.set_xlabel("R²", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.set_title(f"Dict List (296 cols)\np = {dict_full_p:.3f}", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# LLM full
ax = axes[1]
ax.hist(llm_full_shuffle_r2s, bins=20, color="#ED7D31", alpha=0.7, edgecolor="white",
        label=f"Shuffled (n={N_SHUFFLE})")
ax.axvline(r2_fc_llm, color="#C00000", linewidth=2.5, linestyle="--",
           label=f"Actual R² = {r2_fc_llm:.4f}")
ax.set_xlabel("R²", fontsize=11)
ax.set_title(f"LLM List (595 cols)\np = {llm_full_p:.3f}", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

fig.suptitle("Label Shuffle Test: All Sentiment Columns (No PC1 Reduction)", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "check4d_shuffle_full.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Figure saved: {FIG_DIR / 'check4d_shuffle_full.png'}\n")

# ===========================================================================
# CHECK 4e: Correlation of Shuffled vs Original Shocks (PC1-based)
# ===========================================================================
print("=" * 60)
print("CHECK 4e: Shock Correlation — Shuffled vs Original")
print("=" * 60)

# Generate original shocks from forecasts + LLM PC1
def get_shock(X, y):
    X = np.array(X, dtype=float, copy=True)
    y = np.array(y, dtype=float, copy=True)
    X = np.nan_to_num(X, nan=0.0)
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_c, y_c = X[mask], y[mask]
    X_s = StandardScaler().fit_transform(X_c)
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 30), cv=5)
    ridge.fit(X_s, y_c)
    y_pred = ridge.predict(X_s)
    return y_c - y_pred, mask

# Original LLM shock
X_orig = np.column_stack([X_fc, llm_pc1])
shock_orig, _ = get_shock(X_orig, y)

# Shuffled shocks
llm_shock_corrs = []
print(f"  Computing {N_SHUFFLE} shuffled shocks...", end=" ", flush=True)
for seed in range(N_SHUFFLE):
    rng = np.random.RandomState(seed)
    X_shuf = np.zeros_like(X_llm_raw)
    for i in range(X_llm_raw.shape[0]):
        row = X_llm_raw[i, :].copy()
        if np.sum(row != 0) <= 1:
            X_shuf[i, :] = row
        else:
            X_shuf[i, :] = rng.permutation(row)
    X_s = StandardScaler().fit_transform(X_shuf)
    pc1_s = PCA(n_components=1).fit_transform(X_s).flatten()
    X_full = np.column_stack([X_fc, pc1_s])
    shock_s, _ = get_shock(X_full, y)
    corr = np.corrcoef(shock_orig, shock_s)[0, 1]
    llm_shock_corrs.append(corr)
llm_shock_corrs = np.array(llm_shock_corrs)
print(f"done.")

print(f"  LLM Shock corr: mean = {llm_shock_corrs.mean():.4f}, std = {llm_shock_corrs.std():.4f}, "
      f"range = [{llm_shock_corrs.min():.4f}, {llm_shock_corrs.max():.4f}]")

# Dict for comparison
X_dict_full = np.column_stack([X_fc, dict_pc1])
shock_dict_orig, _ = get_shock(X_dict_full, y)

dict_shock_corrs = []
print(f"  Computing {N_SHUFFLE} dict shuffled shocks...", end=" ", flush=True)
for seed in range(N_SHUFFLE):
    rng = np.random.RandomState(seed)
    X_shuf = np.zeros_like(X_dict)
    for i in range(X_dict.shape[0]):
        X_shuf[i, :] = rng.permutation(X_dict[i, :])
    X_s = StandardScaler().fit_transform(X_shuf)
    pc1_s = PCA(n_components=1).fit_transform(X_s).flatten()
    X_full = np.column_stack([X_fc, pc1_s])
    shock_s, _ = get_shock(X_full, y)
    corr = np.corrcoef(shock_dict_orig, shock_s)[0, 1]
    dict_shock_corrs.append(corr)
dict_shock_corrs = np.array(dict_shock_corrs)
print(f"done.")

print(f"  Dict Shock corr: mean = {dict_shock_corrs.mean():.4f}, std = {dict_shock_corrs.std():.4f}, "
      f"range = [{dict_shock_corrs.min():.4f}, {dict_shock_corrs.max():.4f}]")

# Save
pd.DataFrame({
    "list": ["dict"] * N_SHUFFLE + ["llm"] * N_SHUFFLE,
    "shock_corr": np.concatenate([dict_shock_corrs, llm_shock_corrs]),
}).to_csv(TBL_DIR / "check4e_shock_corr.csv", index=False)
print(f"  Saved: {TBL_DIR / 'check4e_shock_corr.csv'}\n")

# Figure: Shock correlation distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.hist(dict_shock_corrs, bins=20, color="#4472C4", alpha=0.7, edgecolor="white")
ax.axvline(np.mean(dict_shock_corrs), color="#C00000", linewidth=2, linestyle="--",
           label=f"Mean = {np.mean(dict_shock_corrs):.4f}")
ax.set_xlabel("Correlation with Original Shock", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.set_title(f"Dict List (296 concepts)\nRange: [{dict_shock_corrs.min():.4f}, {dict_shock_corrs.max():.4f}]",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
ax.hist(llm_shock_corrs, bins=20, color="#ED7D31", alpha=0.7, edgecolor="white")
ax.axvline(np.mean(llm_shock_corrs), color="#C00000", linewidth=2, linestyle="--",
           label=f"Mean = {np.mean(llm_shock_corrs):.4f}")
ax.set_xlabel("Correlation with Original Shock", fontsize=11)
ax.set_title(f"LLM List (595 concepts)\nRange: [{llm_shock_corrs.min():.4f}, {llm_shock_corrs.max():.4f}]",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

fig.suptitle("Shock Correlation: Shuffled Labels vs Original", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "check4e_shock_corr.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Figure saved: {FIG_DIR / 'check4e_shock_corr.png'}\n")

# ===========================================================================
# Summary
# ===========================================================================
print("=" * 60)
print("LLM SHUFFLE TEST — COMPLETE")
print("=" * 60)
print(f"""
# Summary of Findings

## Composition (4a)
- Dict: {np.mean(dict_pos_fracs):.1%} positive per meeting (std {np.std(dict_pos_fracs):.3f}), dense
- LLM:  {np.nanmean(llm_pos_fracs):.1%} positive per meeting (std {np.nanstd(llm_pos_fracs):.3f}), 99% sparse
- LLM list has MORE within-meeting variation — shuffling should be more disruptive IF labels matter

## Baseline R² (4b)
- Forecasts only:              {r2_fc:.4f}
- + Dict PC1:                 {r2_fc_dict_pc1:.4f} (Δ = {r2_fc_dict_pc1 - r2_fc:+.4f})
- + LLM PC1:                 {r2_fc_llm_pc1:.4f} (Δ = {r2_fc_llm_pc1 - r2_fc:+.4f})

## PC1 Shuffle Test (4c)
- Dict: Actual R² = {r2_fc_dict_pc1:.4f}, Shuffle mean R² = {dict_shuffle_r2s.mean():.4f} ± {dict_shuffle_r2s.std():.4f}, p = {dict_p_value:.3f}
- LLM:  Actual R² = {r2_fc_llm_pc1:.4f}, Shuffle mean R² = {llm_shuffle_r2s.mean():.4f} ± {llm_shuffle_r2s.std():.4f}, p = {llm_p_value:.3f}

## Full-Model Shuffle Test (4d)
- Dict: Actual R² = {r2_fc_dict:.4f}, Shuffle mean R² = {dict_full_shuffle_r2s.mean():.4f} ± {dict_full_shuffle_r2s.std():.4f}, p = {dict_full_p:.3f}
- LLM:  Actual R² = {r2_fc_llm:.4f}, Shuffle mean R² = {llm_full_shuffle_r2s.mean():.4f} ± {llm_full_shuffle_r2s.std():.4f}, p = {llm_full_p:.3f}

## Shock Correlation (4e)
- Dict: corr = {dict_shock_corrs.mean():.4f} ± {dict_shock_corrs.std():.4f} [{dict_shock_corrs.min():.4f}, {dict_shock_corrs.max():.4f}]
- LLM:  corr = {llm_shock_corrs.mean():.4f} ± {llm_shock_corrs.std():.4f} [{llm_shock_corrs.min():.4f}, {llm_shock_corrs.max():.4f}]
""")
