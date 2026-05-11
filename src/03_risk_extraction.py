"""
Step 4: Layer 3 — Risk Asymmetry Extraction.
Extract categorical risk ratings + evidence quotes for each FOMC meeting (1982-2008).

Cache strategy (DeepSeek prefix caching):
  SYSTEM_PROMPT is defined once, never modified, and contains no dynamic content.
  It will be cached by DeepSeek after the first call — all subsequent calls pay
  ~10x less for those tokens. Meeting date and full document text go in the USER
  message only, so the static prefix remains byte-for-byte identical.
"""
import pandas as pd
import numpy as np
import re
import json
import time
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from _cache_utils import CacheTracker, filter_risk_paragraphs

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEXT_DIR = PROJECT_ROOT / "data" / "processed" / "texts"
RISK_INDICATORS_PATH = PROJECT_ROOT / "data" / "processed" / "risk_indicators.csv"
FAILED_PATH = PROJECT_ROOT / "data" / "processed" / "risk_indicators_failed.csv"

# Load text index and group by meeting_date (filter 1982-2008)
df_index = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "text_index.csv")
df_index["meeting_date"] = df_index["meeting_date"].astype(str)

# Group documents by meeting date
meeting_groups = df_index.groupby("meeting_date")
meeting_dates = sorted(meeting_groups.groups.keys())

# Filter to 1982-2008 (as per paper's shock sample)
meeting_dates = [d for d in meeting_dates if d <= "2008-12-31"]
print(f"Found {len(meeting_dates)} meetings (1982-2008)")


SYSTEM_PROMPT = """You are a Federal Reserve economist. Analyze the staff's risk language.
Distinguish the modal forecast (most likely outcome) from tail risks described in the text.

Rate the DIRECTION and STRENGTH of risks around the staff's modal forecast for each variable.
For unemployment, quote the specific phrase that most clearly signals the risk direction.

IMPORTANT:
- "upside" risk for unemployment = text says unemployment might come in HIGHER than the forecast
- "downside" risk for unemployment = text says unemployment might come in LOWER than the forecast
- "balanced" = risks on both sides roughly equal
- "not_mentioned" = variable not discussed in meaningful risk context
- For evidence, quote EXACTLY from the text (short phrase, max 20 words)
- If no clear evidence, set evidence to null

Return ONLY this JSON, no other text:
{
  "unemployment_risk": "upside" | "downside" | "balanced" | "not_mentioned",
  "unemployment_strength": "high" | "medium" | "low",
  "unemployment_evidence": "exact short quote or null",
  "inflation_risk": "upside" | "downside" | "balanced" | "not_mentioned",
  "inflation_strength": "high" | "medium" | "low",
  "output_risk": "upside" | "downside" | "balanced" | "not_mentioned",
  "output_strength": "high" | "medium" | "low",
  "output_evidence": "exact short quote or null",
  "inflation_evidence": "exact short quote or null",
  "overall_bias": "upside" | "downside" | "balanced"
}"""

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    timeout=180.0  # 3-minute timeout; retry once on timeout below
)

# Load existing progress if any
if RISK_INDICATORS_PATH.exists():
    existing = pd.read_csv(RISK_INDICATORS_PATH)
    processed_dates = set(existing["meeting_date"].tolist())
    results = existing.to_dict("records")
    print(f"Resuming from {len(results)} previously processed meetings")
else:
    results = []
    processed_dates = set()

failed = []
tracker = CacheTracker()

remaining = [d for d in meeting_dates if d not in processed_dates]
print(f"[Step 4] {len(processed_dates)} done, {len(remaining)} remaining. Processing all.")

new_count = 0

for idx, meeting_date in enumerate(meeting_dates):
    if meeting_date in processed_dates:
        continue

    new_count += 1

    print(f"[{idx+1}/{len(meeting_dates)}] Processing meeting {meeting_date}...", end=" ", flush=True)

    try:
        # Concatenate ALL documents for this meeting into a single text.
        meeting_docs = meeting_groups.get_group(meeting_date)
        all_text = ""
        for _, row in meeting_docs.iterrows():
            txt_path = Path(row["txt_path"])
            if txt_path.exists():
                all_text += txt_path.read_text(encoding="utf-8") + "\n\n"

        if not all_text.strip():
            raise ValueError("No text content found")

        # Filter to risk-relevant paragraphs (±1 context window around each match).
        # This reduces user-message tokens by ~60–80 % while keeping every paragraph
        # that contains risk language and its immediate neighbours for coherence.
        # If no keywords match, filter_risk_paragraphs falls back to the full text.
        context, filt_stats = filter_risk_paragraphs(all_text)
        fallback_note = " [FALLBACK: full text]" if filt_stats["fallback"] else ""
        tier_note = f"T{filt_stats['tier_used']}" if not filt_stats['fallback'] else "FB"
        filter_note = (f"{filt_stats['filtered_words']:,}w "
                       f"({100 - filt_stats['reduction_pct']:.0f}% of {filt_stats['original_words']:,}w, "
                       f"{filt_stats['paragraphs_kept']}/{filt_stats['paragraphs_total']} paras, "
                       f"{tier_note}{fallback_note})")

        # Meeting date and filtered text go in the USER message only.
        # SYSTEM_PROMPT stays identical across all calls → cache hits on system tokens.
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model="deepseek-v4-pro",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Federal Reserve Greenbook, FOMC meeting {meeting_date}.\n\n{context}"}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                break
            except Exception as api_err:
                err_msg = str(api_err)
                if "timed out" in err_msg.lower() or "timeout" in err_msg.lower():
                    if attempt == 0:
                        print(f"[retry after timeout]", end=" ", flush=True)
                        time.sleep(2)
                        continue
                raise

        # Log cache performance for this call
        cache_stats = tracker.update(response)
        cache_note = f"cache {cache_stats['hit']:,}hit/{cache_stats['miss']:,}miss ({cache_stats['rate_pct']:.0f}%)"

        raw = response.choices[0].message.content
        data = json.loads(raw)

        # Validate required fields
        required = ["unemployment_risk", "unemployment_strength", "unemployment_evidence",
                     "inflation_risk", "inflation_strength", "output_risk",
                     "output_strength", "overall_bias",
                     "output_evidence", "inflation_evidence"]
        for field in required:
            if field not in data:
                data[field] = None

        data["meeting_date"] = meeting_date

        # Encode numeric risk score for regression
        direction_map = {"upside": 1, "balanced": 0, "downside": -1, "not_mentioned": None}
        strength_map = {"high": 1.0, "medium": 0.5, "low": 0.25}

        dir_val = direction_map.get(data.get("unemployment_risk"))
        str_val = strength_map.get(data.get("unemployment_strength"), 0)
        data["unemployment_risk_score"] = dir_val * str_val if dir_val is not None else None

        results.append(data)
        print(f"OK | {filter_note} | bias={data.get('overall_bias')}, "
              f"unemp={data.get('unemployment_risk')} | {cache_note}")

    except Exception as e:
        err_str = str(e)
        failed.append({"meeting_date": meeting_date, "error": err_str})
        print(f"FAILED: {err_str}")
        # Stop immediately on balance error — do not waste remaining budget
        if "402" in err_str or "Insufficient Balance" in err_str:
            print("\n⚠️  DeepSeek balance exhausted. Top up account before continuing.")
            break

    # Save incrementally every 10 newly-processed meetings
    if new_count % 10 == 0:
        pd.DataFrame(results).to_csv(RISK_INDICATORS_PATH, index=False)
        pd.DataFrame(failed).to_csv(FAILED_PATH, index=False)
        print(f"  [Checkpoint: {len(results)} meetings saved | "
              f"cumulative cache rate: {tracker.cumulative_rate:.1f}%]")

    time.sleep(1)

# Final save
df_results = pd.DataFrame(results)
df_results.to_csv(RISK_INDICATORS_PATH, index=False)
pd.DataFrame(failed).to_csv(FAILED_PATH, index=False)
print(f"\nFinal: {len(results)} meetings processed, {len(failed)} failed")
tracker.print_summary("Step 4 — Risk Extraction")

# --- Generate Report ---

# Encode risk score for analysis
df_rep = df_results.copy()

# Overall bias distribution
bias_dist = df_rep["overall_bias"].value_counts().to_string()

# By decade
df_rep["year"] = df_rep["meeting_date"].str[:4].astype(int)
df_rep["decade"] = (df_rep["year"] // 10) * 10
bias_by_decade = pd.crosstab(df_rep["decade"], df_rep["overall_bias"]).to_string()

# Top 5 evidence quotes
evidence_sample = df_rep[df_rep["unemployment_evidence"].notna() & (df_rep["unemployment_evidence"] != "null")]
if len(evidence_sample) > 5:
    evidence_sample = evidence_sample.sample(5, random_state=42)

sample_quotes = "\n".join(
    f"- **{row['meeting_date']}** ({row['unemployment_risk']}, {row['unemployment_strength']}): \"{row['unemployment_evidence']}\""
    for _, row in evidence_sample.iterrows()
)

# Check NBER recession correlation (simple)
recession_years = {1990, 1991, 2001, 2008}
df_rep["is_recession"] = df_rep["year"].isin(recession_years)
recession_mean = df_rep[df_rep["is_recession"]]["unemployment_risk_score"].mean()
expansion_mean = df_rep[~df_rep["is_recession"]]["unemployment_risk_score"].mean()

report = f"""# Step 4: Risk Extraction Report

## Summary
- **Meetings targeted**: {len(meeting_dates)} (1982-2008)
- **Successfully processed**: {len(results)}
- **Failed**: {len(failed)}
- **Success rate**: {len(results)/len(meeting_dates)*100:.1f}%

## Text Pre-filtering (Risk Paragraphs)
- Input: full concatenated Greenbook text per meeting
- Method: keep paragraphs containing risk keywords ± 1 context paragraph
- Typical reduction: 60–80 % of tokens in the user message
- Fallback: full text used when no keywords match (rare)

## API Cache Performance
- **Cache hit tokens**: {tracker.hit_tokens:,} ({tracker.cumulative_rate:.1f}%)
- **Cache miss tokens**: {tracker.miss_tokens:,}
- **Estimated saving**: ${tracker.estimated_saving_usd:.4f} USD

## Overall Bias Distribution
{bias_dist}

## Bias by Decade
{bias_by_decade}

## Unemployment Risk Score
- **NBER recession years mean**: {recession_mean:.3f}
- **Expansion years mean**: {expansion_mean:.3f}
- **Difference (recession - expansion)**: {recession_mean - expansion_mean:.3f}

Higher values = higher unemployment risk → consistent with recession timing.

## Sample Unemployment Evidence Quotes
{sample_quotes}

## Failed Meetings
{chr(10).join(f'- {f["meeting_date"]}: {f["error"]}' for f in failed) if failed else 'None'}
"""

report_path = PROJECT_ROOT / "results" / "reports" / "04_risk_report.md"
report_path.write_text(report)
print(f"Report saved to {report_path}")
print("Step 4 complete.")
