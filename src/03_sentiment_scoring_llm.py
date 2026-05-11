"""
Step 3: Layer 1 — LLM Sentiment Scoring (full-text, concept-extraction approach).
For each FOMC meeting (1982–2008), send the FULL concatenated text to DeepSeek.
LLM freely extracts ALL economic concepts it finds and scores their sentiment.
Python fuzzy-matches extracted concepts back to the 595-concept master list.
1 API call per meeting = 216 calls total. Covers all 595 concepts.

Cache strategy (DeepSeek prefix caching):
  SYSTEM_PROMPT is static — identical across all 216 calls — so DeepSeek caches
  it after the first call. Meeting date and document text go in the USER message.
"""
import pandas as pd
import numpy as np
import json
import time
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from fuzzywuzzy import fuzz
from _cache_utils import CacheTracker

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEXT_DIR = PROJECT_ROOT / "data" / "processed" / "texts"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sentiment_llm.csv"

# --- Load master concept list from Step 2 ---
concepts_path = PROJECT_ROOT / "data" / "processed" / "concepts" / "llm_concepts.csv"
if concepts_path.exists():
    df_master = pd.read_csv(concepts_path)
    # Filter to economic concepts only
    if "is_economic_concept" in df_master.columns:
        df_master = df_master[df_master["is_economic_concept"] == True]
    master_concepts = df_master["term"].str.lower().str.strip().tolist()
else:
    # Fallback
    orig = pd.read_csv(PROJECT_ROOT / "data" / "create_sentiments" / "Input_files" / "Concepts_LongList.csv")
    master_concepts = orig["terms"].str.lower().str.strip().tolist()

print(f"Master concept list: {len(master_concepts)} concepts")

# --- Group text files by meeting date (1982–2008) ---
df_index = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "text_index.csv")
df_index["meeting_date"] = df_index["meeting_date"].astype(str)
meeting_groups = df_index.groupby("meeting_date")
meeting_dates = sorted(meeting_groups.groups.keys())
meeting_dates = [d for d in meeting_dates if d <= "2008-12-31"]
print(f"Meetings to process: {len(meeting_dates)}")

# --- API client ---
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

SYSTEM_PROMPT = """You are a senior Federal Reserve economist reading a Greenbook document prepared for an FOMC meeting.

Your task: identify ALL economic concepts discussed in this document, and rate the sentiment
expressed toward each concept on a scale from -1 to +1.

Sentiment scale:
  +0.75 to +1.0 : strongly improving / expanding / favorable / above expectations
  +0.25 to +0.5 : moderately improving
  0             : neutral, balanced, or no clear directional tone
  -0.25 to -0.5 : moderately deteriorating
  -0.75 to -1.0 : strongly deteriorating / contracting / concerning / below expectations

Rules:
- Extract EVERY distinct economic concept you find — be comprehensive, not selective.
- Use canonical names: "inflation", "gdp growth", "housing starts", "unemployment rate", etc.
- If a concept is mentioned multiple times with different tones, give the OVERALL assessment.
- If a concept appears but sentiment is purely factual/neutral, score 0.

Return ONLY this JSON (no other text):
{
  "concepts": {
    "concept_name_1": 0.8,
    "concept_name_2": -0.5,
    "concept_name_3": 0.0
  }
}"""

# --- Load existing progress ---
if OUTPUT_PATH.exists():
    existing = pd.read_csv(OUTPUT_PATH)
    processed_dates = set(existing["meeting_date"].tolist())
    all_rows = existing.to_dict("records")
    print(f"Resuming from {len(all_rows)} meetings")
else:
    all_rows = []
    processed_dates = set()

CHECKPOINT_INTERVAL = 5
tracker = CacheTracker()

remaining = [d for d in meeting_dates if d not in processed_dates]
print(f"[Step 3] {len(processed_dates)} done, {len(remaining)} remaining. Processing all.")

for idx, meeting_date in enumerate(meeting_dates):
    if meeting_date in processed_dates:
        continue

    print(f"[{idx+1}/{len(meeting_dates)}] {meeting_date}...", end=" ", flush=True)

    try:
        # Concatenate ALL documents for this meeting — full text, no truncation.
        # Sentiment scoring requires the LLM to see every concept mention;
        # truncating would cause missed concepts and zero-inflated scores.
        meeting_docs = meeting_groups.get_group(meeting_date)
        full_text = ""
        total_words = 0
        for _, row in meeting_docs.iterrows():
            txt_path = Path(row["txt_path"])
            if txt_path.exists():
                doc_text = txt_path.read_text(encoding="utf-8")
                full_text += doc_text + "\n\n"
                total_words += len(doc_text.split())

        if total_words < 100:
            raise ValueError(f"Insufficient text ({total_words} words)")

        # Meeting date and full Greenbook text go in the USER message only.
        # SYSTEM_PROMPT is static → DeepSeek caches it, we pay ~10x less for it.
        user_msg = f"FOMC meeting date: {meeting_date}\n\nGreenbook text:\n{full_text}"

        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        # Log cache performance for this call
        cache_stats = tracker.update(response)
        cache_note = f"cache {cache_stats['hit']:,}hit/{cache_stats['miss']:,}miss ({cache_stats['rate_pct']:.0f}%)"

        raw = response.choices[0].message.content
        data = json.loads(raw)
        extracted = data.get("concepts", {})

        # Fuzzy-match extracted concepts to master list
        row = {"meeting_date": meeting_date}
        matched_count = 0

        for master_c in master_concepts:
            # Direct match first
            if master_c in extracted:
                row[master_c] = float(extracted[master_c])
                matched_count += 1
            else:
                # Fuzzy match: find best match among extracted
                best_score = 0
                best_val = 0.0
                for ext_c, ext_val in extracted.items():
                    score = fuzz.ratio(master_c.lower(), ext_c.lower())
                    if score > best_score and score >= 85:
                        best_score = score
                        best_val = float(ext_val)
                if best_score >= 85:
                    row[master_c] = best_val
                    matched_count += 1
                else:
                    row[master_c] = 0.0  # not mentioned → 0

        all_rows.append(row)

        # Report novel concepts the LLM found outside the master list
        novel = sum(1 for ext_c in extracted
                    if max(fuzz.ratio(ext_c.lower(), mc.lower()) for mc in master_concepts) < 85)
        print(f"OK ({total_words:,}w, {len(extracted)} concepts, "
              f"{matched_count}/{len(master_concepts)} matched, {novel} novel, {cache_note})")

    except Exception as e:
        err_str = str(e)
        print(f"FAILED: {err_str}")
        if "402" in err_str or "Insufficient Balance" in err_str:
            print("\n⚠️  DeepSeek balance exhausted. Top up account before continuing.")
            break
        # Add zero row so the meeting slot is not lost
        row = {"meeting_date": meeting_date}
        for mc in master_concepts:
            row[mc] = 0.0
        all_rows.append(row)

    # Checkpoint
    if (idx + 1) % CHECKPOINT_INTERVAL == 0:
        pd.DataFrame(all_rows).to_csv(OUTPUT_PATH, index=False)
        print(f"  [Checkpoint: {len(all_rows)} rows saved | "
              f"cumulative cache rate: {tracker.cumulative_rate:.1f}%]")

    time.sleep(1)

# --- Final save ---
df_results = pd.DataFrame(all_rows)
df_results.to_csv(OUTPUT_PATH, index=False)
print(f"\nFinal: {len(all_rows)} meetings × {len(master_concepts)} concepts saved to {OUTPUT_PATH}")
tracker.print_summary("Step 3 — Sentiment Scoring")

# --- Report ---
n_meetings = len(all_rows)
avg_scored = np.mean([
    sum(1 for mc in master_concepts if row.get(mc, 0) != 0)
    for row in all_rows
])

report = f"""# Step 3 (Layer 1): LLM Sentiment Scoring Report

## Approach
- **1 API call per meeting** — LLM reads full Greenbook text and freely extracts
  ALL economic concepts it finds with sentiment scores (-1 to +1).
- Python fuzzy-matches extracted concepts back to the {len(master_concepts)}-concept
  master list from Step 2 (85% threshold).
- Concepts not mentioned → 0.0.

## Summary
- **Meetings processed**: {n_meetings}
- **Master concepts**: {len(master_concepts)}
- **Average concepts scored non-zero per meeting**: {avg_scored:.1f}
- **Total API calls**: {n_meetings}
- **Output**: `data/processed/sentiment_llm.csv`

## API Cache Performance
- **Cache hit tokens**: {tracker.hit_tokens:,} ({tracker.cumulative_rate:.1f}%)
- **Cache miss tokens**: {tracker.miss_tokens:,}
- **Estimated saving**: ${tracker.estimated_saving_usd:.4f} USD

## Comparison with Original Dictionary Scores
See `results/reports/06_evaluation_report.md` for full comparison.
"""

report_path = PROJECT_ROOT / "results" / "reports" / "03_sentiment_report.md"
report_path.write_text(report)
print(f"Report saved to {report_path}")
print("Step 3 complete.")
