"""
Step 1a: LLM Sentiment Scoring — Full-Context Validation of ±10-Word Window.

================================================================================
DESIGN RATIONALE
================================================================================

Aruoba & Drechsel (2024) score sentiment for 296 economic concepts using the
Loughran-McDonald dictionary within ±10-word windows around each concept
mention. This implicitly assumes that 20 words of local context are sufficient
to determine the directional tone of any concept reference.

We TEST this assumption by having an LLM (DeepSeek V4 Pro) read the ENTIRE
Greenbook1 document — the staff's qualitative economic assessment — and score
the SAME 296 concepts with full document context.

This is NOT "LLM replacing the dictionary." It is a robustness check on the
window-width assumption. If the LLM's full-context PC1 is highly correlated
with the dictionary's ±10-word PC1 and produces similar first-stage fit, then
the narrow window is not a binding constraint on information extraction.

Why Greenbook1 only (not Greenbook2 / Redbook):
  - Greenbook1 contains the staff's qualitative assessment of current economic
    conditions — where sentiment signals are densest.
  - Greenbook2 is primarily numerical forecast tables with sparse prose.
  - Redbook (Board discussion) exists for only 11/216 meetings.
  - Using Greenbook1 only is a DESIGN FEATURE, not a compromise: we are
    testing whether FULL CONTEXT (the entire document, not ±10 words) matters.
    Greenbook1 provides 8K–24K words of prose — far more context than the
    dictionary's 20-word window. If full-context LLM scoring can't improve on
    dictionary scoring with this much additional context, the window is not
    the bottleneck.

Why 296 concepts (not 398):
  - These are the EXACT concepts scored by the dictionary method.
  - Using the same concept set makes the two PC1s directly comparable.
  - The 100 extra concepts in Concepts_LongList.csv have no dictionary scores
    to compare against, so including them would add noise without value.

What this script does:
  1. Loads the 296 overlapping concepts (dict sentiment ∩ LongList)
  2. For each meeting 1982–2008, reads ONLY greenbook1 text
  3. Sends the full greenbook1 text + concept list to DeepSeek
  4. LLM scores each concept from -1 to +1
  5. Saves incrementally to sentiment_llm_v2.csv

Expected output:
  - 216 meetings × 296 concepts
  - Input: ~15K words (~20K tokens) per meeting → manageable
  - ~15–30 seconds per API call → ~1–2 hours total

Cache strategy: the 296-concept list is static → goes in SYSTEM_PROMPT for
automatic DeepSeek prefix caching across all calls in the session.
================================================================================
"""
import pandas as pd
import numpy as np
import json
import re
import time
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from _cache_utils import CacheTracker


# ---------------------------------------------------------------------------
# Robust JSON extraction from LLM output
# ---------------------------------------------------------------------------

def safe_json_parse(raw_text):
    """Extract valid JSON from LLM response, handling common formatting issues.

    DeepSeek may wrap JSON in markdown fences (```json ... ```) or include
    explanatory text before/after the JSON object. This function applies
    increasingly aggressive cleaning to recover the JSON payload.
    """
    if raw_text is None:
        raise ValueError("Response content is None")

    text = raw_text.strip()

    # Attempt 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 3: regex extract JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    preview = raw_text[:300] if raw_text else "(empty)"
    raise ValueError(f"Cannot parse JSON from response. Preview: {preview}")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Paths ---
DICT_SENT_PATH = (PROJECT_ROOT / "data" / "replication_files" / "Sentiment_Final"
                  / "total_sentiment_file_std10.csv")
LONGLIST_PATH  = PROJECT_ROOT / "data" / "create_sentiments" / "Input_files" / "Concepts_LongList.csv"
TEXT_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "text_index.csv"
OUTPUT_PATH    = PROJECT_ROOT / "data" / "processed" / "sentiment_llm_v2.csv"
FAILED_PATH    = PROJECT_ROOT / "data" / "processed" / "sentiment_llm_v2_failed.csv"

# ---------------------------------------------------------------------------
# 1. Determine the 296 overlapping concepts
# ---------------------------------------------------------------------------

# Dictionary sentiment columns (296 concepts with scores from the replication package)
df_dict = pd.read_csv(DICT_SENT_PATH)
dict_cols_raw = [c for c in df_dict.columns
                 if c not in (df_dict.columns[0], "meeting_date") and str(c) != "nan"]
dict_concepts = set(c.lower().strip() for c in dict_cols_raw if c)
print(f"Dictionary sentiment concepts: {len(dict_concepts)}")

# LongList (398 concepts from the paper's concept-selection algorithm)
df_long = pd.read_csv(LONGLIST_PATH)
long_col = df_long.columns[0]
long_concepts = set(df_long[long_col].dropna().astype(str).str.strip().str.lower())
print(f"LongList concepts: {len(long_concepts)}")

# Overlap — these are the concepts we score with LLM
overlap_concepts = sorted(dict_concepts & long_concepts)
print(f"Overlap (concepts to score): {len(overlap_concepts)}")

# Report any dictionary concepts NOT in LongList (should be 0)
dict_only = dict_concepts - long_concepts
if dict_only:
    print(f"WARNING: {len(dict_only)} dict concepts not in LongList: {sorted(dict_only)[:10]}")

# ---------------------------------------------------------------------------
# 2. Build meeting list — Greenbook1 only
# ---------------------------------------------------------------------------

df_index = pd.read_csv(TEXT_INDEX_PATH)
df_index["meeting_date"] = df_index["meeting_date"].astype(str)

# FILTER: only greenbook1 documents (the staff qualitative assessment)
# This is the key design choice — see module docstring for rationale.
df_gb1 = df_index[df_index["doc_type"] == "greenbook1"].copy()
print(f"\nGreenbook1 files: {len(df_gb1)}")

# Group by meeting date
meeting_groups = df_gb1.groupby("meeting_date")
all_meeting_dates = sorted(meeting_groups.groups.keys())

# Restrict to 1982–2008 (paper's Table 3 sample period)
meeting_dates = [d for d in all_meeting_dates if "1982" <= d[:4] <= "2008"]
print(f"Meetings to process (1982–2008, greenbook1 only): {len(meeting_dates)}")

# Quick stats on text sizes
word_counts = []
for meeting_date in meeting_dates:
    meeting_docs = meeting_groups.get_group(meeting_date)
    total_words = 0
    for _, row in meeting_docs.iterrows():
        txt_path = Path(row["txt_path"])
        if txt_path.exists():
            total_words += len(txt_path.read_text(encoding="utf-8").split())
    word_counts.append(total_words)

wc = np.array(word_counts)
print(f"Greenbook1 word count: mean={wc.mean():,.0f}, median={np.median(wc):,.0f}, "
      f"min={wc.min():,}, max={wc.max():,}")
print(f"Estimated tokens/meeting: ~{wc.mean()*1.3:,.0f} (input) + ~{len(overlap_concepts)*5} (output)")

# ---------------------------------------------------------------------------
# 3. Build system prompt (static → cached by DeepSeek)
# ---------------------------------------------------------------------------

# Compact JSON array of concept names
concept_list_json = json.dumps(overlap_concepts)

SYSTEM_PROMPT = (
    "You are a Federal Reserve economist. Your task is to rate the sentiment "
    "expressed toward each economic concept listed below, based on the Greenbook "
    "document in the user message.\n\n"
    "Scale:\n"
    "  +0.75 to +1.0 : strongly improving, expanding, favorable, above expectations\n"
    "  +0.25 to +0.5 : moderately improving, somewhat favorable\n"
    "   0.0          : neutral, balanced, purely factual, or NOT mentioned\n"
    "  -0.25 to -0.5 : moderately deteriorating, somewhat concerning\n"
    "  -0.75 to -1.0 : strongly deteriorating, contracting, concerning, below expectations\n\n"
    "Rules:\n"
    "1. Score EVERY concept in the list. If a concept is not discussed, assign 0.0.\n"
    "2. If a concept appears multiple times with mixed tones, give the OVERALL net assessment.\n"
    "3. Base your assessment on the FULL document context, not just adjacent words.\n"
    "4. Return ONLY valid JSON. No markdown fences, no explanatory text.\n\n"
    f"Concepts to score ({len(overlap_concepts)} total):\n{concept_list_json}"
)

print(f"System prompt: {len(SYSTEM_PROMPT):,} chars (~{len(SYSTEM_PROMPT)//4:,} tokens)")

# ---------------------------------------------------------------------------
# 4. Resume from existing progress (if any)
# ---------------------------------------------------------------------------

processed_dates = set()
if OUTPUT_PATH.exists():
    existing = pd.read_csv(OUTPUT_PATH)
    processed_dates = set(existing["meeting_date"].astype(str).tolist())
    print(f"Resuming: {len(processed_dates)} meetings already processed")
else:
    print("Starting fresh")

# Previously failed meetings (will retry)
if FAILED_PATH.exists():
    failed_df = pd.read_csv(FAILED_PATH)
    failed_dates = set(failed_df["meeting_date"].astype(str).tolist())
    print(f"Previously failed (will retry): {len(failed_dates)} meetings")

remaining = [d for d in meeting_dates if d not in processed_dates]
print(f"[Step 1a] {len(processed_dates)} done, {len(remaining)} remaining. Processing all.\n")

# ---------------------------------------------------------------------------
# 5. API client
# ---------------------------------------------------------------------------

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

CHECKPOINT_INTERVAL = 10
tracker = CacheTracker()
all_rows = []
total_call_time = 0.0

if OUTPUT_PATH.exists():
    all_rows = pd.read_csv(OUTPUT_PATH).to_dict("records")

# ---------------------------------------------------------------------------
# 6. Main loop — one API call per meeting
# ---------------------------------------------------------------------------

for i, meeting_date in enumerate(meeting_dates):
    if meeting_date in processed_dates:
        continue

    # Progress counter
    n_processed = len([r for r in all_rows if r["meeting_date"] in processed_dates
                       or r["meeting_date"] not in processed_dates])
    print(f"[{i+1}/{len(meeting_dates)}] {meeting_date}...", end=" ", flush=True)

    try:
        # --- Assemble greenbook1 text for this meeting ---
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

        # --- API call ---
        user_msg = (f"FOMC meeting date: {meeting_date}\n\n"
                    f"Greenbook document (staff qualitative assessment):\n\n{full_text}")

        t0 = time.time()
        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.1
        )
        elapsed = time.time() - t0
        total_call_time += elapsed

        # --- Cache monitoring ---
        cache_stats = tracker.update(response)
        cache_note = (f"cache {cache_stats['hit']:,}hit/{cache_stats['miss']:,}miss "
                      f"({cache_stats['rate_pct']:.0f}%)")

        # --- Parse response ---
        raw = response.choices[0].message.content
        finish = response.choices[0].finish_reason

        if finish != "stop":
            print(f"WARN: finish_reason={finish}, content_len={len(raw or '')}",
                  end=" ", flush=True)

        data = safe_json_parse(raw)

        # --- Build output row ---
        row = {"meeting_date": meeting_date}
        n_scored = 0
        n_positive = 0
        n_negative = 0

        for c in overlap_concepts:
            score = float(data.get(c, 0.0))
            score = max(-1.0, min(1.0, score))  # clamp to [-1, 1]
            row[c] = score
            if score != 0.0:
                n_scored += 1
                if score > 0:
                    n_positive += 1
                elif score < 0:
                    n_negative += 1

        all_rows.append(row)
        print(f"OK ({total_words:,}w input, {n_scored}/{len(overlap_concepts)} non-zero, "
              f"+{n_positive}/-{n_negative}, {elapsed:.1f}s, {cache_note})")

    except Exception as e:
        err_str = str(e)
        print(f"FAILED: {err_str[:150]}")

        # 402 = out of money — stop immediately
        if "402" in err_str or "Insufficient Balance" in err_str:
            print("\n⚠️  DeepSeek balance exhausted. Top up account before continuing.")
            pd.DataFrame(all_rows).to_csv(OUTPUT_PATH, index=False)
            print(f"Progress saved ({len(all_rows)} rows). Exiting.")
            raise SystemExit(1)

        # Log failure for debugging
        raw_preview = ""
        try:
            raw_preview = (raw or "")[:300]
        except Exception:
            pass
        failed_row = {
            "meeting_date": meeting_date,
            "error": err_str[:200],
            "raw_preview": raw_preview
        }
        if FAILED_PATH.exists():
            pd.DataFrame([failed_row]).to_csv(FAILED_PATH, mode='a', header=False, index=False)
        else:
            pd.DataFrame([failed_row]).to_csv(FAILED_PATH, index=False)

        # Add zero row so the meeting slot is not lost
        row = {"meeting_date": meeting_date}
        for c in overlap_concepts:
            row[c] = 0.0
        all_rows.append(row)

    # --- Checkpoint save every N meetings ---
    if len(all_rows) % CHECKPOINT_INTERVAL == 0:
        df_checkpoint = pd.DataFrame(all_rows)
        df_checkpoint.to_csv(OUTPUT_PATH, index=False)
        avg_time = total_call_time / max(len(all_rows), 1)
        remaining_est = len(remaining) * avg_time / 60
        print(f"  [Checkpoint: {len(all_rows)} rows | "
              f"cumulative cache: {tracker.cumulative_rate:.1f}% | "
              f"est. remaining: {remaining_est:.0f}min]")

    time.sleep(1.5)

# ---------------------------------------------------------------------------
# 7. Final save and summary
# ---------------------------------------------------------------------------

df_results = pd.DataFrame(all_rows)
cols = ["meeting_date"] + overlap_concepts
df_results = df_results[cols]
df_results.to_csv(OUTPUT_PATH, index=False)

n_meetings = len(df_results)
scores_array = df_results[overlap_concepts].astype(float).values
avg_nonzero = (scores_array != 0).sum(axis=1).mean()
avg_positive = (scores_array > 0).sum(axis=1).mean()
avg_negative = (scores_array < 0).sum(axis=1).mean()
all_zero_ratio = (scores_array.sum(axis=1) == 0).mean()

print(f"\n{'='*52}")
print(f"  Step 1a — LLM Sentiment Scoring Complete")
print(f"  Meetings processed:   {n_meetings}")
print(f"  Concepts scored:      {len(overlap_concepts)}")
print(f"  Avg non-zero/meeting: {avg_nonzero:.1f}")
print(f"  Avg positive/meeting: {avg_positive:.1f}")
print(f"  Avg negative/meeting: {avg_negative:.1f}")
print(f"  All-zero meetings:    {all_zero_ratio:.1%}")
print(f"  Total API time:       {total_call_time/60:.1f} min")
print(f"  Output:               {OUTPUT_PATH}")
print(f"{'='*52}")

tracker.print_summary("Step 1a — LLM Sentiment Scoring (greenbook1, 296 concepts)")

if all_zero_ratio > 0.5:
    print("WARNING: >50% meetings have all-zero scores. Prompt or parsing may be broken.\n")
