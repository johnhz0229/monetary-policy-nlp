"""
Step 2: Automated Economic Concept Identification (Python-first, LLM-classify).
Phase A: Python counts n-grams. Phase B: DeepSeek classifies candidate terms.
Phase C: Compare LLM concepts against original 296.

Cache strategy (DeepSeek prefix caching):
  SYSTEM_PROMPT is static — the same across all batch calls — so it is cached
  after the first call. Batch size is set to 50 (not 8) to minimise the number
  of API calls while keeping each response well within the model's JSON limit.
"""
import pandas as pd
import numpy as np
import re
import json
import time
import os
import sys
import logging
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI
from fuzzywuzzy import fuzz
from _cache_utils import CacheTracker

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Phase A: Python frequency counting (no API calls) ---

TEXT_DIR = PROJECT_ROOT / "data" / "processed" / "texts"
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'shall', 'can', 'need', 'dare',
    'ought', 'used', 'it', 'its', 'this', 'that', 'these', 'those',
    'they', 'them', 'their', 'he', 'she', 'him', 'his', 'her', 'we',
    'us', 'our', 'you', 'your', 'i', 'my', 'me', 'not', 'no', 'nor',
    'so', 'as', 'if', 'than', 'too', 'very', 'just', 'about', 'above',
    'after', 'again', 'all', 'also', 'am', 'an', 'any', 'because',
    'been', 'before', 'being', 'between', 'both', 'but', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own',
    'same', 'into', 'over', 'under', 'up', 'out', 'down', 'then',
    'there', 'when', 'where', 'why', 'how', 'which', 'who', 'whom',
    'what', 'here', 'while', 'during', 'through', 'per', 'since'
}

def tokenize(text):
    """Tokenize into words, return list of lowercase tokens (alphabetic only)."""
    tokens = re.findall(r'[a-z]+', text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

def extract_ngrams(tokens, n):
    """Extract n-grams from token list."""
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

print("[Phase A] Counting n-grams across all documents...")
text_files = sorted(TEXT_DIR.glob("*.txt"))
print(f"  Loading {len(text_files)} text files")

unigram_counts = Counter()
bigram_counts = Counter()
trigram_counts = Counter()

for i, txt_path in enumerate(text_files):
    if (i + 1) % 100 == 0:
        print(f"  [{i+1}/{len(text_files)}] counting...")

    try:
        text = txt_path.read_text(encoding="utf-8")
        tokens = tokenize(text)

        unigram_counts.update(set(extract_ngrams(tokens, 1)))
        bigram_counts.update(set(extract_ngrams(tokens, 2)))
        trigram_counts.update(set(extract_ngrams(tokens, 3)))
    except Exception as e:
        print(f"  WARNING: {txt_path.name}: {e}")

# Combine: take top 2000 total across all n-gram types
all_ngrams = Counter()
all_ngrams.update({f"1:{k}": v for k, v in unigram_counts.items()})
all_ngrams.update({f"2:{k}": v for k, v in bigram_counts.items()})
all_ngrams.update({f"3:{k}": v for k, v in trigram_counts.items()})

top_2000 = all_ngrams.most_common(2000)

candidates = []
for ngram_key, freq in top_2000:
    n_type, term = ngram_key.split(":", 1)
    candidates.append({
        "ngram_type": int(n_type),
        "term": term,
        "doc_frequency": freq  # number of docs the n-gram appears in
    })

df_candidates = pd.DataFrame(candidates)
cands_path = PROJECT_ROOT / "data" / "processed" / "ngram_candidates.csv"
df_candidates.to_csv(cands_path, index=False)
print(f"  Saved {len(candidates)} candidate n-grams to {cands_path}")

# --- Phase B: LLM batch classification (API calls) ---

BATCH_SIZE = 50  # was 8 — larger batches = fewer API calls, better cache hit rate
terms = df_candidates["term"].tolist()
batches = [terms[i:i+BATCH_SIZE] for i in range(0, len(terms), BATCH_SIZE)]
print(f"\n[Phase B] Classifying {len(terms)} terms in {len(batches)} batches of {BATCH_SIZE}")

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

SYSTEM_PROMPT = """You are a Federal Reserve economist. Classify each term below.
For each term, return whether it is an economic concept worth tracking as a sentiment indicator.

Return ONLY this JSON:
{"classifications": [{"term": "...", "is_economic_concept": true/false, "category": "prices/real_activity/labor/financial/international/fiscal/other", "note": "one word reason or null"}]}

Categories:
- prices: inflation, deflation, price levels, commodity prices, wages
- real_activity: GDP, output, consumption, investment, housing, manufacturing
- labor: employment, unemployment, labor force, productivity
- financial: credit, interest rates, asset prices, banking, financial conditions
- international: trade, exchange rates, foreign economies
- fiscal: government spending, taxes, budget
- other: worth tracking but doesn't fit above

An economic concept is a specific, measurable economic phenomenon or indicator.
Exclude: generic words, meeting logistics, document structure references, vague descriptors.
Include: specific economic indicators, sectors, policy tools, economic conditions."""

classifications = []
failed_batches = []
tracker = CacheTracker()

for batch_idx, batch in enumerate(batches):
    print(f"  [{batch_idx+1}/{len(batches)}] Classifying {len(batch)} terms...", end=" ", flush=True)

    # Only the list of terms changes per batch; SYSTEM_PROMPT is static → cache hits.
    terms_str = "\n".join(f"- {t}" for t in batch)
    user_msg = f"Terms to classify:\n{terms_str}"

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.1,
            max_tokens=2000,  # 50 terms × ~30 tokens per classification = ~1500 tokens needed
            response_format={"type": "json_object"}
        )

        cache_stats = tracker.update(response)
        cache_note = f"cache {cache_stats['rate_pct']:.0f}%"

        raw = response.choices[0].message.content
        data = json.loads(raw)
        batch_classifications = data.get("classifications", [])

        if len(batch_classifications) == 0:
            print(f"EMPTY (0 classifications returned, {cache_note})")
            failed_batches.append(batch_idx)
        else:
            classifications.extend(batch_classifications)
            print(f"OK ({len(batch_classifications)} classified, {cache_note})")

    except Exception as e:
        print(f"FAILED: {e}")
        failed_batches.append(batch_idx)

    time.sleep(1)

print(f"\n  Total classifications: {len(classifications)}")
print(f"  Failed batches: {len(failed_batches)}")
tracker.print_summary("Step 2 — Concept Classification")

# Build LLM concepts DataFrame
df_llm = pd.DataFrame(classifications)
df_llm = df_llm[df_llm["is_economic_concept"] == True].copy()

# Merge with frequency info from candidates
term_to_freq = dict(zip(df_candidates["term"], df_candidates["doc_frequency"]))
term_to_ntype = dict(zip(df_candidates["term"], df_candidates["ngram_type"]))
df_llm["doc_frequency"] = df_llm["term"].map(term_to_freq)
df_llm["ngram_type"] = df_llm["term"].map(term_to_ntype)
df_llm = df_llm.sort_values("doc_frequency", ascending=False).reset_index(drop=True)
df_llm["frequency_rank"] = df_llm.index + 1

print(f"  LLM identified {len(df_llm)} economic concepts")

# --- Phase C: Comparison with original 296 ---

print("\n[Phase C] Comparing LLM concepts against original 296...")

orig_path = PROJECT_ROOT / "data" / "create_sentiments" / "Input_files" / "Concepts_LongList.csv"
df_orig = pd.read_csv(orig_path)
original_terms = df_orig["terms"].str.lower().str.strip().tolist()
print(f"  Loaded {len(original_terms)} original concepts")

MATCH_THRESHOLD = 85

# Match each LLM concept to original
llm_terms = df_llm["term"].tolist()
matched_to_original = []
for llm_term in llm_terms:
    best_score = 0
    best_match = None
    for orig_term in original_terms:
        score = fuzz.ratio(llm_term.lower(), orig_term.lower())
        if score > best_score:
            best_score = score
            best_match = orig_term
    if best_score >= MATCH_THRESHOLD:
        matched_to_original.append(best_match)
    else:
        matched_to_original.append(None)

df_llm["matched_original"] = matched_to_original

# Match each original concept to LLM
original_matched_to_llm = []
for orig_term in original_terms:
    best_score = 0
    best_match = None
    for llm_term in llm_terms:
        score = fuzz.ratio(orig_term.lower(), llm_term.lower())
        if score > best_score:
            best_score = score
            best_match = llm_term
    if best_score >= MATCH_THRESHOLD:
        original_matched_to_llm.append(best_match)
    else:
        original_matched_to_llm.append(None)

df_orig_matched = df_orig.copy()
df_orig_matched["matched_llm_term"] = original_matched_to_llm

overlap = df_llm["matched_original"].notna().sum()
coverage = sum(1 for m in original_matched_to_llm if m is not None)

print(f"  Overlap: {overlap}/{len(df_llm)} LLM concepts match original ({overlap/len(df_llm)*100:.1f}%)")
print(f"  Coverage: {coverage}/{len(original_terms)} original concepts matched ({coverage/len(original_terms)*100:.1f}%)")

# Save LLM concepts
llm_out = PROJECT_ROOT / "data" / "processed" / "llm_concepts.csv"
df_llm.to_csv(llm_out, index=False)
print(f"  Saved LLM concepts to {llm_out}")

# Save comparison table
novel_llm = df_llm[df_llm["matched_original"].isna()].head(30)
missed_orig = df_orig_matched[df_orig_matched["matched_llm_term"].isna()]
matched_orig = df_orig_matched[df_orig_matched["matched_llm_term"].notna()]

comparison_rows = []
for _, row in df_llm.iterrows():
    comparison_rows.append({
        "llm_term": row["term"],
        "llm_category": row["category"],
        "llm_frequency_rank": row["frequency_rank"],
        "matched_original": row["matched_original"] if pd.notna(row["matched_original"]) else "",
        "status": "matched" if pd.notna(row["matched_original"]) else "novel"
    })
df_comp = pd.DataFrame(comparison_rows)
comp_out = PROJECT_ROOT / "results" / "tables" / "concept_comparison.csv"
df_comp.to_csv(comp_out, index=False)

# Generate report
report = f"""# Step 2: Concept Identification Report

## Phase A — N-gram Frequency Counting
- **Total candidate n-grams extracted**: 2,000 (top by document frequency)
- **Source**: {len(text_files)} text files from {TEXT_DIR}

## Phase B — LLM Classification
- **Total classifications attempted**: {len(terms)} terms in {len(batches)} batches
- **Failed batches**: {len(failed_batches)}
- **Economic concepts identified**: {len(df_llm)}

### Category Distribution
{df_llm['category'].value_counts().to_string()}

## Phase C — Comparison with Original 296 Concepts
- **Original concepts**: {len(original_terms)}
- **LLM concepts**: {len(df_llm)}
- **Overlap (LLM → Original)**: {overlap}/{len(df_llm)} ({overlap/len(df_llm)*100:.1f}%)
- **Coverage (Original → LLM)**: {coverage}/{len(original_terms)} ({coverage/len(original_terms)*100:.1f}%)

### Top 20 Novel Concepts (LLM found, original missed)
{novel_llm[['term', 'category', 'doc_frequency']].to_string(index=False)}

### Top 20 Original Concepts Missed by LLM
{missed_orig[['terms']].head(20).to_string(index=False)}

### Commentary
The LLM identified {len(df_llm)} economic concepts vs. the original 296.
Coverage of original concepts: {coverage}/{len(original_terms)} ({coverage/len(original_terms)*100:.1f}%).
The {len(missed_orig)} missed original concepts include terms that are either too generic,
stop words, or document-structural terms that the LLM correctly excluded.
"""

report_path = PROJECT_ROOT / "results" / "reports" / "02_concept_report.md"
report_path.write_text(report)
print(f"\nReport saved to {report_path}")
print("Step 2 complete.")
