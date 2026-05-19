# monetary-policy-nlp
## Replication & Extension of Aruoba & Drechsel (2024, Econometrica) via LLM-Based NLP

The full paper is at `papers/Original_paper_Aruoba_Drechsel.pdf`. Read it before proceeding on any step that touches methodology, variable definitions, table formats, or result interpretation.


## API Configuration

```python
from openai import OpenAI
from dotenv import load_dotenv
import os, time

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
MODEL = "deepseek-v4-pro"   # maps to DeepSeek-V4-pro
# Use temperature=0.1 for all LLM calls
# Do NOT set max_tokens unless necessary — let the model auto-size output
# Do NOT set response_format={"type": "json_object"} for large-output tasks
#   (the model may refuse to generate if it cannot guarantee complete JSON)
# Use safe_json_parse() from 01a_llm_sentiment_v2.py instead
# Add time.sleep(1.5) between calls
# On 402 error: stop immediately and print "Top up DeepSeek balance before continuing"
```

## Data Paths

| File | Path |
|------|------|
| FOMC PDFs (787) | `data/create_sentiments/FOMC_Texts/` |
| Extracted TXTs | `data/processed/texts/` |
| Greenbook forecasts | `data/create_sentiments/Input_files/Greenbook_Regressioninput_Original.csv` |
| FFR target | `data/create_sentiments/Input_files/FFR_Target.csv` |
| L-M dictionary | `data/create_sentiments/Input_files/Sentiment_Dictionary_NoConcepts_Prolonged.csv` |
| Original 296 concepts | `data/create_sentiments/Input_files/Concepts_LongList.csv` |
| Original sentiment outputs | `data/replication_files/` |
| Authors' shock series | `data/replication_package/descriptives/AD shock.csv` |
| Actual unemployment | `data/replication_package/replication_tables_forecast_error/BLS UNRATE.xls` |

---

## Step 1a — LLM Sentiment Scoring for Robustness Check

### Design Rationale

Aruoba & Drechsel (2024) score sentiment using a dictionary within ±10-word windows
around each concept mention. This implicitly assumes 20 words of local context are
sufficient to determine directional tone.

**We test this assumption** by having an LLM read the ENTIRE Greenbook1 document
and score the SAME 296 concepts with full document context. This is NOT "LLM
replacing dictionary" — it is a robustness check on whether the narrow window is
a binding constraint on information extraction.

### Why Greenbook1 Only

- Greenbook1 = staff qualitative assessment of current economic conditions
  (8K–24K words). Sentiment signals are densest here.
- Greenbook2 = primarily numerical forecast tables with sparse prose.
- Redbook = Board discussion; exists for only 11/216 meetings.
- **This is a design feature, not a compromise**: Greenbook1 provides 8K–24K words
  of prose — orders of magnitude more context than the dictionary's 20-word window.
  If LLM full-context scoring cannot improve on dictionary window scoring with this
  much additional context, the window is not the bottleneck.

### Why 296 Concepts (Not 398)

Only the 296 concepts that have dictionary scores in `total_sentiment_file_std10.csv`
are scored. This makes the two PC1s directly comparable (same concept set, different
scoring method).

### Output

`data/processed/sentiment_llm_v2.csv`: 216 meetings × 296 concepts, scores in [-1, +1].

---

## Non-Negotiable Execution Rules

### Full Coverage — No Truncation of Document Content UNLESS explicitely allowed
- Process **every** meeting in the dataset. Never cut a loop short to save costs or test a subset.
- Before any LLM loop, print: `print(f"[Step X] {len(done)} done, {len(remaining)} remaining. Processing all.")`
- If uncertain which meetings remain, query `text_index.csv` for the full list.
- **Do not truncate Greenbook text** sent to the LLM. Sentiment scoring and risk extraction require reading the full document to avoid missing concepts or risk language. Truncating content to save tokens defeats the purpose of the replication.

### Cache Optimization (DeepSeek Prefix Caching)
DeepSeek caches identical prompt prefixes automatically at ~10x lower cost. The rule is simple: **system prompt must be byte-for-byte identical across all calls in a session.**

- Define `SYSTEM_PROMPT` as a module-level constant. Never interpolate dynamic content (dates, filenames, run numbers) into it.
- All variable content — meeting date, document text, term lists — goes in the **user message only**.
- Use `CacheTracker` from `src/_cache_utils.py` to monitor hit rates in every script:
  ```python
  from _cache_utils import CacheTracker
  tracker = CacheTracker()
  # after each API call:
  stats = tracker.update(response)
  print(f"  cache {stats['hit']:,}hit/{stats['miss']:,}miss ({stats['rate_pct']:.0f}%)")
  # at end of script:
  tracker.print_summary("Step X — description")
  ```
- Run a step **in one continuous session** when possible. Cache entries have a TTL; splitting a 216-meeting job across days wastes cache warming.
- Batch concept classification at **50 terms per call** (not 8). Fewer calls = faster cache warming on the system prompt prefix.

### Token Efficiency (Without Compromising Content)
- `response_format={"type": "json_object"}` and `temperature=0.1` on every LLM call.
- Set `max_tokens` appropriately per task: ~2000 for 50-term classification batches, default for sentiment/risk.
- For non-LLM preprocessing (n-gram counting, fuzzy matching), use Python — no API cost.

### Incremental Saving
- Save every 10 rows; never lose progress on crash.
- Check for existing progress at script start; skip already-processed meetings.
- Write failures to a `*_failed.csv` alongside each output file.

### Error Handling
- **402 error**: stop the loop immediately, print "⚠️ DeepSeek balance exhausted. Top up before continuing.", do not skip ahead.
- Other errors: log to `*_failed.csv`, skip that meeting, continue with next.
- Never halt the entire pipeline on a single meeting failure.

### Code and Reports
- All code comments, variable names, and reports in **English**.
- Reports in Markdown saved to `results/reports/`.
- Scripts in `src/`, one file per numbered step.

