# Step 3 (Layer 1): LLM Sentiment Scoring Report

## Approach
- **1 API call per meeting** — LLM reads full Greenbook text and freely extracts
  ALL economic concepts it finds with sentiment scores (-1 to +1).
- Python fuzzy-matches extracted concepts back to the 595-concept
  master list from Step 2 (85% threshold).
- Concepts not mentioned → 0.0.

## Summary
- **Meetings processed**: 216
- **Master concepts**: 595
- **Average concepts scored non-zero per meeting**: 5.7
- **Total API calls**: 216
- **Output**: `data/processed/sentiment_llm.csv`

## Comparison with Original Dictionary Scores
See `results/reports/06_evaluation_report.md` for full comparison.
