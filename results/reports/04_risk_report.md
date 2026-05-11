# Step 4: Risk Extraction Report

## Summary
- **Meetings targeted**: 216 (1982-2008)
- **Successfully processed**: 216
- **Failed**: 0
- **Success rate**: 100.0%

## Text Pre-filtering (Risk Paragraphs)
- Input: full concatenated Greenbook text per meeting
- Method: keep paragraphs containing risk keywords ± 1 context paragraph
- Typical reduction: 60–80 % of tokens in the user message
- Fallback: full text used when no keywords match (rare)

## API Cache Performance
- **Cache hit tokens**: 145,536 (7.8%)
- **Cache miss tokens**: 1,732,257
- **Estimated saving**: $0.0183 USD

## Overall Bias Distribution
overall_bias
balanced         118
downside          64
upside            33
not_mentioned      1

## Bias by Decade
overall_bias  balanced  downside  not_mentioned  upside
decade                                                 
1980                44        13              0       7
1990                33        24              1      22
2000                41        27              0       4

## Unemployment Risk Score
- **NBER recession years mean**: 0.536
- **Expansion years mean**: 0.064
- **Difference (recession - expansion)**: 0.472

Higher values = higher unemployment risk → consistent with recession timing.

## Sample Unemployment Evidence Quotes
- **2002-12-10** (upside, medium): "the duration of this period of subpar economic growth is obviously uncertain"
- **1982-11-16** (upside, medium): "If sustained, the recent increase in stock prices also could stimulate household outlays."
- **1994-11-15** (downside, low): "we would view the probability distribution as skewed slightly toward the latter"
- **2000-11-15** (upside, medium): "the unemployment rate rises to 5-1/4 percent by the end of 2002"
- **1988-12-14** (upside, medium): "one or two quarters of negative growth--particularly in domestic demand--certainly cannot be ruled out"

## Failed Meetings
None
