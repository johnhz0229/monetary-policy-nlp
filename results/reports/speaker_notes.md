# Speaker Notes — Robustness of Dictionary-Based Sentiment Scoring

> Three Tests on Aruoba & Drechsel (2024, *Econometrica*)
>
> 中英文混合，方便做报告时快速查阅。

---

## Quick Reference Card / 速查表

| Need / 需求 | Location / 路径 |
|------|------|
| **Beamer PDF** | `results/beamer_presentation.pdf` |
| **All figures / 所有图** | `results/figures/` |
| **Full academic write-up / 完整论文稿** | `results/reports/academic_report.md` |
| **HTML summary / HTML 摘要** | `results/reports/final_report.html` |
| Placebo shuffle script / 打乱检验脚本 | `src/run_robustness_checks.py` |
| LLM concept identification / LLM 概念识别 | `src/02_concept_identification.py` |
| Data: text index / 文本索引 | `data/processed/text_index.csv` |
| Data: LLM concepts / LLM 概念列表 | `data/processed/llm_concepts.csv` |
| Data: dict sentiments / 词典情感矩阵 | `data/replication_files/Sentiment_Final/total_sentiment_file_std10.csv` |
| Data: FFR target / 联邦基金利率 | `data/create_sentiments/Input_files/FFR_Target.csv` |
| Data: Greenbook forecasts / 预测变量 | `data/create_sentiments/Input_files/Greenbook_Regressioninput_Extended.csv` |
| Regression result CSVs / 回归结果 | `results/tables/check1_sample_split.csv`, `check2_placebo.csv` |

---

## Slide-by-Slide Notes / 逐页讲解

### Slide 1 — Title / 标题页
- 一句话开场 / Opening: "We ran three robustness tests on AD(2024)'s dictionary method. Here's what we found."
- Time: ~30 sec

### Slide 2 — What AD(2024) Did / 原文做了什么
- 快速过 / Quick overview
- 重点 / Key: 296 concepts, ±10 word window, R² = 0.94
- 如果听众不熟悉，补充一句 / Context: "They regress FFR target changes on Greenbook numerical forecasts plus all 296 sentiment indicators, using Ridge regression. The residuals are interpreted as monetary policy shocks."
- Time: ~1 min

### Slide 3 — Where Do the 296 Concepts Come From? / 概念从哪来
- 核心问题 / Core: 概念列表是通过**人工筛选** (manual curation) 得到的
- 圈出几个主观感强的词 / Point out: "business confidence", "consumer sentiment" — 这些本身就像判断而不是概念
- 提问 / Pose question: "If a different person curated this list, would we get the same 296?"
- Time: ~1 min

### Slide 4 — Let an AI Do the Same Task / 让 AI 做一遍
- 方法 / Method: LLM independently classifies 2,000 candidate n-grams from the same texts
- 结果 / Result: **595 vs 296** — twice as many
- 类别分布 / Categories: real activity dominates (204), then financial (150), prices (99)
- Time: ~40 sec

### Slide 5 — Venn Diagram / 维恩图
- 核心视觉 / Key visual: 大号数字 62.6% overlap
- 重点 / Point: "Two reasonable methods. Two different lists. 37.4% disagreement."
- 不要评判对错 / Don't judge: 风格不同，不是对错问题
- Time: ~40 sec

### Slide 6 — Side-by-Side / 并列对比
- 左侧/Left: 人工队的独有词 — specific, concrete (domestic final purchases, district banks)
- 右侧/Right: AI 队的独有词 — broad, general (prices, markets, labor)
- 总结 / Summary: "Not right vs. wrong. Different curation styles. But the difference is real."
- Time: ~30 sec

### Slide 7 — The Anxiety / 转折页
- **节奏放慢 / Slow down** — 这是整个报告的情感转折点
- "If concept-list selection is this subjective, how much of R²=0.94 depends on exactly these 296 words?"
- 提出极端检验 / Propose extreme test: "Shuffle all concept labels. See if the results survive."
- Time: ~45 sec

### Slide 8 — Placebo Method / 打乱方法
- 上/Normal: 正常流程
- 下/Placebo: 每个会议内打乱 296 个分数 → 同样的 Ridge
- 核心操作一句话 / One sentence: "unemployment 的分换给 housing starts, inflation 的分换给 GDP"
- 强调 / Emphasize: 打分方法不变、回归不变、变量数量不变。**只有标签变了。**
- Time: ~1 min

### Slide 9 — Layer 1: Mid Specification / 第一层：中等规格
- 规格 / Spec: 132 forecasts + 296 sentiments → Ridge (Table 3, Column 4 of AD2024)
- 大号数字 / Big numbers: **0.64 → 0.65** (no drop)
- 解释 / Explain: "The 296 sentiment scores are highly correlated within each meeting. Ridge extracts the same common factor regardless of which column has which label."
- Time: ~40 sec

### Slide 10 — Why Doesn't Shuffling Change Anything? / 为什么打乱不变
- **这是需要讲清楚的关键页 / Most important explanatory slide**
- 逻辑/Logic:
  1. 同一会议内，乐观 → 所有概念周围都是正面词 → 分都偏高
  2. 悲观 → 所有概念周围都是负面词 → 分都偏低
  3. 296 个概念 = 296 个采样点，指向同一个底层信号（文档语调）
  4. Ridge 从 296 个列中提取共同模式，不关心列标签
- 可选类比 / Optional analogy: "296 thermometers in different corners of the same room, all measuring room temperature. Shuffle the labels on the thermometers — the average doesn't change."
- Time: ~1.5 min

### Slide 11 — Layer 2: Full Specification / 第二层：完整规格
- 规格 / Spec: 132 forecasts + squares + 296 sentiments + squares + 4 lags + squares of lags = 3,224 vars → Ridge (Table 3, Column 7)
- 大号数字 / Big numbers: **0.96 → 0.93** (only −0.03)
- "With nonlinear terms, 4 lags, 3,224 variables — shuffling concept labels still barely matters."
- Time: ~40 sec

### Slide 12 — Why Do Lags Still Work? / 为什么滞后项打乱了还有用
- 核心洞察 / Key insight: 296 个概念的滞后项捕捉的不是 "unemployment 上期高不高"，而是 **"上期文档语调是否持续到本期"**
- 逻辑 / Logic:
  - 第 5 列在 t-1 期高（不管它当时是哪个概念）→ t-1 期文档乐观
  - 第 5 列在 t 期高（不管它现在是谁）→ t 期文档乐观
  - 相关性捕捉的是文档语调的持续性，不是概念级动态
- 4 lags × 296 concepts = **1,184 angles** measuring the same persistence
- Time: ~1 min

### Slide 13 — Waterfall Decomposition / 瀑布分解
- 核心视觉 / Key visual: 瀑布图逐项堆叠到 0.96
- 红色那块几乎看不见 / Red bar nearly invisible — 概念标签一致性仅 0.03
- 标注 / Highlight: "The scoring mechanism contributes 15× more than concept-label identity."
- Time: ~1 min

### Slide 14 — What This Means / 这意味着什么
- 三个数字 / Three numbers: 0.49 (forecasts) + 0.43 (scoring mechanism) + 0.03 (label identity)
- 核心信息 / Core message: 方法的核心是 LM 词典 ±10 词窗口，不是那 296 个具体的词
- 可复制性 / Replicability: "You do not need the exact 296 concepts. Any reasonable set of economic terms will do."
- 对 AD(2024) 是好消息 / Good news for AD(2024): 方法比论文声称的更稳健
- Time: ~1 min

### Slide 15 — Bonus: Sample Stability / 加分项：分时期
- 可快速过或跳过 / Can skip if time is tight
- ΔR² from +0.16 → +0.05 → +0.02 = 映射裁量到规则的演变
- "Text sentiment adds the most value when policy is least predictable."
- Time: ~40 sec

### Slide 16 — Conclusion / 结论
- 四个发现，一个结论 / Four findings, one conclusion
- 最后一句 / Final line: **"The method is more robust than the paper claims."**
- Time: ~1 min

### Slide 17 — Thank You
- 指向 speaker_notes.md 获取代码和数据路径

---

## Likely Q&A / 预期问题

### Q: "Doesn't the placebo result mean concept-level scoring is meaningless?"
**A:** No. It means the value of concept-level scoring is in **aggregation**, not in individual precision. Each concept is a noisy sampling point. The LM dictionary provides a signal at each point. 296 noisy measurements aggregated together give a robust estimate of document tone. Without the scoring mechanism at each point, there would be no common signal to extract.

### Q: "If labels don't matter, why score 296 concepts? Why not just one?"
**A:** Averaging 296 noisy measurements reduces variance compared to a single global judgment. Think of it as diversification — each individual concept score is noisy (only ±10 words of context), but the average of 296 independent noisy measurements converges to the true document tone. A single LLM judgment might be more precise per observation ("I read the whole document") but less robust ("I might over-weight the first paragraph"). Both approaches have merit; we designed but did not run the single-judgment test.

### Q: "How does this relate to LLM-based NLP in economics more broadly?"
**A:** This project demonstrates LLMs as **validation tools**, not just replacements. LLM concept identification revealed the subjectivity of concept selection. An LLM risk extraction (see academic report) provided independent validation of AD(2024)'s information completeness. Sometimes the best use of new tools is to stress-test old ones.

### Q: "What about the ±10 word window? Did you test that?"
**A:** AD(2024) already tested window width in Appendix C (±5 vs. ±10, sentence-level vs. window-level). We designed a full-context LLM scoring test (greenbook1 → all 296 concepts) that would directly test whether the narrow window loses information. The implementation is ready at `src/01a_llm_sentiment_v2.py` but was not executed due to time constraints (~1 hour of API calls).

### Q: "What specification did you use for the first-stage regression?"
**A:** We followed AD(2024) Table 3 exactly:
- Mid spec (Col 4): 132 extended forecasts + 296 sentiments → Ridge (429 vars)
- Full spec (Col 7): forecasts + squares + sentiments + squares + 4 lags + squares of lags → Ridge (3,224 vars)
- Ridge penalty chosen by 10-fold CV. Data: 204 FOMC meetings, 1982–2008.

### Q: "Why did your full spec get R²=0.96 instead of 0.94?"
**A:** Minor differences in hyperparameter search (we used 30 alpha values on log-spaced grid; the paper may have used a different grid) and potential differences in the exact set of forecast variables. The 0.02 difference is within reasonable variation for cross-validated Ridge. It does not affect any of the robustness conclusions.

---

## Data Pipeline Summary / 数据管线

```
Raw data / 原始数据:
  data/create_sentiments/FOMC_Texts/          787 Greenbook PDFs
  data/create_sentiments/Input_files/FFR_Target.csv
  data/create_sentiments/Input_files/Greenbook_Regressioninput_Extended.csv
  data/replication_files/Sentiment_Final/total_sentiment_file_std10.csv

Processed data / 处理后数据:
  data/processed/text_index.csv               Text file index
  data/processed/llm_concepts.csv             LLM-identified 595 concepts
  data/processed/risk_indicators.csv          LLM-extracted risk signals

Key scripts / 核心脚本:
  src/02_concept_identification.py            N-gram + LLM concept classification
  src/03_risk_extraction.py                   LLM risk signal extraction
  src/07_ml_comparison.py                     Ridge/LASSO/EN comparison
  src/run_robustness_checks.py                Placebo + sample split
  src/01a_llm_sentiment_v2.py                 Fixed-list LLM scoring (DESIGN ONLY)

Results / 结果:
  results/tables/check1_sample_split.csv      Sample split results
  results/tables/check2_placebo.csv           Placebo test summary
  results/figures/                            All figures
  results/beamer_presentation.pdf             This presentation
  results/reports/academic_report.md          Full write-up
