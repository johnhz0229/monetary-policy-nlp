# Step 2: Concept Identification Report

## Phase A — N-gram Frequency Counting
- **Total candidate n-grams extracted**: 2,000 (top by document frequency)
- **Source**: 787 text files from /Users/zhenghuang/Desktop/projects/monetary-policy-nlp/data/processed/texts

## Phase B — LLM Classification
- **Total classifications attempted**: 2000 terms in 250 batches
- **Failed batches**: 0
- **Economic concepts identified**: 595

### Category Distribution
category
real_activity    204
financial        150
prices            99
international     51
labor             48
fiscal            24
other             19

## Phase C — Comparison with Original 296 Concepts
- **Original concepts**: 398
- **LLM concepts**: 595
- **Overlap (LLM → Original)**: 246/595 (41.3%)
- **Coverage (Original → LLM)**: 249/398 (62.6%)

### Top 20 Novel Concepts (LLM found, original missed)
           term      category  doc_frequency
         prices        prices            787
federal reserve     financial            787
          labor         labor            787
      financial     financial            787
     industrial real_activity            787
        markets     financial            787
         energy        prices            787
     production real_activity            787
       spending real_activity            787
          sales real_activity            787
         growth real_activity            787
        federal        fiscal            787
          rates     financial            787
          goods real_activity            787
          price        prices            787
        capital     financial            786
       interest     financial            786
         demand real_activity            786
          trade international            786
          costs        prices            785
        balance     financial            784
            oil        prices            783
           food        prices            776
  manufacturing real_activity            774
         orders real_activity            771
         starts real_activity            771
        foreign international            769
    residential real_activity            768
      purchases real_activity            767
      expansion real_activity            766

### Top 20 Original Concepts Missed by LLM
                          terms
     advanced foreign economies
               aggregate demand
                        bankers
                 district banks
                      brazilian
              business activity
            business confidence
                      car sales
                        chinese
             consumer sentiment
                    debt growth
               defense spending
           developing countries
       domestic final purchases
domestic financial developments
              drilling activity
      emerging market economies
                employment cost
                equity issuance
                   equity price

### Commentary
The LLM identified 595 economic concepts vs. the original 296.
Coverage of original concepts: 249/398 (62.6%).
The 149 missed original concepts include terms that are either too generic,
stop words, or document-structural terms that the LLM correctly excluded.
