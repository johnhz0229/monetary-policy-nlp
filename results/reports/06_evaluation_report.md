# Step 6: Improvement Evaluation Report

## Final Comparison Table (R²)
horizon                              Current Q  1Q ahead  1Y ahead  2Y ahead
spec                                                                        
Baseline (orig PC1)                     0.0445    0.1484    0.2473    0.2062
C1-LayerA (LLM score, 217 concepts)     0.0000    0.0227    0.0180    0.0000
C1-LayerB (LLM score, 595 concepts)     0.0001    0.0001    0.0000    0.0000
C2-Combined (PC1 + risk)                0.0894    0.2179    0.3142    0.2395
C2-Risk (unemp risk score)              0.0061    0.0947    0.1809    0.1825

## Best Method by Horizon
- **Current Q**: Best = C2-Combined (PC1 + risk) (R²=0.089)
- **1Q ahead**: Best = C2-Combined (PC1 + risk) (R²=0.218)
- **1Y ahead**: Best = C2-Combined (PC1 + risk) (R²=0.314)
- **2Y ahead**: Best = C2-Combined (PC1 + risk) (R²=0.239)


## Baseline Verification
- Our baseline 1Y R²: **0.2473** (paper: 0.248)

## Contribution 1: LLM NLP Features
### Layer 1A — LLM Scoring on Original Concepts
Uses LLM sentiment scoring on the 217 concepts that overlap with the original 296.
Compares directly to baseline (same concepts, different scoring method).

### Layer 1B — LLM Scoring on LLM-Identified Concepts
Uses LLM sentiment scoring on all 595 LLM-identified concepts.
Tests whether LLM concept identification adds value beyond LLM scoring alone.

## Contribution 2: Risk Asymmetry (Direct Test of Modal-Forecast Mechanism)
### F-test for Incremental R²
- **Current Q**: ΔR²=0.01, F=1.5976, p=0.2083
- **1Q ahead**: ΔR²=0.0029, F=0.5372, p=0.4648
- **1Y ahead**: ΔR²=0.0203, F=4.3248**, p=0.0393
- **2Y ahead**: ΔR²=0.0514, F=3.3778*, p=0.072


## Sample Unemployment Risk Evidence Quotes
- **2003-01-28** (upside, medium): "firms clearly remain reluctant to add new workers"
- **1990-03-27** (balanced, low): "the staff has partially discounted the labor market data"
- **1987-08-18** (downside, medium): "the unemployment rate were to remain on the downward course seen in recent months"
- **2004-03-16** (balanced, medium): "4.9–6.1"
- **2000-03-21** (downside, medium): "pushes the unemployment rate down almost to 3-1/2 percent"

## Limitations
- Layer 1A only covers 217 of the ~296 original concepts (LLM free-naming + fuzzy matching missed some)
- LLM sentiment matrix is sparse (595 concepts but only ~5-40 non-zero per meeting)
- Risk score encoding (categorical→continuous) involves judgment calls
- Sample limited to 1982-2008 (210 observations with pc1_std)
