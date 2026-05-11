"""
Shared DeepSeek API cache monitoring utilities + text pre-filtering helpers.

DeepSeek prefix caching is automatic — no code changes needed to enable it.
These utilities let you measure whether it's working.

The key rule for cache hits:
  System prompt must be byte-for-byte identical across all calls in a session.
  Variable content (meeting date, document text) goes in the user message only.

Usage:
    from _cache_utils import CacheTracker, filter_risk_paragraphs

    tracker = CacheTracker()
    response = client.chat.completions.create(...)
    stats = tracker.update(response)
    print(f"  cache: {stats['hit']:,} hit / {stats['miss']:,} miss ({stats['rate_pct']:.0f}%)")
    ...
    tracker.print_summary("Step 4 — Risk Extraction")
"""

from dataclasses import dataclass


@dataclass
class CacheTracker:
    """Accumulates cache hit/miss stats across all API calls in a script run."""
    hit_tokens: int = 0
    miss_tokens: int = 0
    total_calls: int = 0
    completion_tokens: int = 0

    def update(self, response) -> dict:
        """Read cache stats from a DeepSeek response and add to running totals.

        Returns per-call dict: {"hit": int, "miss": int, "rate_pct": float}
        Log this inline so you can see per-meeting cache performance.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"hit": 0, "miss": 0, "rate_pct": 0.0}

        hit  = getattr(usage, "prompt_cache_hit_tokens",  0) or 0
        miss = getattr(usage, "prompt_cache_miss_tokens", 0) or 0
        comp = getattr(usage, "completion_tokens",         0) or 0

        self.hit_tokens        += hit
        self.miss_tokens       += miss
        self.completion_tokens += comp
        self.total_calls       += 1

        total = hit + miss
        rate  = hit / total * 100 if total > 0 else 0.0
        return {"hit": hit, "miss": miss, "rate_pct": rate}

    @property
    def cumulative_rate(self) -> float:
        total = self.hit_tokens + self.miss_tokens
        return self.hit_tokens / total * 100 if total > 0 else 0.0

    @property
    def estimated_saving_usd(self) -> float:
        """Estimated USD saved vs paying full price for all input tokens.

        Rates (deepseek-chat / deepseek-v4-pro):
          Full price:       $0.14 / M tokens
          Cache hit price:  $0.014 / M tokens  (10x cheaper)
        """
        full_price   = (self.hit_tokens + self.miss_tokens) / 1_000_000 * 0.14
        actual_price = (self.hit_tokens  / 1_000_000 * 0.014
                        + self.miss_tokens / 1_000_000 * 0.14)
        return full_price - actual_price

    def print_summary(self, label: str = "Session"):
        total_in = self.hit_tokens + self.miss_tokens
        print(f"\n{'='*52}")
        print(f"  Cache summary — {label}")
        print(f"  API calls:         {self.total_calls}")
        print(f"  Hit tokens:        {self.hit_tokens:>10,}  ({self.cumulative_rate:.1f}%)")
        print(f"  Miss tokens:       {self.miss_tokens:>10,}")
        print(f"  Total input:       {total_in:>10,}")
        print(f"  Completion tokens: {self.completion_tokens:>10,}")
        print(f"  Est. saving:       ${self.estimated_saving_usd:.4f} USD")
        print(f"{'='*52}\n")


# ---------------------------------------------------------------------------
# Text pre-filtering for risk extraction (Step 4 only)
# ---------------------------------------------------------------------------
#
# Two-tier strategy:
#
#   Tier 1 — STRONG signals: directional/structural risk phrases that are
#   highly specific to risk-discussion paragraphs. These work well in all
#   decades (1982–2016).
#
#   Tier 2 — BROADER fallback: adds plain "risk" and "could/might" patterns.
#   Used only when Tier 1 returns fewer than MIN_RETAIN_FRAC of the document,
#   which flags early Greenbooks (pre-1990) that used less standardised language.
#
# In both cases ±1 neighbour paragraphs are included for coherence.
# No hard character cap — all matched paragraphs are sent to the LLM.

import re as _re

# Minimum fraction of paragraphs to retain before triggering Tier 2 fallback.
_MIN_RETAIN_FRAC = 0.15   # if Tier 1 keeps < 15 % of paras, try Tier 2
# Minimum paragraph length to filter out page headers / blank lines.
_MIN_PARA_CHARS  = 80

_TIER1 = [
    # Directional risk terms (specific enough to avoid noise)
    r"\bupside\b",
    r"\bdownside\b",
    r"\buncertaint",            # uncertainty / uncertainties
    r"\basymmetr",              # asymmetric / asymmetry
    r"balance of risk",
    r"risks? to (?:the )?(?:outlook|forecast|projection|growth|inflation|employment|activity)",
    r"risks? (?:are|remain|appear|seem) (?:tilted|skewed|balanced|elevated|weighted)",
    r"risks? on (?:both )?sides",
    r"alternative (?:scenario|forecast|path|projection)",
    r"\bvulnerab",              # vulnerable / vulnerability
    r"\bcontingenc",            # contingency / contingencies
    r"\btail risk",
    r"\bskewed\b",              # skewed (not "Secretariat")
    # "higher/lower than expected/projected" — common in all eras
    r"(?:higher|lower|stronger|weaker) than (?:expected|projected|forecast|anticipated)",
    # "could be higher/lower" — explicit counterfactual
    r"could (?:be |come in )?(?:higher|lower|stronger|weaker|above|below)",
]

_TIER2 = _TIER1 + [
    # Add broader matches only used as fallback for early-era documents
    r"\brisk\b",                # standalone "risk" (noisy but needed pre-1990)
    r"\bscenario\b",            # any scenario reference
    r"might (?:exceed|fall short|be (?:higher|lower))",
    r"(?:higher|lower|exceed|fall short) (?:than )?(?:we |the staff )?(?:expect|project|anticipat|forecast)",
]


def _apply_keywords(paragraphs: list, patterns: list) -> set:
    """Return index set of paragraphs matching any pattern, plus ±1 neighbours."""
    matching = set()
    compiled = [_re.compile(p) for p in patterns]
    for i, para in enumerate(paragraphs):
        pl = para.lower()
        if any(pat.search(pl) for pat in compiled):
            matching.add(i)
            if i > 0:
                matching.add(i - 1)
            if i < len(paragraphs) - 1:
                matching.add(i + 1)
    return matching


def filter_risk_paragraphs(text: str) -> tuple:
    """Return only risk-relevant paragraphs from a Greenbook document.

    Uses a two-tier keyword strategy so early-era documents (pre-1990) that
    lack standardised "upside/downside risk" language are handled gracefully.

    Returns:
        (filtered_text: str, stats: dict)

        stats keys:
          original_words, filtered_words, paragraphs_total, paragraphs_kept,
          reduction_pct, tier_used (1 or 2), fallback (bool — full text used)
    """
    paragraphs = [p.strip() for p in text.split("\n\n")
                  if len(p.strip()) >= _MIN_PARA_CHARS]
    original_words = len(text.split())

    if not paragraphs:
        return text, _stats(original_words, original_words, 0, 0, tier=0, fallback=True)

    # --- Tier 1 ---
    matching = _apply_keywords(paragraphs, _TIER1)
    tier = 1

    # --- Tier 2 fallback if Tier 1 is too sparse ---
    if len(matching) < _MIN_RETAIN_FRAC * len(paragraphs):
        matching = _apply_keywords(paragraphs, _TIER2)
        tier = 2

    # --- Final fallback: full text ---
    if not matching:
        return text, _stats(original_words, original_words,
                            len(paragraphs), len(paragraphs), tier=0, fallback=True)

    selected = [paragraphs[i] for i in sorted(matching)]
    filtered_text = "\n\n".join(selected)
    filtered_words = len(filtered_text.split())

    return filtered_text, _stats(original_words, filtered_words,
                                 len(paragraphs), len(selected), tier=tier)


def _stats(orig_w, filt_w, total_p, kept_p, tier=1, fallback=False):
    reduction = (1 - filt_w / orig_w) * 100 if orig_w > 0 else 0.0
    return {
        "original_words":  orig_w,
        "filtered_words":  filt_w,
        "paragraphs_total": total_p,
        "paragraphs_kept":  kept_p,
        "reduction_pct":   reduction,
        "tier_used":       tier,
        "fallback":        fallback,
    }
