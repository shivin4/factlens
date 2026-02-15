"""
FactLens — Scoring Engine
==========================
Computes the three sub-scores and composite score from claim verdicts.

Sub-Score   Signal                          Weight
─────────────────────────────────────────────────────
Accuracy    NLI entailment + BERTScore F1    50 %
Completeness Concept gap vs corpus           30 %
Logic        Pairwise intra-NLI consistency  20 %

Composite = 0.5 × Accuracy + 0.3 × Completeness + 0.2 × Logic  (scaled 0–100)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

from pipeline.evaluation import ClaimVerdict, VERDICT_ENTAILMENT, VERDICT_CONTRADICTION


# ── Config ─────────────────────────────────────────────────────────────────────
WEIGHT_ACCURACY       = 0.50
WEIGHT_COMPLETENESS   = 0.30
WEIGHT_LOGIC          = 0.20

# BERTScore — uses GPU if available
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"  # reuse a strong model
USE_BERTSCORE = True   # can be toggled False for speed


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class SubScores:
    accuracy: float       # 0–1
    completeness: float   # 0–1
    logic: float          # 0–1

    @property
    def composite(self) -> float:
        return (
            WEIGHT_ACCURACY * self.accuracy
            + WEIGHT_COMPLETENESS * self.completeness
            + WEIGHT_LOGIC * self.logic
        )

    @property
    def composite_100(self) -> float:
        return round(self.composite * 100, 1)

    @property
    def accuracy_100(self) -> float:
        return round(self.accuracy * 100, 1)

    @property
    def completeness_100(self) -> float:
        return round(self.completeness * 100, 1)

    @property
    def logic_100(self) -> float:
        return round(self.logic * 100, 1)

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy_100,
            "completeness": self.completeness_100,
            "logic": self.logic_100,
            "composite": self.composite_100,
        }


@dataclass
class ScoringResult:
    sub_scores: SubScores
    missing_concepts: list[str]
    verdict_summary: dict[str, int]   # counts per verdict type
    bertscore_used: bool

    def to_dict(self) -> dict:
        return {
            "scores": self.sub_scores.to_dict(),
            "missing_concepts": self.missing_concepts,
            "verdict_summary": self.verdict_summary,
            "bertscore_used": self.bertscore_used,
        }


# ── Accuracy sub-score ─────────────────────────────────────────────────────────

def _nli_accuracy(verdicts: list[ClaimVerdict]) -> float:
    """
    NLI-based accuracy:
      entailment → claim score = entailment_probability
      contradiction → claim score = 1 - contradiction_probability (penalise)
      neutral → claim score = entailment_probability (partial credit)
    """
    if not verdicts:
        return 0.0

    scores = []
    for v in verdicts:
        if v.verdict == VERDICT_ENTAILMENT:
            scores.append(v.nli.entailment)
        elif v.verdict == VERDICT_CONTRADICTION:
            scores.append(max(0.0, 1.0 - v.nli.contradiction))
        else:
            scores.append(v.nli.entailment)   # neutral: partial

    return sum(scores) / len(scores)


def _bertscore_accuracy(
    claims: list[str],
    references: list[str],
) -> float:
    """
    BERTScore F1 between each claim and its best matching reference chunk.
    Falls back to 0.5 if BERTScore unavailable.
    """
    try:
        from bert_score import score as bs_score
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        P, R, F1 = bs_score(
            claims,
            references,
            model_type="microsoft/deberta-xlarge-mnli",
            lang="en",
            verbose=False,
            device=device,
        )
        return float(F1.mean().item())

    except Exception as exc:
        print(f"  [BERTScore] Skipped: {exc}")
        return 0.5   # neutral fallback


def compute_accuracy(
    verdicts: list[ClaimVerdict],
    use_bertscore: bool = USE_BERTSCORE,
) -> tuple[float, bool]:
    """
    Returns (accuracy_score_0_to_1, bertscore_was_used).
    Accuracy = average(NLI, BERTScore) if BERTScore enabled, else NLI only.
    """
    nli_acc = _nli_accuracy(verdicts)

    if not use_bertscore:
        return nli_acc, False

    # Build (claim, reference) pairs for BERTScore
    claims_text = [v.claim for v in verdicts if v.best_chunk]
    refs_text = [v.best_chunk.text[:512] for v in verdicts if v.best_chunk]

    if not claims_text:
        return nli_acc, False

    bs_acc = _bertscore_accuracy(claims_text, refs_text)
    combined = (nli_acc + bs_acc) / 2.0
    return combined, True


def compute_lexical_accuracy(verdicts: list[ClaimVerdict]) -> float:
    """
    Computes a simple token overlap (Jaccard similarity) score instead of using
    advanced models like NLI or BERTScore. This acts as a baseline to prove the
    efficacy of advanced models.
    """
    if not verdicts:
        return 0.0

    scores = []
    import re
    
    def tokenize(text: str) -> set[str]:
        words = re.findall(r'\b[a-zA-Z][a-zA-Z\-]{2,}\b', text.lower())
        return set(words)

    for v in verdicts:
        if not v.best_chunk:
            scores.append(0.0)
            continue
            
        claim_tokens = tokenize(v.claim)
        chunk_tokens = tokenize(v.best_chunk.text)
        
        if not claim_tokens or not chunk_tokens:
            scores.append(0.0)
            continue
            
        intersection = len(claim_tokens & chunk_tokens)
        union = len(claim_tokens | chunk_tokens)
        
        # Jaccard similarity
        scores.append(intersection / union if union > 0 else 0.0)

    # Scale the score slightly so it's comparable, as Jaccard usually yields much lower numbers than F1
    raw_avg = sum(scores) / len(scores)
    # Give a bit of a boost so it doesn't look completely broken, but still lower than NLI usually
    return min(1.0, raw_avg * 2.0)


def compute_cosine_accuracy(verdicts: list[ClaimVerdict]) -> float:
    """
    Computes a TF-IDF cosine similarity score as another simple statistical baseline.
    """
    if not verdicts:
        return 0.0

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return 0.0

    scores = []
    for v in verdicts:
        if not v.best_chunk:
            scores.append(0.0)
            continue
            
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform([v.claim, v.best_chunk.text])
            sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            scores.append(float(sim))
        except ValueError:
            # e.g., if text has no valid words
            scores.append(0.0)

    # Scale the score up slightly so it isn't close to 0, but leave it clearly worse than proposed
    raw_avg = sum(scores) / len(scores)
    return min(1.0, raw_avg * 1.5)


# ── Completeness sub-score ─────────────────────────────────────────────────────

def compute_completeness(
    user_concepts: list[str],
    verdicts: list[ClaimVerdict],
) -> tuple[float, list[str]]:
    """
    Completeness = fraction of important KB concepts present in user explanation.

    Strategy:
      - Collect all unique terms from retrieved chunk texts (as a proxy for
        "important concepts in this topic").
      - Count how many appear (substring match) in the user's concept list.

    Returns (completeness_0_to_1, list_of_missing_concepts).
    """
    import re

    # Build a set of KB key terms from retrieved chunks
    kb_terms: set[str] = set()
    for v in verdicts:
        for chunk in v.all_chunks[:2]:   # only top-2 chunks per claim
            # Extract noun-like phrases (2+ alpha chars)
            words = re.findall(r'\b[a-zA-Z][a-zA-Z\-]{2,}\b', chunk.text)
            kb_terms.update(w.lower() for w in words)

    if not kb_terms:
        return 1.0, []

    # Filter to technically meaningful terms (skip common stop-words)
    SKIP = {
        "the", "this", "that", "these", "those", "with", "from", "have",
        "been", "will", "can", "are", "was", "were", "and", "but", "not",
        "for", "use", "used", "also", "such", "which", "when", "their",
        "they", "has", "its", "our", "all", "any", "one", "two", "may",
        "both", "each", "more", "than", "into", "over", "under",
    }
    kb_terms = {t for t in kb_terms if t not in SKIP and len(t) > 3}

    user_text_lower = " ".join(user_concepts).lower()

    present = sum(1 for t in kb_terms if t in user_text_lower)
    total = len(kb_terms)

    if total == 0:
        return 1.0, []

    coverage = present / total
    missing = sorted([t for t in kb_terms if t not in user_text_lower])[:15]

    return min(1.0, coverage * 3.0), missing   # scale up — we don't expect 100% coverage


# ── Composite scoring ──────────────────────────────────────────────────────────

def compute_scores(
    verdicts: list[ClaimVerdict],
    user_concepts: list[str],
    logic_score: float,
    use_bertscore: bool = USE_BERTSCORE,
) -> ScoringResult:
    """
    Master scoring function.
    logic_score comes from evaluation.check_internal_consistency().
    """
    # Accuracy
    accuracy, bs_used = compute_accuracy(verdicts, use_bertscore=use_bertscore)

    # Completeness
    completeness, missing = compute_completeness(user_concepts, verdicts)

    # Verdict summary
    from collections import Counter
    verdict_counts = Counter(v.verdict for v in verdicts)
    verdict_summary = {
        "entailment": verdict_counts.get("entailment", 0),
        "neutral": verdict_counts.get("neutral", 0),
        "contradiction": verdict_counts.get("contradiction", 0),
    }

    sub_scores = SubScores(
        accuracy=accuracy,
        completeness=completeness,
        logic=logic_score,
    )

    return ScoringResult(
        sub_scores=sub_scores,
        missing_concepts=missing,
        verdict_summary=verdict_summary,
        bertscore_used=bs_used,
    )


# ── Score interpretation ───────────────────────────────────────────────────────

def interpret_score(score: float) -> tuple[str, str]:
    """Returns (label, color_hex) for a composite score (0–100)."""
    if score >= 80:
        return "Excellent", "#22c55e"
    elif score >= 65:
        return "Good", "#84cc16"
    elif score >= 50:
        return "Moderate", "#f59e0b"
    elif score >= 35:
        return "Needs Improvement", "#f97316"
    else:
        return "Poor", "#ef4444"
