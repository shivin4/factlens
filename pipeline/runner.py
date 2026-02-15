"""
FactLens — Pipeline Runner
===========================
Orchestrates all three stages for a single user explanation.
Convenience wrapper used by the Streamlit app.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict

from pipeline.linguistic import analyze, LinguisticOutput
from pipeline.retrieval import retrieve_batch, RetrievedChunk
from pipeline.evaluation import evaluate_all, check_internal_consistency, ClaimVerdict
from pipeline.scoring import compute_scores, ScoringResult, interpret_score


@dataclass
class FactLensResult:
    """Complete evaluation result for one user explanation."""
    linguistic: LinguisticOutput
    verdicts: list[ClaimVerdict]
    scoring: ScoringResult
    elapsed_seconds: float

    def score_label(self) -> tuple[str, str]:
        return interpret_score(self.scoring.sub_scores.composite_100)

    def to_dict(self) -> dict:
        return {
            "linguistic": self.linguistic.to_dict(),
            "verdicts": [v.to_dict() for v in self.verdicts],
            "scoring": self.scoring.to_dict(),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "score_label": self.score_label()[0],
        }


def run_pipeline(
    user_text: str,
    use_bertscore: bool = True,
    top_k: int = 5,
) -> FactLensResult:
    """
    Full FactLens pipeline:
      Stage 1 → Linguistic analysis
      Stage 2 → Neural retrieval
      Stage 3 → NLI evaluation + scoring

    Args:
        user_text:     Raw explanation text from the user
        use_bertscore: Whether to include BERTScore in accuracy sub-score
        top_k:         Number of KB chunks to retrieve per claim

    Returns:
        FactLensResult with full verdicts and scores
    """
    t0 = time.time()

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    print("\n[Stage 1] Linguistic analysis...")
    linguistic_out = analyze(user_text)
    claims_text = [c.as_sentence() for c in linguistic_out.claims]

    print(f"  → {len(linguistic_out.sentences)} sentences")
    print(f"  → {len(linguistic_out.concepts)} concepts")
    print(f"  → {len(claims_text)} claims extracted")

    if not claims_text:
        print("  [WARN] No claims extracted. Using raw sentences as claims.")
        claims_text = linguistic_out.sentences[:10]   # fallback

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    print("\n[Stage 2] Neural retrieval...")
    retrieved: list[list[RetrievedChunk]] = retrieve_batch(claims_text, top_k=top_k)
    total_chunks = sum(len(r) for r in retrieved)
    print(f"  → Retrieved {total_chunks} chunks across {len(claims_text)} claims")

    # ── Stage 3a: NLI evaluation ──────────────────────────────────────────────
    print("\n[Stage 3] NLI evaluation...")
    verdicts = evaluate_all(claims_text, retrieved)

    # ── Stage 3b: Internal consistency ────────────────────────────────────────
    print("[Stage 3] Checking internal consistency...")
    logic_score = check_internal_consistency(claims_text)
    print(f"  → Logic score: {logic_score:.3f}")

    # ── Stage 3c: Composite scoring ───────────────────────────────────────────
    print("[Stage 3] Computing composite scores...")
    scoring = compute_scores(
        verdicts=verdicts,
        user_concepts=linguistic_out.concepts,
        logic_score=logic_score,
        use_bertscore=use_bertscore,
    )

    elapsed = time.time() - t0
    print(f"\n✅ Pipeline complete in {elapsed:.1f}s")
    print(f"   Composite score: {scoring.sub_scores.composite_100}/100")

    return FactLensResult(
        linguistic=linguistic_out,
        verdicts=verdicts,
        scoring=scoring,
        elapsed_seconds=elapsed,
    )


def run_ablation_study(
    user_text: str,
    pipeline_result: FactLensResult,
    top_k: int = 5,
) -> dict:
    """
    Runs the ablation study baselines:
    1. Baseline 1 (No NLP Pipeline): Raw user text -> NLI/BERTScore.
    2. Baseline 2 (No Advanced Models): FactLens Pipeline -> Lexical Scoring.
    """
    from copy import deepcopy
    from pipeline.scoring import compute_lexical_accuracy, compute_cosine_accuracy

    print("\n[Ablation] Running Baseline 1: No NLP Pipeline (Raw Text)")
    
    # Baseline 1: Raw Text
    # We treat the entire user_text as one single claim.
    raw_claims = [user_text]
    raw_retrieved = retrieve_batch(raw_claims, top_k=top_k)
    raw_verdicts = evaluate_all(raw_claims, raw_retrieved)
    raw_logic_score = 1.0  # Only one claim, logic is 1.0
    
    # Score it
    raw_scoring = compute_scores(
        verdicts=raw_verdicts,
        user_concepts=pipeline_result.linguistic.concepts,
        logic_score=raw_logic_score,
        use_bertscore=True,
    )

    # Baseline 2: No Advanced Models (Lexical Only)
    print("[Ablation] Running Baseline 2: No Advanced Models (Lexical Only)")
    
    # We reuse the verdicts from the pipeline, but we recalculate the accuracy
    # using ONLY the lexical overlap, overriding the NLI/BERTScore.
    lexical_accuracy = compute_lexical_accuracy(pipeline_result.verdicts)
    
    # Baseline 3: No Advanced Models (Cosine Similarity)
    print("[Ablation] Running Baseline 3: No Advanced Models (Cosine Similarity)")
    cosine_accuracy = compute_cosine_accuracy(pipeline_result.verdicts)
    
    return {
        "baseline_raw_text": {
            "accuracy": raw_scoring.sub_scores.accuracy_100,
            "composite": raw_scoring.sub_scores.composite_100,
        },
        "baseline_lexical": {
            "accuracy": round(lexical_accuracy * 100, 1),
        },
        "baseline_cosine": {
            "accuracy": round(cosine_accuracy * 100, 1),
        }
    }
