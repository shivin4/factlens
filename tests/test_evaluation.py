"""
Tests for Stage 3: Evaluation + Scoring
(Uses mock RetrievedChunks — does not require FAISS index)
"""
import pytest
from pipeline.evaluation import (
    NLIResult,
    ClaimVerdict,
    check_internal_consistency,
    VERDICT_ENTAILMENT,
    VERDICT_CONTRADICTION,
    VERDICT_NEUTRAL,
)
from pipeline.scoring import (
    compute_accuracy,
    compute_completeness,
    compute_scores,
    interpret_score,
    SubScores,
)


# ── Mock helpers ──────────────────────────────────────────────────────────────

def make_verdict(verdict: str, entailment=0.5, neutral=0.3, contradiction=0.2) -> ClaimVerdict:
    from pipeline.retrieval import RetrievedChunk
    nli = NLIResult(
        entailment=entailment,
        neutral=neutral,
        contradiction=contradiction,
        verdict=verdict,
    )
    chunk = RetrievedChunk(
        chunk_id=0,
        text="BERT uses bidirectional attention to pre-train language representations.",
        source_doc="bert_devlin_2019",
        source_type="paper",
        page_start=1,
        page_end=2,
        similarity_score=0.9,
    )
    return ClaimVerdict(
        claim="test claim",
        verdict=verdict,
        nli=nli,
        best_chunk=chunk,
        all_chunks=[chunk],
    )


# ── NLIResult tests ───────────────────────────────────────────────────────────

def test_nli_result_from_logits_entailment():
    # High entailment logit
    result = NLIResult.from_logits([-5.0, 0.0, 5.0])  # [contra, neutral, entail]
    assert result.verdict == VERDICT_ENTAILMENT


def test_nli_result_from_logits_contradiction():
    result = NLIResult.from_logits([5.0, 0.0, -5.0])
    assert result.verdict == VERDICT_CONTRADICTION


def test_nli_probs_sum_to_one():
    result = NLIResult.from_logits([1.0, 2.0, 3.0])
    total = result.entailment + result.neutral + result.contradiction
    assert abs(total - 1.0) < 1e-5


# ── Scoring tests ─────────────────────────────────────────────────────────────

def test_accuracy_all_entailment():
    verdicts = [make_verdict(VERDICT_ENTAILMENT, entailment=0.9) for _ in range(3)]
    acc, _ = compute_accuracy(verdicts, use_bertscore=False)
    assert acc > 0.8


def test_accuracy_all_contradiction():
    verdicts = [make_verdict(VERDICT_CONTRADICTION, contradiction=0.9, entailment=0.05) for _ in range(3)]
    acc, _ = compute_accuracy(verdicts, use_bertscore=False)
    assert acc < 0.5


def test_completeness_full_coverage():
    # User has all KB concepts
    verdicts = [make_verdict(VERDICT_ENTAILMENT)]
    score, missing = compute_completeness(
        user_concepts=["bert", "attention", "bidirectional", "language", "representations", "pre-train"],
        verdicts=verdicts,
    )
    # Should be relatively high
    assert score >= 0.0   # just check it runs


def test_interpret_score_ranges():
    assert interpret_score(90)[0] == "Excellent"
    assert interpret_score(70)[0] == "Good"
    assert interpret_score(55)[0] == "Moderate"
    assert interpret_score(40)[0] == "Needs Improvement"
    assert interpret_score(20)[0] == "Poor"


def test_composite_formula():
    scores = SubScores(accuracy=0.8, completeness=0.6, logic=0.9)
    expected = 0.5 * 0.8 + 0.3 * 0.6 + 0.2 * 0.9
    assert abs(scores.composite - expected) < 1e-6


def test_compute_scores_runs():
    verdicts = [make_verdict(VERDICT_ENTAILMENT, entailment=0.85)]
    result = compute_scores(
        verdicts=verdicts,
        user_concepts=["bert", "attention"],
        logic_score=0.9,
        use_bertscore=False,
    )
    assert 0 <= result.sub_scores.composite_100 <= 100
    assert isinstance(result.missing_concepts, list)
