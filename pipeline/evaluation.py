"""
FactLens — Stage 3: Evaluation Engine (NLI)
=============================================
Input : List of (claim_str, [RetrievedChunk, ...]) pairs
Output: List of ClaimVerdict — per-claim verdict with NLI scores & citations

Model: cross-encoder/nli-deberta-v3-large
  Returns logits for [contradiction, neutral, entailment]
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

from sentence_transformers import CrossEncoder

from pipeline.retrieval import RetrievedChunk

# ── Model ──────────────────────────────────────────────────────────────────────
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-large"
_nli_model: Optional[CrossEncoder] = None


def _get_nli_model() -> CrossEncoder:
    global _nli_model
    if _nli_model is None:
        print(f"[Evaluation] Loading NLI model: {NLI_MODEL_NAME} ...")
        _nli_model = CrossEncoder(NLI_MODEL_NAME)
    return _nli_model


# ── Data types ─────────────────────────────────────────────────────────────────

VERDICT_ENTAILMENT = "entailment"
VERDICT_NEUTRAL    = "neutral"
VERDICT_CONTRADICTION = "contradiction"

VERDICT_EMOJI = {
    VERDICT_ENTAILMENT:    "✅",
    VERDICT_NEUTRAL:       "⚠️",
    VERDICT_CONTRADICTION: "❌",
}


@dataclass
class NLIResult:
    entailment: float
    neutral: float
    contradiction: float
    verdict: str   # one of the VERDICT_* constants

    @classmethod
    def from_logits(cls, logits: list[float]) -> "NLIResult":
        import torch
        import torch.nn.functional as F
        probs = F.softmax(
            torch.tensor(logits, dtype=torch.float32), dim=-1
        ).tolist()
        # DeBERTa NLI label order: [contradiction, neutral, entailment]
        contradiction, neutral, entailment = probs
        verdict_idx = probs.index(max(probs))
        verdict = [VERDICT_CONTRADICTION, VERDICT_NEUTRAL, VERDICT_ENTAILMENT][verdict_idx]
        return cls(
            entailment=entailment,
            neutral=neutral,
            contradiction=contradiction,
            verdict=verdict,
        )


@dataclass
class ClaimVerdict:
    claim: str
    verdict: str
    nli: NLIResult
    best_chunk: Optional[RetrievedChunk]
    all_chunks: list[RetrievedChunk] = field(default_factory=list)

    def citation(self) -> str:
        return self.best_chunk.citation() if self.best_chunk else "No source found"

    def excerpt(self) -> str:
        return self.best_chunk.short_excerpt(300) if self.best_chunk else ""

    def emoji(self) -> str:
        return VERDICT_EMOJI.get(self.verdict, "❓")

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "verdict": self.verdict,
            "emoji": self.emoji(),
            "nli_scores": asdict(self.nli),
            "citation": self.citation(),
            "excerpt": self.excerpt(),
        }


# ── Core evaluation ────────────────────────────────────────────────────────────

def _score_pair(claim: str, chunk_text: str) -> NLIResult:
    """Run NLI cross-encoder on a single (claim, premise) pair."""
    model = _get_nli_model()
    logits = model.predict([(claim, chunk_text)], apply_softmax=False)[0]
    return NLIResult.from_logits(logits.tolist())


def evaluate_claim(claim: str, chunks: list[RetrievedChunk]) -> ClaimVerdict:
    """
    Evaluate a single claim against its retrieved chunks.

    Strategy:
      1. Score each (claim, chunk) pair with NLI.
      2. Select best chunk by entailment score.
      3. If best entailment is still low, fall back to the contradiction/neutral verdict.
    """
    if not chunks:
        # No KB context found — mark as neutral/unsupported
        neutral_nli = NLIResult(
            entailment=0.0, neutral=1.0, contradiction=0.0,
            verdict=VERDICT_NEUTRAL
        )
        return ClaimVerdict(
            claim=claim,
            verdict=VERDICT_NEUTRAL,
            nli=neutral_nli,
            best_chunk=None,
            all_chunks=[],
        )

    model = _get_nli_model()

    # Build all (claim, chunk_text) pairs for this claim
    pairs = [(claim, chunk.text) for chunk in chunks]

    # Batch predict for efficiency
    logits_batch = model.predict(pairs, apply_softmax=False)
    nli_results = [NLIResult.from_logits(lg.tolist()) for lg in logits_batch]

    # Select the chunk with the highest entailment score as the citation
    best_idx = max(range(len(nli_results)), key=lambda i: nli_results[i].entailment)
    best_nli = nli_results[best_idx]
    best_chunk = chunks[best_idx]

    # However, if any chunk shows strong contradiction, prefer that verdict
    max_contradiction = max(r.contradiction for r in nli_results)
    if max_contradiction > 0.7 and best_nli.entailment < 0.5:
        contra_idx = max(range(len(nli_results)), key=lambda i: nli_results[i].contradiction)
        best_nli = nli_results[contra_idx]
        best_chunk = chunks[contra_idx]

    return ClaimVerdict(
        claim=claim,
        verdict=best_nli.verdict,
        nli=best_nli,
        best_chunk=best_chunk,
        all_chunks=chunks,
    )


def evaluate_all(
    claims: list[str],
    retrieved: list[list[RetrievedChunk]],
) -> list[ClaimVerdict]:
    """
    Evaluate all claims. claims[i] is evaluated against retrieved[i].
    """
    if len(claims) != len(retrieved):
        raise ValueError("Number of claims must match number of retrieved lists.")

    verdicts: list[ClaimVerdict] = []
    for i, (claim, chunks) in enumerate(zip(claims, retrieved)):
        print(f"  [NLI] ({i+1}/{len(claims)}) {claim[:80]}...")
        verdict = evaluate_claim(claim, chunks)
        verdicts.append(verdict)

    return verdicts


# ── Intra-claim logic check ────────────────────────────────────────────────────

def check_internal_consistency(claims: list[str]) -> float:
    """
    Pairwise NLI between the user's own claims to detect internal contradictions.
    Returns a logic score in [0, 1] — 1.0 means all claims are mutually consistent.
    """
    if len(claims) < 2:
        return 1.0  # single claim is trivially consistent

    model = _get_nli_model()

    pairs = [
        (claims[i], claims[j])
        for i in range(len(claims))
        for j in range(i + 1, len(claims))
    ]

    if not pairs:
        return 1.0

    logits_batch = model.predict(pairs, apply_softmax=False)
    nli_results = [NLIResult.from_logits(lg.tolist()) for lg in logits_batch]

    contradiction_scores = [r.contradiction for r in nli_results]
    max_contradiction = max(contradiction_scores) if contradiction_scores else 0.0

    # Logic score = 1 - max_contradiction (if any pair contradicts, score drops)
    logic_score = max(0.0, 1.0 - max_contradiction)
    return logic_score


# ── CLI test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pipeline.retrieval import retrieve_batch

    claims = [
        "BERT uses bidirectional attention to pre-train language models.",
        "Word2Vec generates embeddings using a recurrent neural network.",
    ]

    print("Retrieving chunks...")
    retrieved = retrieve_batch(claims)

    print("\nEvaluating claims...")
    verdicts = evaluate_all(claims, retrieved)

    for v in verdicts:
        print(f"\n  {v.emoji()} [{v.verdict.upper()}] {v.claim}")
        print(f"     Entailment={v.nli.entailment:.3f}  Contradiction={v.nli.contradiction:.3f}")
        print(f"     Citation: {v.citation()}")
        print(f"     Excerpt: {v.excerpt()[:150]}...")
