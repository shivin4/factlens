"""
FactLens — Stage 2: Neural Retrieval
======================================
Input : List of claim strings (from Stage 1)
Output: For each claim → top-k RetrievedChunk objects with metadata

Uses:
  - multi-qa-mpnet-base-dot-v1 (SentenceTransformer) for encoding
  - FAISS IndexFlatIP for nearest-neighbour search
  - SQLite for chunk metadata lookup
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
FAISS_PATH = ROOT / "knowledge_base" / "faiss_index" / "index.faiss"
META_DB_PATH = ROOT / "knowledge_base" / "metadata.db"

# ── Config ─────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "multi-qa-mpnet-base-dot-v1"
TOP_K = 5   # number of chunks to retrieve per claim

# ── Lazy singletons ────────────────────────────────────────────────────────────
_model: Optional[SentenceTransformer] = None
_index: Optional[faiss.Index] = None
_db_conn: Optional[sqlite3.Connection] = None


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk_id: int
    text: str
    source_doc: str
    source_type: str          # "textbook" | "paper"
    page_start: int
    page_end: int
    similarity_score: float   # dot-product score (higher = more similar)

    def citation(self) -> str:
        pages = (
            f"p.{self.page_start}"
            if self.page_start == self.page_end
            else f"pp.{self.page_start}–{self.page_end}"
        )
        return f"{self.source_doc} ({pages})"

    def short_excerpt(self, max_chars: int = 300) -> str:
        return (self.text[:max_chars] + "…") if len(self.text) > max_chars else self.text


# ── Lazy loaders ───────────────────────────────────────────────────────────────

def _load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[Retrieval] Loading embedding model: {EMBEDDING_MODEL} ...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _load_index() -> faiss.Index:
    global _index
    if _index is None:
        if not FAISS_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_PATH}.\n"
                "Run: python -m knowledge_base.build_kb"
            )
        print(f"[Retrieval] Loading FAISS index ({FAISS_PATH.name}) ...")
        _index = faiss.read_index(str(FAISS_PATH))
        print(f"[Retrieval] Index loaded — {_index.ntotal} vectors.")
    return _index


def _load_db() -> sqlite3.Connection:
    global _db_conn
    if _db_conn is None:
        if not META_DB_PATH.exists():
            raise FileNotFoundError(
                f"Metadata DB not found at {META_DB_PATH}.\n"
                "Run: python -m knowledge_base.build_kb"
            )
        _db_conn = sqlite3.connect(str(META_DB_PATH), check_same_thread=False)
        _db_conn.row_factory = sqlite3.Row
    return _db_conn


# ── Core retrieval ─────────────────────────────────────────────────────────────

def _fetch_chunk_meta(chunk_ids: list[int]) -> dict[int, sqlite3.Row]:
    """Batch-fetch chunk metadata from SQLite."""
    conn = _load_db()
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    return {row["chunk_id"]: row for row in rows}


def retrieve(claim: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
    """
    Retrieve the top-k most relevant KB chunks for a single claim string.
    """
    model = _load_model()
    index = _load_index()

    # Encode claim
    query_vec = model.encode(
        [claim],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    # FAISS search
    scores, ids = index.search(query_vec, top_k)
    scores = scores[0].tolist()
    ids = ids[0].tolist()

    # Filter invalid IDs (FAISS returns -1 when fewer results exist)
    valid = [(s, i) for s, i in zip(scores, ids) if i >= 0]
    if not valid:
        return []

    valid_scores, valid_ids = zip(*valid)

    # Fetch metadata
    meta = _fetch_chunk_meta(list(valid_ids))

    results: list[RetrievedChunk] = []
    for score, cid in zip(valid_scores, valid_ids):
        row = meta.get(cid)
        if row is None:
            continue
        results.append(RetrievedChunk(
            chunk_id=cid,
            text=row["text"],
            source_doc=row["source_doc"],
            source_type=row["source_type"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            similarity_score=float(score),
        ))

    return results


def retrieve_batch(claims: list[str], top_k: int = TOP_K) -> list[list[RetrievedChunk]]:
    """
    Retrieve top-k chunks for multiple claims at once (batch encoding for speed).
    Returns a list of lists — one inner list per claim.
    """
    if not claims:
        return []

    model = _load_model()
    index = _load_index()

    query_vecs = model.encode(
        claims,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    ).astype("float32")

    scores_batch, ids_batch = index.search(query_vecs, top_k)

    # Collect all unique chunk IDs for a single DB round-trip
    all_ids = list({int(cid) for row in ids_batch for cid in row if cid >= 0})
    meta = _fetch_chunk_meta(all_ids)

    results: list[list[RetrievedChunk]] = []
    for scores, ids in zip(scores_batch, ids_batch):
        claim_results: list[RetrievedChunk] = []
        for score, cid in zip(scores.tolist(), ids.tolist()):
            if cid < 0:
                continue
            row = meta.get(cid)
            if row is None:
                continue
            claim_results.append(RetrievedChunk(
                chunk_id=cid,
                text=row["text"],
                source_doc=row["source_doc"],
                source_type=row["source_type"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                similarity_score=float(score),
            ))
        results.append(claim_results)

    return results


# ── Key-concept completeness helper ───────────────────────────────────────────

def find_missing_concepts(user_concepts: list[str], top_docs: int = 3) -> list[str]:
    """
    Find important KB concepts not present in the user's concept list.
    Strategy: retrieve top chunks for each user concept, extract all chunk
    concepts, then diff against user's concept list.
    (Lightweight heuristic — full concept gap analysis happens in scoring.py)
    """
    conn = _load_db()
    # Sample the top-N most-retrieved documents concepts
    rows = conn.execute(
        "SELECT text FROM chunks LIMIT 200"
    ).fetchall()

    import re
    kb_words: set[str] = set()
    for row in rows:
        # Extract multi-word noun-like phrases using simple heuristic
        words = re.findall(r'\b[a-z][a-z\-]+\b', row["text"].lower())
        kb_words.update(words)

    user_set = {c.lower() for c in user_concepts}
    missing = [w for w in sorted(kb_words) if w not in user_set and len(w) > 4]
    return missing[:20]   # return top-20 missing for display


# ── CLI test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_claim = "BERT uses bidirectional attention to pre-train language representations"
    print(f"Query: {test_claim}\n")
    chunks = retrieve(test_claim)
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] Score={c.similarity_score:.4f} | {c.citation()}")
        print(f"       {c.short_excerpt(150)}\n")
