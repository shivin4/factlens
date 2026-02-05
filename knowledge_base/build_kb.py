"""
FactLens — Knowledge Base Builder
==================================
Ingests PDFs from data/raw/ using Docling, chunks the text,
encodes with sentence-transformers, and stores:
  - FAISS index  →  knowledge_base/faiss_index/index.faiss
  - Metadata DB  →  knowledge_base/metadata.db  (SQLite)

Usage:
  python -m knowledge_base.build_kb              # build if not exists
  python -m knowledge_base.build_kb --rebuild    # force rebuild
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

# Force UTF-8 output on Windows to avoid charmap errors from Docling log messages
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_BOOKS = ROOT / "data" / "raw" / "books"
DATA_PAPERS = ROOT / "data" / "raw" / "papers"
KB_DIR = ROOT / "knowledge_base"
FAISS_DIR = KB_DIR / "faiss_index"
FAISS_PATH = FAISS_DIR / "index.faiss"
META_DB_PATH = KB_DIR / "metadata.db"
PROCESSED_DIR = ROOT / "data" / "processed"

FAISS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Chunking parameters ────────────────────────────────────────────────────────
CHUNK_TOKEN_SIZE = 250      # target tokens per chunk
CHUNK_OVERLAP = 50          # token overlap between consecutive chunks
MIN_CHUNK_CHARS = 100       # skip chunks shorter than this (likely headers/noise)
MIN_PAGE_TEXT_RATIO = 0.1   # skip pages where text < 10% of average (figure-only pages)

# ── Embedding model ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "multi-qa-mpnet-base-dot-v1"
EMBEDDING_DIM = 768


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def collect_pdfs() -> list[dict]:
    """Return list of {path, source_type} for all PDFs in books/ and papers/."""
    pdfs = []
    for pdf_path in sorted(DATA_BOOKS.glob("*.pdf")):
        pdfs.append({"path": pdf_path, "source_type": "textbook"})
    for pdf_path in sorted(DATA_PAPERS.glob("*.pdf")):
        pdfs.append({"path": pdf_path, "source_type": "paper"})
    return pdfs


def extract_text(pdf_path: Path) -> list[dict]:
    """
    Extract text from a PDF using pymupdf4llm.

    pymupdf4llm converts each page to clean markdown, handling:
      - Multi-column academic layouts
      - Tables (rendered as markdown tables)
      - Correct reading order
      - Skips pure-image pages gracefully

    Returns list of {page_num, text}.
    """
    import pymupdf4llm

    print(f"    Extracting: {pdf_path.name} ...")

    # to_markdown returns a list of dicts when page_chunks=True
    page_chunks = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,      # one dict per page
        show_progress=False,
    )

    pages = []
    for chunk in page_chunks:
        text = chunk.get("text", "").strip()
        page_num = chunk.get("metadata", {}).get("page", 0) + 1  # 0-indexed → 1-indexed
        if len(text) > MIN_CHUNK_CHARS:
            pages.append({"page_num": page_num, "text": text})

    return pages


def simple_tokenize(text: str) -> list[str]:
    """Lightweight whitespace tokenizer for chunk sizing (no spaCy needed here)."""
    return text.split()


def chunk_pages(pages: list[dict], source_doc: str) -> list[dict]:
    """
    Sliding-window chunk over page texts.
    Yields {chunk_id, source_doc, page_num, chunk_index, text}.
    """
    # Flatten all page tokens while tracking page boundaries
    all_tokens = []
    token_pages = []   # page_num for each token

    for page in pages:
        tokens = simple_tokenize(page["text"])
        all_tokens.extend(tokens)
        token_pages.extend([page["page_num"]] * len(tokens))

    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(all_tokens):
        end = min(start + CHUNK_TOKEN_SIZE, len(all_tokens))
        chunk_tokens = all_tokens[start:end]
        chunk_text = " ".join(chunk_tokens).strip()

        if len(chunk_text) >= MIN_CHUNK_CHARS:
            # Determine the page range this chunk spans
            page_start = token_pages[start]
            page_end = token_pages[end - 1]

            chunks.append({
                "source_doc": source_doc,
                "page_start": page_start,
                "page_end": page_end,
                "chunk_index": chunk_idx,
                "text": chunk_text,
            })
            chunk_idx += 1

        # Slide forward (skip overlap)
        start += CHUNK_TOKEN_SIZE - CHUNK_OVERLAP

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────────────────────────────────────

def init_db(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id    INTEGER PRIMARY KEY,   -- matches FAISS index row
            source_doc  TEXT NOT NULL,
            source_type TEXT NOT NULL,
            page_start  INTEGER,
            page_end    INTEGER,
            chunk_index INTEGER,
            text        TEXT NOT NULL
        )
    """)
    conn.commit()


def insert_chunks(conn: sqlite3.Connection, chunks: list[dict], source_type: str, start_id: int):
    rows = [
        (
            start_id + i,
            c["source_doc"],
            source_type,
            c["page_start"],
            c["page_end"],
            c["chunk_index"],
            c["text"],
        )
        for i, c in enumerate(chunks)
    ]
    conn.executemany(
        "INSERT INTO chunks VALUES (?,?,?,?,?,?,?)", rows
    )
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Main build
# ─────────────────────────────────────────────────────────────────────────────

def build(rebuild: bool = False):
    if FAISS_PATH.exists() and META_DB_PATH.exists() and not rebuild:
        print("Knowledge base already exists. Use --rebuild to force rebuild.")
        return

    # Lazy imports (heavy)
    import faiss
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    pdfs = collect_pdfs()
    if not pdfs:
        print("ERROR: No PDFs found in data/raw/books/ or data/raw/papers/")
        print("  → Place textbook PDFs in:  data/raw/books/")
        print("  → Run download_papers.py for research papers")
        sys.exit(1)

    print(f"\nFound {len(pdfs)} PDFs to process.\n")

    # ── DB setup ──────────────────────────────────────────────────────────────
    if META_DB_PATH.exists() and rebuild:
        META_DB_PATH.unlink()
    conn = sqlite3.connect(META_DB_PATH)
    init_db(conn)

    # ── Collect all chunks ────────────────────────────────────────────────────
    all_chunks: list[dict] = []
    source_types: list[str] = []

    for pdf_info in pdfs:
        pdf_path = pdf_info["path"]
        source_type = pdf_info["source_type"]
        print(f"\n[{source_type.upper()}] {pdf_path.name}")

        try:
            pages = extract_text(pdf_path)
            chunks = chunk_pages(pages, pdf_path.stem)
            print(f"    → {len(pages)} pages, {len(chunks)} chunks")
            all_chunks.extend(chunks)
            source_types.extend([source_type] * len(chunks))
        except Exception as exc:
            print(f"    [ERROR] Skipping {pdf_path.name}: {exc}")

    if not all_chunks:
        print("ERROR: No chunks extracted from any PDF.")
        sys.exit(1)

    print(f"\nTotal chunks: {len(all_chunks)}")

    # ── Embed ─────────────────────────────────────────────────────────────────
    print(f"\nLoading embedding model: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in all_chunks]
    print("Encoding chunks (this may take a while on first run)...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # for dot-product similarity
    )
    embeddings = np.array(embeddings, dtype="float32")

    # ── Write DB ──────────────────────────────────────────────────────────────
    print("\nWriting metadata to SQLite ...")
    for i, (chunk, stype) in enumerate(zip(all_chunks, source_types)):
        insert_chunks(conn, [chunk], stype, i)

    conn.close()

    # ── Build FAISS index ─────────────────────────────────────────────────────
    print("Building FAISS index ...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)   # Inner-product (dot) for normalized vecs
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_PATH))
    print(f"FAISS index saved: {FAISS_PATH}  ({index.ntotal} vectors)")

    # ── Save processed chunk texts for inspection ─────────────────────────────
    chunks_json = PROCESSED_DIR / "chunks.json"
    with open(chunks_json, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"Chunk texts saved: {chunks_json}")

    print("\n✅ Knowledge base built successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FactLens knowledge base")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild even if index exists")
    args = parser.parse_args()
    build(rebuild=args.rebuild)
