# FactLens

Verify NLP explanations against trusted knowledge sources.

## Pipeline (in progress)

| Stage | Module | Status |
|-------|--------|--------|
| 1 | `pipeline/linguistic.py` | Done |
| 2 | Retrieval | Planned |
| 3 | NLI evaluation | Planned |

## Knowledge base

Place textbook PDFs in `data/raw/books/` (local only). Download open-access papers with `scripts/download_papers.py`, then build the index with `python -m knowledge_base.build_kb`.
