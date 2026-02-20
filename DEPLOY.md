# Deploying FactLens (public demo)

## What goes on GitHub vs what stays local

| Asset | In repo? | Notes |
|-------|----------|--------|
| Source code | Yes | `pipeline/`, `app/`, `knowledge_base/build_kb.py` |
| Textbook PDFs | **No** | `data/raw/books/` — gitignored |
| Paper PDFs | **No** | Fetched via `scripts/download_papers.py` |
| `chunks.json`, FAISS index | **No** | Build locally or ship as a **GitHub Release** (papers-only) |

## Papers-only demo knowledge base

1. Ensure `data/raw/books/` is empty (or move textbooks aside).
2. `python scripts/download_papers.py`
3. `python -m knowledge_base.build_kb --rebuild`
4. Zip `knowledge_base/faiss_index/index.faiss` and `knowledge_base/metadata.db` → upload as release asset `demo-kb-papers.zip`.

## Streamlit Community Cloud

1. Push this repo to GitHub (no secrets required for a static demo KB baked into a release download step, or document manual build on first run).
2. [share.streamlit.io](https://share.streamlit.io) → New app → repo `factlens`, main file `app/streamlit_app.py`.
3. Add `packages.txt` at repo root (spaCy model).
4. Free tier is CPU-only and ~1 GB RAM — first query may take several minutes; consider disabling BERTScore in the UI for demos.

## Resume links

- **Repo:** `https://github.com/<username>/factlens`
- **Live demo:** `https://<app-name>.streamlit.app`
