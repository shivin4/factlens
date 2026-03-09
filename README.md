# FactLens 🔬

> **Verify NLP explanations against trusted knowledge sources — textbooks & research papers.**

FactLens is a three-stage NLP pipeline that takes a user's explanation of a concept and produces a scored, citation-backed evaluation report. Built for the domain of **Natural Language Processing (v1)**.

---

## Architecture

```
User Explanation (raw text)
        │
        ▼
┌────────────────────────────────────┐
│  Stage 1 — Linguistic Analysis     │
│  spaCy en_core_web_trf             │
│  → Sentences → NER → Noun Chunks  │
│  → SVO Claims                      │
└─────────────────┬──────────────────┘
                  │ claims
                  ▼
┌────────────────────────────────────┐
│  Stage 2 — Neural Retrieval        │
│  multi-qa-mpnet-base-dot-v1        │
│  FAISS IndexFlatIP + SQLite meta   │
└─────────────────┬──────────────────┘
                  │ (claim, chunk) pairs
                  ▼
┌────────────────────────────────────┐
│  Stage 3 — Evaluation Engine       │
│  cross-encoder/nli-deberta-v3-large│
│  → Entailment / Contradiction      │
│  → Weighted Composite Score        │
│  → Feedback + Citations            │
└────────────────────────────────────┘
                  │
                  ▼
         Streamlit Dashboard
```

## Scoring

| Sub-Score | Signal | Weight |
|---|---|---|
| Accuracy | NLI Entailment + BERTScore F1 | 50% |
| Completeness | Concept gap vs corpus | 30% |
| Logic | Pairwise intra-claim NLI | 20% |

`Composite = 0.5 × Accuracy + 0.3 × Completeness + 0.2 × Logic`

## Knowledge Base (v1 — NLP Domain)

**Textbooks (add manually to `data/raw/books/`):**
- Jurafsky & Martin — *Speech and Language Processing* (3rd ed.)
- Manning & Schütze — *Foundations of Statistical NLP*
- Silge & Robinson — *Text Mining with R*
- Bengfort et al. — *Applied Text Analysis with Python*

**Research Papers (auto-downloaded):**
- word2vec (Mikolov et al., 2013)
- GloVe (Pennington et al., 2014)
- Attention Is All You Need (Vaswani et al., 2017)
- BERT (Devlin et al., 2019)
- ELMo (Peters et al., 2018)
- FastText (Bojanowski et al., 2017)
- NER Survey (Li et al., 2020)
- Text Classification Survey (2022)
- QA Survey (Zhu et al., 2020)
- ERNIE-NLI (ACL 2020)

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
# For building the knowledge base from PDFs (includes Docling):
pip install -r requirements-build.txt
```

### 3. Add textbook PDFs

Copy your textbook PDFs into:
```
data/raw/books/
```

### 4. Download research papers

```bash
python scripts/download_papers.py
```

### 5. Build the knowledge base

```bash
python -m knowledge_base.build_kb
# Force rebuild: python -m knowledge_base.build_kb --rebuild
```

> ⏱️ First build takes 5–15 minutes depending on PDF length and hardware. One-time only.

### 6. Launch the app

```bash
streamlit run app/streamlit_app.py
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
factlens/
├── data/
│   ├── raw/
│   │   ├── books/          ← Add textbook PDFs here
│   │   └── papers/         ← Papers downloaded automatically
│   └── processed/          ← Chunk JSON output
├── knowledge_base/
│   ├── build_kb.py         ← KB builder (pymupdf4llm + FAISS + SQLite)
│   ├── faiss_index/        ← FAISS binary index
│   └── metadata.db         ← SQLite chunk metadata
├── pipeline/
│   ├── linguistic.py       ← Stage 1: spaCy NLP
│   ├── retrieval.py        ← Stage 2: FAISS retrieval
│   ├── evaluation.py       ← Stage 3: NLI evaluation
│   ├── scoring.py          ← Composite scoring engine
│   └── runner.py           ← Pipeline orchestrator
├── app/
│   └── streamlit_app.py    ← Dashboard UI
├── scripts/
│   └── download_papers.py  ← Auto paper downloader
├── tests/
│   ├── test_linguistic.py
│   └── test_evaluation.py
└── requirements.txt
```

## PDF Extraction Note

FactLens uses **pymupdf4llm** for fast and robust PDF extraction. It:
- Understands multi-column academic layouts and reading order
- Extracts tables natively as structured markdown
- Skips images automatically without crashing on complex layouts
- Is lightweight enough to process hundreds of pages in minutes using purely CPU

This avoids the memory overhead and `std::bad_alloc` issues common with heavy vision models.

---

## Requirements

- Python 3.10–3.12 locally; **3.11** recommended for Streamlit Cloud (`runtime.txt`)
- CUDA GPU recommended (RTX 4000+ / 16GB VRAM ideal)
  - Works on CPU but evaluation is significantly slower (~2–5 min/query)

---

## Data & GitHub

Textbook PDFs and built indexes are **not** committed (see `.gitignore`). Open-access papers are fetched via `scripts/download_papers.py`.
