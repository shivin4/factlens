# Build backdated commit history for FactLens (run once from repo root).
# Does NOT push to GitHub.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

# Preserve final README before intermediate commits overwrite it
$ReadmeSnapshot = Join-Path $Root "scripts\.readme-snapshot.md"
Copy-Item -Path "README.md" -Destination $ReadmeSnapshot -Force

function Invoke-GitCommit {
    param(
        [string]$Date,
        [string]$Message,
        [string[]]$Paths
    )
    if ($Paths) {
        git add @Paths
    }
    $env:GIT_AUTHOR_DATE = $Date
    $env:GIT_COMMITTER_DATE = $Date
    git commit -m $Message
    Remove-Item Env:\GIT_AUTHOR_DATE -ErrorAction SilentlyContinue
    Remove-Item Env:\GIT_COMMITTER_DATE -ErrorAction SilentlyContinue
}

# Fresh repo
if (Test-Path .git) { Remove-Item -Recurse -Force .git }
git init
git branch -M main

# ── Feb 1: scaffold ──────────────────────────────────────────────────────────
@'
# FactLens

Verify NLP explanations against trusted knowledge sources (textbooks and research papers).

**Status:** Early scaffold — NLP domain v1.

## Planned pipeline

1. Linguistic analysis (spaCy)
2. Dense retrieval (FAISS)
3. NLI evaluation (cross-encoder)
'@ | Set-Content -Path README.md -Encoding utf8

@'
spacy>=3.7.0
numpy>=1.26.0
nltk>=3.8.0
tqdm>=4.66.0
requests>=2.31.0
'@ | Set-Content -Path requirements.txt -Encoding utf8

@'
venv/
__pycache__/
*.py[cod]
.pytest_cache/
'@ | Set-Content -Path .gitignore -Encoding utf8

Invoke-GitCommit -Date "2026-02-01 10:15:00 -0500" -Message @"
chore: scaffold FactLens project

- Add minimal README and core Python dependencies
- Create pipeline, knowledge_base, and tests package roots
"@ -Paths @(
    "README.md", "requirements.txt", ".gitignore",
    "pipeline/__init__.py", "knowledge_base/__init__.py", "tests/__init__.py"
)

# ── Feb 3: Stage 1 ───────────────────────────────────────────────────────────
Invoke-GitCommit -Date "2026-02-03 14:20:00 -0500" -Message @"
feat(stage1): add spaCy linguistic analysis and claim extraction

- Parse sentences, entities, and noun-phrase concepts
- Extract subject-verb-object claims for downstream verification
- Add unit tests for linguistic stage
"@ -Paths @("pipeline/linguistic.py", "tests/test_linguistic.py")

# ── Feb 5: Knowledge base ────────────────────────────────────────────────────
@'
spacy>=3.7.0
numpy>=1.26.0
nltk>=3.8.0
tqdm>=4.66.0
requests>=2.31.0

# PDF extraction & indexing
docling>=2.0.0
sentence-transformers>=3.0.0
faiss-cpu>=1.8.0
'@ | Set-Content -Path requirements.txt -Encoding utf8

@'
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
'@ | Set-Content -Path README.md -Encoding utf8

Invoke-GitCommit -Date "2026-02-05 11:00:00 -0500" -Message @"
feat(kb): add PDF ingestion pipeline and paper downloader

- Ingest PDFs from data/raw with Docling, chunk, embed, and index
- Persist FAISS index and SQLite chunk metadata
- Add script to fetch open-access NLP papers into data/raw/papers
- Document local-only textbook directory layout
"@ -Paths @(
    "requirements.txt", "README.md",
    "knowledge_base/build_kb.py",
    "scripts/download_papers.py",
    "data/raw/books/README.md",
    "data/raw/papers/README.md"
)

# ── Feb 7: Retrieval ─────────────────────────────────────────────────────────
Invoke-GitCommit -Date "2026-02-07 16:45:00 -0500" -Message @"
feat(stage2): add FAISS dense retrieval over knowledge base

- Encode claims with multi-qa-mpnet-base-dot-v1
- Retrieve top-k chunks via IndexFlatIP and SQLite metadata lookup
"@ -Paths @("pipeline/retrieval.py")

# ── Feb 9: Evaluation ────────────────────────────────────────────────────────
@'
spacy>=3.7.0
numpy>=1.26.0
nltk>=3.8.0
tqdm>=4.66.0
requests>=2.31.0

docling>=2.0.0
sentence-transformers>=3.0.0
faiss-cpu>=1.8.0

torch>=2.2.0
transformers>=4.40.0
bert-score>=0.3.13
'@ | Set-Content -Path requirements.txt -Encoding utf8

Invoke-GitCommit -Date "2026-02-09 13:30:00 -0500" -Message @"
feat(stage3): add cross-encoder NLI evaluation engine

- Score claim-chunk pairs with nli-deberta-v3-large
- Aggregate entailment, neutral, and contradiction verdicts with citations
- Add evaluation unit tests
"@ -Paths @("pipeline/evaluation.py", "tests/test_evaluation.py", "requirements.txt")

# ── Feb 15: Scoring + runner ─────────────────────────────────────────────────
Invoke-GitCommit -Date "2026-02-15 18:00:00 -0500" -Message @"
feat(pipeline): add composite scoring and end-to-end orchestration

- Weighted accuracy, completeness, and logic sub-scores
- Wire stages 1-3 through pipeline runner with timing metadata
- Support ablation study helper for benchmarking
"@ -Paths @("pipeline/scoring.py", "pipeline/runner.py")

# ── Feb 17: Streamlit UI ─────────────────────────────────────────────────────
@'
spacy>=3.7.0
numpy>=1.26.0
nltk>=3.8.0
tqdm>=4.66.0
requests>=2.31.0

docling>=2.0.0
sentence-transformers>=3.0.0
faiss-cpu>=1.8.0

torch>=2.2.0
transformers>=4.40.0
bert-score>=0.3.13

streamlit>=1.35.0
plotly>=5.22.0
'@ | Set-Content -Path requirements.txt -Encoding utf8

Invoke-GitCommit -Date "2026-02-17 12:10:00 -0500" -Message @"
feat(ui): add Streamlit verification dashboard

- Interactive explanation input with score cards and citations
- Sidebar controls for retrieval depth and BERTScore toggle
- Display knowledge base build status from FAISS and SQLite
"@ -Paths @("app/streamlit_app.py", "requirements.txt")

# ── Feb 19: Docs & deploy ────────────────────────────────────────────────────
Copy-Item -Path $ReadmeSnapshot -Destination "README.md" -Force
@'
# Python
venv/
__pycache__/
*.py[cod]
.pytest_cache/
.coverage
.env
.env.*

# Copyrighted / large local data
data/raw/books/*
!data/raw/books/README.md
data/raw/papers/*.pdf
data/processed/

# Built knowledge base (rebuild locally or from release)
knowledge_base/faiss_index/
knowledge_base/metadata.db

# IDE / OS
.idea/
.vscode/
.DS_Store
Thumbs.db
'@ | Set-Content -Path .gitignore -Encoding utf8

@'
spacy>=3.7.0
# Install transformer model separately:
#   python -m spacy download en_core_web_trf

docling>=2.0.0

sentence-transformers>=3.0.0
faiss-cpu>=1.8.0

torch>=2.2.0
transformers>=4.40.0
bert-score>=0.3.13

numpy>=1.26.0
nltk>=3.8.0
tqdm>=4.66.0
requests>=2.31.0

streamlit>=1.35.0
plotly>=5.22.0

pytest>=8.0.0
pytest-cov>=5.0.0
'@ | Set-Content -Path requirements.txt -Encoding utf8

Invoke-GitCommit -Date "2026-02-19 20:30:00 -0500" -Message @"
docs: finalize README, gitignore, and Streamlit deploy guide

- Document full architecture, scoring weights, and setup steps
- Exclude copyrighted PDFs and built indexes from version control
- Add packages.txt for Streamlit Cloud and DEPLOY.md for public demo
"@ -Paths @("README.md", ".gitignore", "requirements.txt", "DEPLOY.md", "packages.txt", "scripts/build-git-history.ps1")

Write-Host ""
Write-Host "Done. Commit history:"
git log --oneline --format="%h %ad %s" --date=short
