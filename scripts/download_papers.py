"""
FactLens — Research Paper Downloader
Downloads 10 foundational NLP papers (all open-access) into data/raw/papers/
"""

import os
import time
import requests
from pathlib import Path

# ── Target directory ──────────────────────────────────────────────────────────
PAPERS_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "papers"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

# ── Paper registry (name → direct PDF URL) ───────────────────────────────────
PAPERS = [
    {
        "id": "word2vec_2013",
        "title": "Efficient Estimation of Word Representations in Vector Space (word2vec)",
        "url": "https://arxiv.org/pdf/1301.3781.pdf",
        "filename": "word2vec_mikolov_2013.pdf",
    },
    {
        "id": "glove_2014",
        "title": "GloVe: Global Vectors for Word Representation",
        "url": "https://nlp.stanford.edu/pubs/glove.pdf",
        "filename": "glove_pennington_2014.pdf",
    },
    {
        "id": "attention_2017",
        "title": "Attention Is All You Need (Transformer)",
        "url": "https://arxiv.org/pdf/1706.03762.pdf",
        "filename": "attention_is_all_you_need_2017.pdf",
    },
    {
        "id": "bert_2019",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "url": "https://arxiv.org/pdf/1810.04805.pdf",
        "filename": "bert_devlin_2019.pdf",
    },
    {
        "id": "elmo_2018",
        "title": "Deep contextualized word representations (ELMo)",
        "url": "https://arxiv.org/pdf/1802.05365.pdf",
        "filename": "elmo_peters_2018.pdf",
    },
    {
        "id": "fasttext_2017",
        "title": "Enriching Word Vectors with Subword Information (FastText)",
        "url": "https://arxiv.org/pdf/1607.04606.pdf",
        "filename": "fasttext_bojanowski_2017.pdf",
    },
    {
        "id": "ner_survey_2020",
        "title": "A Survey on Named Entity Recognition",
        "url": "https://arxiv.org/pdf/2006.15509.pdf",
        "filename": "ner_survey_li_2020.pdf",
    },
    {
        "id": "text_classification_survey_2022",
        "title": "A Survey on Text Classification: From Shallow to Deep Learning",
        "url": "https://arxiv.org/pdf/2008.00364.pdf",
        "filename": "text_classification_survey_2022.pdf",
    },
    {
        "id": "qa_survey_2020",
        "title": "A Survey on Machine Reading Comprehension and Question Answering",
        "url": "https://arxiv.org/pdf/2001.08900.pdf",
        "filename": "qa_survey_zhu_2020.pdf",
    },
    {
        "id": "ernie_nli_2020",
        "title": "ERNIE-NLI: Domain-Specific External Knowledge on NLI (ACL 2020)",
        "url": "https://arxiv.org/pdf/2010.02301.pdf",
        "filename": "ernie_nli_acl2020.pdf",
    },
]


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


def download_paper(paper: dict, retries: int = 3) -> bool:
    dest = PAPERS_DIR / paper["filename"]
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"  [SKIP] Already exists: {paper['filename']}")
        return True

    for attempt in range(1, retries + 1):
        try:
            print(f"  [{attempt}/{retries}] Downloading: {paper['title']}")
            resp = requests.get(paper["url"], headers=HEADERS, timeout=60, stream=True)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)

            size_kb = dest.stat().st_size / 1024
            print(f"  [OK] Saved {paper['filename']} ({size_kb:.0f} KB)")
            return True

        except Exception as exc:
            print(f"  [WARN] Attempt {attempt} failed: {exc}")
            if attempt < retries:
                time.sleep(3)

    print(f"  [FAIL] Could not download: {paper['filename']}")
    return False


def main():
    print(f"\nFactLens — Paper Downloader")
    print(f"Saving to: {PAPERS_DIR}\n")

    success, fail = 0, 0
    for paper in PAPERS:
        ok = download_paper(paper)
        if ok:
            success += 1
        else:
            fail += 1
        time.sleep(1)   # be polite to arxiv

    print(f"\nDone. {success}/{len(PAPERS)} papers downloaded.")
    if fail:
        print(f"  {fail} failed — check your internet connection and retry.")


if __name__ == "__main__":
    main()
