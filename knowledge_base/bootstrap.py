"""
Download and extract a pre-built demo knowledge base (papers-only) when
FAISS/SQLite artifacts are not present — used on Streamlit Community Cloud.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "knowledge_base"
FAISS_DIR = KB_DIR / "faiss_index"
FAISS_PATH = FAISS_DIR / "index.faiss"
META_PATH = KB_DIR / "metadata.db"


def kb_is_ready() -> bool:
    return FAISS_PATH.is_file() and META_PATH.is_file()


def _resolve_demo_kb_url(explicit_url: str | None = None) -> str | None:
    if explicit_url:
        return explicit_url.strip() or None
    env_url = os.environ.get("DEMO_KB_URL", "").strip()
    if env_url:
        return env_url
    try:
        import streamlit as st

        if hasattr(st, "secrets") and "DEMO_KB_URL" in st.secrets:
            return str(st.secrets["DEMO_KB_URL"]).strip() or None
    except Exception:
        pass
    return None


def ensure_knowledge_base(demo_kb_url: str | None = None) -> bool:
    """
    If the KB is missing, download demo-kb-papers.zip from DEMO_KB_URL
    (GitHub Release asset URL) and extract into knowledge_base/.
    """
    if kb_is_ready():
        return True

    url = _resolve_demo_kb_url(demo_kb_url)
    if not url:
        return False

    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = KB_DIR / "_demo_kb_download.zip"

    print(f"[bootstrap] Downloading demo KB from {url} ...")
    urlretrieve(url, zip_path)

    print("[bootstrap] Extracting demo KB ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(KB_DIR)

    zip_path.unlink(missing_ok=True)
    ready = kb_is_ready()
    if ready:
        print("[bootstrap] Demo knowledge base ready.")
    else:
        print("[bootstrap] ERROR: extract did not produce index.faiss and metadata.db")
    return ready
