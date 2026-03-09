"""
Zip the built knowledge base for upload as a GitHub Release asset.

Run after a papers-only build:
  python scripts/download_papers.py
  python -m knowledge_base.build_kb --rebuild
  python scripts/package_demo_kb.py
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAISS = ROOT / "knowledge_base" / "faiss_index" / "index.faiss"
META = ROOT / "knowledge_base" / "metadata.db"
OUT = ROOT / "demo-kb-papers.zip"


def main() -> None:
    if not FAISS.is_file() or not META.is_file():
        raise SystemExit(
            "Missing knowledge base. Build first:\n"
            "  python scripts/download_papers.py\n"
            "  python -m knowledge_base.build_kb --rebuild"
        )

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(FAISS, "faiss_index/index.faiss")
        zf.write(META, "metadata.db")

    size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"Created {OUT} ({size_mb:.1f} MB)")
    print("Upload this file to a GitHub Release (e.g. tag v1.0.0).")


if __name__ == "__main__":
    main()
