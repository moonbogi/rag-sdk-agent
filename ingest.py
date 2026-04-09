"""
ingest.py — PDF → chunks → embeddings → local pickle index
Usage: python ingest.py path/to/sdk-docs.pdf
"""

import os
import sys
import pickle
import numpy as np
import fitz  # pymupdf
from pathlib import Path
from sentence_transformers import SentenceTransformer
import db

# ── Config ────────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 400   # words per chunk
CHUNK_OVERLAP = 60    # overlap words between consecutive chunks
INDEX_PATH    = "sdk_index.pkl"
EMBED_MODEL   = "all-MiniLM-L6-v2"   # ~80 MB, fast, runs fully local
BATCH_SIZE    = 32


def extract_pages(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append({"source": Path(pdf_path).name, "page": i + 1, "text": text})
    doc.close()
    print(f"  Extracted {len(pages)} non-empty pages from {Path(pdf_path).name}")
    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    words: list[str] = []
    meta: list[tuple[str, int]] = []

    for p in pages:
        page_words = p["text"].split()
        words.extend(page_words)
        meta.extend([(p["source"], p["page"])] * len(page_words))

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for start in range(0, len(words), step):
        end = min(start + CHUNK_SIZE, len(words))
        source, page = meta[start]
        chunks.append({"text": " ".join(words[start:end]), "source": source, "page": page})
        if end == len(words):
            break
    return chunks


def load_index() -> dict:
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "rb") as f:
            return pickle.load(f)
    return {"chunks": [], "embeddings": None}


def save_index(index: dict):
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
    print(f"  Index saved → {INDEX_PATH}  ({len(index['chunks'])} total chunks)")


def ingest_pdf(pdf_path: str) -> int:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)
    print(f"\n[ingest] {pdf_path}")
    pages  = extract_pages(pdf_path)
    chunks = chunk_pages(pages)
    print(f"  Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding with '{EMBED_MODEL}' ...")
    model      = SentenceTransformer(EMBED_MODEL)
    texts      = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=BATCH_SIZE,
                              show_progress_bar=True, normalize_embeddings=True)
    if db.using_postgres():
        db.init_db()
        new_chunks = [c for c in chunks if not db.chunk_exists(c["text"])]
        if new_chunks:
            new_embs = embeddings[len(chunks) - len(new_chunks):]
            db.upsert_chunks(new_chunks, new_embs)
        print(f"  Saved {len(new_chunks)} new chunks to Postgres")
        return len(new_chunks)

    index = load_index()
    index["chunks"].extend(chunks)
    index["embeddings"] = (
        embeddings if index["embeddings"] is None
        else np.vstack([index["embeddings"], embeddings])
    )
    save_index(index)
    return len(chunks)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path-to-pdf> [<path-to-pdf> ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        n = ingest_pdf(path)
        print(f"✓ {path} → {n} new chunks\n")
