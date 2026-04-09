"""
ingest_url.py — Crawl SDK documentation websites → chunk → embed → sdk_index.pkl

Supports:
  Stripe iOS:   https://docs.stripe.com/sdks/ios
  Spotify iOS:  https://developer.spotify.com/documentation/ios
  Any other public SDK doc site

Usage:
  python ingest_url.py https://docs.stripe.com/sdks/ios --max-pages 60
  python ingest_url.py https://developer.spotify.com/documentation/ios --max-pages 40
"""

import sys
import time
import hashlib
import argparse
import pickle
import re
from urllib.parse import urljoin, urlparse

import requests
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from ingest import (
    CHUNK_SIZE, CHUNK_OVERLAP, INDEX_PATH, EMBED_MODEL, BATCH_SIZE,
    chunk_pages, load_index, save_index,
)

# ── Site-specific config ───────────────────────────────────────────────────────
SITE_CONFIGS = {
    "docs.stripe.com": {
        "stay_under":   "/sdks/ios",          # only crawl iOS SDK pages
        "content_sel":  "article",             # main content selector
        "strip_tags":   ["nav", "footer", "aside", ".sidebar", ".breadcrumb"],
        "delay":        0.3,
    },
    "developer.spotify.com": {
        "stay_under":   "/documentation/ios",
        "content_sel":  "main",
        "strip_tags":   ["nav", "footer", "aside"],
        "delay":        0.5,
    },
}

DEFAULT_CONFIG = {
    "stay_under":  None,
    "content_sel": "main, article, .content, #content, body",
    "strip_tags":  ["nav", "footer", "aside", "header", ".sidebar"],
    "delay":       0.4,
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_config(base_url: str) -> dict:
    host = urlparse(base_url).netloc
    return SITE_CONFIGS.get(host, DEFAULT_CONFIG)


def clean_text(text: str) -> str:
    """Collapse whitespace, remove code-noise."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def html_to_text(html: str, cfg: dict) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove noisy elements
    for sel in cfg["strip_tags"]:
        for tag in soup.select(sel):
            tag.decompose()

    # Try specific content container first
    container = soup.select_one(cfg["content_sel"])
    if not container:
        container = soup.body or soup

    return clean_text(container.get_text(separator="\n"))


def same_domain_links(html: str, base_url: str, stay_under: str | None) -> list[str]:
    soup  = BeautifulSoup(html, "html.parser")
    host  = urlparse(base_url).netloc
    links = []

    for a in soup.find_all("a", href=True):
        href     = a["href"].split("#")[0]          # strip fragments
        absolute = urljoin(base_url, href)
        parsed   = urlparse(absolute)

        if parsed.netloc != host:
            continue
        if stay_under and not parsed.path.startswith(stay_under):
            continue
        if parsed.path.endswith((".png", ".jpg", ".svg", ".zip", ".pdf")):
            continue

        links.append(absolute.split("?")[0])       # strip query params

    return list(set(links))


# ── Crawler ────────────────────────────────────────────────────────────────────
def crawl(start_url: str, max_pages: int) -> list[dict]:
    """BFS crawl. Returns list of {source, page, text} dicts (page = URL)."""
    cfg      = get_config(start_url)
    visited  = set()
    queue    = [start_url]
    pages    = []
    domain   = urlparse(start_url).netloc

    print(f"[crawl] Starting at {start_url}  (max {max_pages} pages)")
    print(f"[crawl] Config: stay_under='{cfg['stay_under']}'  content='{cfg['content_sel']}'")

    while queue and len(pages) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code != 200 or "text/html" not in r.headers.get("Content-Type", ""):
                continue
        except Exception as e:
            print(f"  SKIP {url}  ({e})")
            continue

        text = html_to_text(r.text, cfg)
        if len(text.split()) < 40:          # skip near-empty pages
            continue

        pages.append({
            "source": f"{domain}",          # grouped by domain for display
            "page":   url,                  # full URL as "page" reference
            "text":   text,
        })
        print(f"  [{len(pages):03d}] {url}  ({len(text.split())} words)")

        new_links = same_domain_links(r.text, url, cfg["stay_under"])
        for link in new_links:
            if link not in visited:
                queue.append(link)

        time.sleep(cfg["delay"])

    print(f"[crawl] Done — {len(pages)} pages fetched")
    return pages


# ── Fingerprint for dedup ──────────────────────────────────────────────────────
def _chunk_fp(chunk: dict) -> str:
    return hashlib.md5(chunk["text"].encode()).hexdigest()


# ── Main pipeline ──────────────────────────────────────────────────────────────
def ingest_url(start_url: str, max_pages: int = 60) -> int:
    pages  = crawl(start_url, max_pages=max_pages)
    chunks = chunk_pages(pages)
    print(f"[ingest_url] {len(chunks)} chunks from {len(pages)} pages")

    # Dedup against existing index
    index      = load_index()
    existing   = {_chunk_fp(c) for c in index["chunks"]}
    new_chunks = [c for c in chunks if _chunk_fp(c) not in existing]
    print(f"[ingest_url] {len(new_chunks)} new (skipping {len(chunks)-len(new_chunks)} duplicates)")

    if not new_chunks:
        print("[ingest_url] Nothing new to add.")
        return 0

    print(f"[ingest_url] Embedding {len(new_chunks)} chunks with '{EMBED_MODEL}' ...")
    model      = SentenceTransformer(EMBED_MODEL)
    texts      = [c["text"] for c in new_chunks]
    embeddings = model.encode(texts, batch_size=BATCH_SIZE,
                              show_progress_bar=True, normalize_embeddings=True)

    index["chunks"].extend(new_chunks)
    index["embeddings"] = (
        embeddings if index["embeddings"] is None
        else np.vstack([index["embeddings"], embeddings])
    )

    save_index(index)
    return len(new_chunks)


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crawl SDK doc website and ingest into local vector index"
    )
    parser.add_argument("url", help="Start URL (e.g. https://docs.stripe.com/sdks/ios)")
    parser.add_argument("--max-pages", type=int, default=60,
                        help="Maximum pages to crawl (default 60)")
    parser.add_argument("--no-filter", action="store_true",
                        help="Ignore stay_under restriction — crawl entire domain for larger corpus")
    args = parser.parse_args()

    if args.no_filter:
        host = urlparse(args.url).netloc
        if host in SITE_CONFIGS:
            SITE_CONFIGS[host]["stay_under"] = None
        DEFAULT_CONFIG["stay_under"] = None
        print("[ingest_url] --no-filter: crawling entire domain (no path restriction)")

    n = ingest_url(args.url, max_pages=args.max_pages)
    print(f"\n✓ Done — {n} new chunks added to {INDEX_PATH}")
