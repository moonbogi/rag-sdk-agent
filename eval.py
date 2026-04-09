"""
eval.py — Retrieval quality evaluation for the RAG pipeline

Metrics computed:
  - Precision@K   : fraction of retrieved chunks that are relevant
  - Recall@K      : fraction of relevant chunks that were retrieved
  - MRR           : Mean Reciprocal Rank (how high the first relevant chunk ranks)
  - NDCG@K        : Normalized Discounted Cumulative Gain

Ablations:
  - Embedding models  : MiniLM vs mpnet-base vs bge-small
  - Chunk sizes       : 200 / 400 / 600 words
  - Top-K             : 4 / 8 / 12

Usage:
  # Run full ablation suite (builds fresh indexes — takes ~10-15 min first time)
  python eval.py --url https://docs.stripe.com/sdks/ios --full-ablation

  # Quick eval on your existing index
  python eval.py --quick

  # Save results to CSV for your resume/writeup
  python eval.py --full-ablation --output results.csv
"""

import os
import time
import pickle
import hashlib
import argparse
import itertools
import csv
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# ── Eval query set ─────────────────────────────────────────────────────────────
# Each entry: query + list of keywords that MUST appear in a relevant chunk.
# Relevance = chunk text contains at least one keyword (weak supervision).
# For a stronger eval, manually label 20-30 queries — but this gets you real numbers.
EVAL_QUERIES = [
    {
        "query": "how to initialize the Stripe iOS SDK",
        "relevant_keywords": ["StripeAPI", "publishableKey", "STPAPIClient", "apiKey", "configure"],
    },
    {
        "query": "PaymentSheet setup and configuration",
        "relevant_keywords": ["PaymentSheet", "PaymentSheetResult", "paymentIntentClientSecret"],
    },
    {
        "query": "Apple Pay integration with Stripe",
        "relevant_keywords": ["PKPaymentRequest", "STPApplePayContext", "applePayContext", "merchantIdentifier"],
    },
    {
        "query": "card payment UI element STPCardFormView",
        "relevant_keywords": ["STPCardFormView", "STPPaymentCardTextField", "cardParams"],
    },
    {
        "query": "3D Secure authentication handling",
        "relevant_keywords": ["SFSafariViewController", "3D Secure", "redirectURL", "STPRedirectContext"],
    },
    {
        "query": "how to handle payment errors and failures",
        "relevant_keywords": ["STPError", "error.localizedDescription", "STPCardErrorCode", "decline"],
    },
    {
        "query": "setup payment intent on client side",
        "relevant_keywords": ["paymentIntentClientSecret", "STPPaymentIntentParams", "confirmPayment"],
    },
    {
        "query": "Swift Package Manager installation Stripe",
        "relevant_keywords": ["Package Dependencies", "github.com/stripe", "SPM", "Swift Package"],
    },
    {
        "query": "collect billing address from customer",
        "relevant_keywords": ["billingDetails", "STPPaymentMethodBillingDetails", "address", "postalCode"],
    },
    {
        "query": "test mode publishable key sandbox",
        "relevant_keywords": ["pk_test", "test mode", "testPublishableKey", "sandbox"],
    },
]

# Fallback queries for Spotify (auto-selected if Spotify is the indexed source)
EVAL_QUERIES_SPOTIFY = [
    {
        "query": "how to authenticate with Spotify iOS SDK",
        "relevant_keywords": ["SPTSessionManager", "SPTConfiguration", "clientID", "redirectURL"],
    },
    {
        "query": "connect to Spotify App Remote",
        "relevant_keywords": ["SPTAppRemote", "appRemoteDidEstablishConnection", "connect"],
    },
    {
        "query": "subscribe to player state changes",
        "relevant_keywords": ["playerStateDidChange", "SPTAppRemotePlayerState", "subscribeToPlayerState"],
    },
    {
        "query": "play a track with Spotify URI",
        "relevant_keywords": ["playURI", "play:", "spotifyURI", "SPTAppRemotePlayerAPI"],
    },
    {
        "query": "handle Spotify authorization callback",
        "relevant_keywords": ["application:openURL", "handleAuthCallback", "accessToken"],
    },
]


# ── Data structures ────────────────────────────────────────────────────────────
@dataclass
class QueryResult:
    query:        str
    precision_at_k: float
    recall_at_k:  float
    mrr:          float
    ndcg_at_k:    float
    relevant_found: int
    total_retrieved: int
    top_chunk_score: float


@dataclass
class EvalConfig:
    embed_model:  str
    chunk_size:   int
    chunk_overlap: int
    top_k:        int


@dataclass
class EvalResult:
    config:        EvalConfig
    mean_precision: float
    mean_recall:   float
    mean_mrr:      float
    mean_ndcg:     float
    latency_ms:    float          # avg retrieval latency
    index_size:    int            # number of chunks
    query_results: list[QueryResult] = field(default_factory=list)


# ── Relevance judge ────────────────────────────────────────────────────────────
def is_relevant(chunk_text: str, keywords: list[str]) -> bool:
    """Weak-supervision: relevant if any keyword appears in the chunk (case-insensitive)."""
    lower = chunk_text.lower()
    return any(kw.lower() in lower for kw in keywords)


# ── Metrics ───────────────────────────────────────────────────────────────────
def precision_at_k(relevance: list[bool]) -> float:
    if not relevance:
        return 0.0
    return sum(relevance) / len(relevance)


def recall_at_k(relevance: list[bool], total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    return sum(relevance) / total_relevant


def mrr(relevance: list[bool]) -> float:
    for i, rel in enumerate(relevance):
        if rel:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(relevance: list[bool]) -> float:
    def dcg(rels):
        return sum(r / np.log2(i + 2) for i, r in enumerate(rels))
    actual = dcg(relevance)
    ideal  = dcg(sorted(relevance, reverse=True))
    return actual / ideal if ideal > 0 else 0.0


# ── Core retrieval (works with any pre-built index) ───────────────────────────
def retrieve_from_index(index: dict, model: SentenceTransformer,
                        query: str, top_k: int) -> tuple[list[dict], float]:
    """Returns (top_k chunks, latency_ms)."""
    t0    = time.perf_counter()
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    scores = index["embeddings"] @ q_emb
    top_idx = np.argsort(scores)[::-1][:top_k]
    elapsed = (time.perf_counter() - t0) * 1000

    chunks = [
        {**index["chunks"][i], "score": float(scores[i])}
        for i in top_idx
    ]
    return chunks, elapsed


# ── Count total relevant docs in index (for recall denominator) ───────────────
def count_relevant_in_index(index: dict, keywords: list[str]) -> int:
    return sum(1 for c in index["chunks"] if is_relevant(c["text"], keywords))


# ── Evaluate one (index, model, top_k) combination ───────────────────────────
def evaluate_config(
    index:      dict,
    model:      SentenceTransformer,
    top_k:      int,
    queries:    list[dict],
    config:     EvalConfig,
) -> EvalResult:
    query_results = []
    total_latency = 0.0

    for q in queries:
        retrieved, latency = retrieve_from_index(index, model, q["query"], top_k)
        total_latency += latency

        relevance      = [is_relevant(c["text"], q["relevant_keywords"]) for c in retrieved]
        total_relevant = count_relevant_in_index(index, q["relevant_keywords"])

        query_results.append(QueryResult(
            query           = q["query"],
            precision_at_k  = precision_at_k(relevance),
            recall_at_k     = recall_at_k(relevance, total_relevant),
            mrr             = mrr(relevance),
            ndcg_at_k       = ndcg_at_k(relevance),
            relevant_found  = sum(relevance),
            total_retrieved = len(retrieved),
            top_chunk_score = retrieved[0]["score"] if retrieved else 0.0,
        ))

    n = len(query_results)
    return EvalResult(
        config          = config,
        mean_precision  = sum(r.precision_at_k for r in query_results) / n,
        mean_recall     = sum(r.recall_at_k    for r in query_results) / n,
        mean_mrr        = sum(r.mrr            for r in query_results) / n,
        mean_ndcg       = sum(r.ndcg_at_k      for r in query_results) / n,
        latency_ms      = total_latency / n,
        index_size      = len(index["chunks"]),
        query_results   = query_results,
    )


# ── Build a fresh index for ablation ─────────────────────────────────────────
def build_index(pages: list[dict], embed_model: str,
                chunk_size: int, chunk_overlap: int) -> dict:
    """Build an in-memory index — does NOT overwrite sdk_index.pkl."""
    from ingest import BATCH_SIZE

    # chunk
    words: list[str] = []
    meta:  list[tuple] = []
    for p in pages:
        pw = p["text"].split()
        words.extend(pw)
        meta.extend([(p["source"], p["page"])] * len(pw))

    chunks = []
    step = chunk_size - chunk_overlap
    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        src, pg = meta[start]
        chunks.append({"text": " ".join(words[start:end]), "source": src, "page": pg})
        if end == len(words):
            break

    model      = SentenceTransformer(embed_model)
    embeddings = model.encode(
        [c["text"] for c in chunks],
        batch_size=BATCH_SIZE, show_progress_bar=False, normalize_embeddings=True,
    )
    return {"chunks": chunks, "embeddings": embeddings}


# ── Pretty printer ─────────────────────────────────────────────────────────────
def print_result(r: EvalResult, verbose: bool = False):
    c = r.config
    print(f"\n{'─'*70}")
    print(f"  Model:  {c.embed_model}")
    print(f"  Chunks: size={c.chunk_size} overlap={c.chunk_overlap}  |  "
          f"Total chunks in index: {r.index_size}")
    print(f"  Top-K:  {c.top_k}")
    print(f"  {'Precision@K':<16} {r.mean_precision:.3f}")
    print(f"  {'Recall@K':<16} {r.mean_recall:.3f}")
    print(f"  {'MRR':<16} {r.mean_mrr:.3f}")
    print(f"  {'NDCG@K':<16} {r.mean_ndcg:.3f}")
    print(f"  {'Latency (avg)':<16} {r.latency_ms:.1f} ms")
    if verbose:
        print()
        for qr in r.query_results:
            flag = "✓" if qr.relevant_found > 0 else "✗"
            print(f"  {flag} P={qr.precision_at_k:.2f} R={qr.recall_at_k:.2f} "
                  f"MRR={qr.mrr:.2f}  '{qr.query[:55]}'")


# ── Save to CSV ───────────────────────────────────────────────────────────────
def save_csv(results: list[EvalResult], path: str):
    rows = []
    for r in results:
        rows.append({
            "embed_model":     r.config.embed_model,
            "chunk_size":      r.config.chunk_size,
            "chunk_overlap":   r.config.chunk_overlap,
            "top_k":           r.config.top_k,
            "index_size":      r.index_size,
            "precision_at_k":  round(r.mean_precision, 4),
            "recall_at_k":     round(r.mean_recall,    4),
            "mrr":             round(r.mean_mrr,        4),
            "ndcg_at_k":       round(r.mean_ndcg,       4),
            "latency_ms":      round(r.latency_ms,      2),
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✓ Results saved → {path}")


# ── Detect which queries to use based on indexed sources ─────────────────────
def auto_select_queries(index: dict) -> list[dict]:
    sources = {c["source"] for c in index["chunks"]}
    if any("spotify" in s.lower() for s in sources):
        print("[eval] Detected Spotify index — using Spotify query set")
        return EVAL_QUERIES_SPOTIFY
    return EVAL_QUERIES


# ── CLI entry points ──────────────────────────────────────────────────────────
def run_quick(args):
    """Evaluate the existing sdk_index.pkl across top-K values."""
    INDEX_PATH = "sdk_index.pkl"
    if not os.path.exists(INDEX_PATH):
        print(f"No index found at {INDEX_PATH}. Run ingest first.")
        return

    with open(INDEX_PATH, "rb") as f:
        index = pickle.load(f)

    queries = auto_select_queries(index)
    print(f"\n[eval] Quick eval — {len(index['chunks'])} chunks, {len(queries)} queries")

    embed_model = "all-MiniLM-L6-v2"
    model       = SentenceTransformer(embed_model)
    results     = []

    for top_k in [4, 8, 12]:
        cfg = EvalConfig(embed_model=embed_model, chunk_size=400,
                         chunk_overlap=60, top_k=top_k)
        r   = evaluate_config(index, model, top_k, queries, cfg)
        print_result(r, verbose=(top_k == 8))
        results.append(r)

    if args.output:
        save_csv(results, args.output)

    print_summary(results)


def run_full_ablation(args):
    """
    Full ablation:
      3 embedding models × 3 chunk sizes × 3 top-K values = 27 combinations
    Requires --url to crawl fresh pages (or uses cached _pages.pkl if present).
    """
    EMBED_MODELS   = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "BAAI/bge-small-en-v1.5"]
    CHUNK_SIZES    = [200, 400, 600]
    CHUNK_OVERLAPS = [30,  60,  90]    # ~15% overlap ratio kept constant
    TOP_KS         = [4,   8,  12]

    PAGES_CACHE = "_eval_pages.pkl"

    # Crawl once, reuse for all ablations
    if args.recrawl and os.path.exists(PAGES_CACHE):
        os.remove(PAGES_CACHE)
        print(f"[eval] Deleted cached pages (--recrawl)")

    if os.path.exists(PAGES_CACHE):
        print(f"[eval] Loading cached pages from {PAGES_CACHE}")
        with open(PAGES_CACHE, "rb") as f:
            pages = pickle.load(f)
    else:
        if not args.url:
            print("--url required for full ablation (e.g. https://docs.stripe.com/sdks/ios)")
            print("Tip: use --no-filter to crawl entire domain for a larger corpus")
            return
        from ingest_url import crawl, SITE_CONFIGS, DEFAULT_CONFIG
        from urllib.parse import urlparse
        if getattr(args, 'no_filter', False):
            host = urlparse(args.url).netloc
            if host in SITE_CONFIGS:
                SITE_CONFIGS[host]["stay_under"] = None
            DEFAULT_CONFIG["stay_under"] = None
            print("[eval] --no-filter: crawling entire domain for larger corpus")
        pages = crawl(args.url, max_pages=args.max_pages)
        with open(PAGES_CACHE, "wb") as f:
            pickle.dump(pages, f)
        print(f"[eval] Pages cached → {PAGES_CACHE}")

    # Warn if corpus is too small for meaningful eval
    total_words = sum(len(p["text"].split()) for p in pages)
    est_chunks_400 = total_words // (400 - 60)
    if est_chunks_400 < 100:
        print(f"\n⚠️  WARNING: corpus is small (~{est_chunks_400} chunks at chunk_size=400).")
        print(f"   MRR scores at this scale are not meaningful for a resume.")
        print(f"   Run with --no-filter --max-pages 150 to get 200+ chunks.\n")

    # Pick query set
    source_hint = args.url or ""
    queries = EVAL_QUERIES_SPOTIFY if "spotify" in source_hint else EVAL_QUERIES

    total   = len(EMBED_MODELS) * len(CHUNK_SIZES) * len(TOP_KS)
    done    = 0
    results = []

    for embed_model, (chunk_size, chunk_overlap) in itertools.product(
        EMBED_MODELS, zip(CHUNK_SIZES, CHUNK_OVERLAPS)
    ):
        print(f"\n[ablation] Building index: model={embed_model}  "
              f"chunk={chunk_size}/overlap={chunk_overlap} ...")
        t0    = time.time()
        index = build_index(pages, embed_model, chunk_size, chunk_overlap)
        build_time = time.time() - t0
        print(f"           {len(index['chunks'])} chunks built in {build_time:.1f}s")

        model = SentenceTransformer(embed_model)

        for top_k in TOP_KS:
            done += 1
            cfg  = EvalConfig(embed_model=embed_model, chunk_size=chunk_size,
                              chunk_overlap=chunk_overlap, top_k=top_k)
            r    = evaluate_config(index, model, top_k, queries, cfg)
            print_result(r)
            results.append(r)
            print(f"  [{done}/{total} done]")

    if args.output:
        save_csv(results, args.output)

    print_summary(results)


def print_summary(results: list[EvalResult]):
    """Print ranked leaderboard by MRR."""
    print(f"\n{'═'*70}")
    print("  LEADERBOARD — ranked by MRR")
    print(f"{'═'*70}")
    ranked = sorted(results, key=lambda r: r.mean_mrr, reverse=True)
    print(f"  {'Model':<28} {'ChunkSz':>7} {'K':>3} {'P@K':>6} {'R@K':>6} "
          f"{'MRR':>6} {'NDCG':>6} {'ms':>6}")
    print(f"  {'-'*68}")
    for r in ranked[:10]:
        short_model = r.config.embed_model.split("/")[-1][:26]
        print(f"  {short_model:<28} {r.config.chunk_size:>7} {r.config.top_k:>3} "
              f"{r.mean_precision:>6.3f} {r.mean_recall:>6.3f} "
              f"{r.mean_mrr:>6.3f} {r.mean_ndcg:>6.3f} {r.latency_ms:>6.1f}")

    best = ranked[0]
    print(f"\n  ★ Best config:")
    print(f"    Model: {best.config.embed_model}")
    print(f"    Chunk size: {best.config.chunk_size} (overlap {best.config.chunk_overlap})")
    print(f"    Top-K: {best.config.top_k}")
    print(f"    MRR: {best.mean_mrr:.3f}  |  Precision@K: {best.mean_precision:.3f}  "
          f"|  NDCG: {best.mean_ndcg:.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval quality with ablations"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick",          action="store_true",
                      help="Eval existing sdk_index.pkl across top-K values")
    mode.add_argument("--full-ablation",  action="store_true",
                      help="Full 27-config ablation (models × chunk sizes × K)")

    parser.add_argument("--url",       help="SDK doc URL to crawl (for --full-ablation)")
    parser.add_argument("--max-pages", type=int, default=100,
                        help="Pages to crawl (default 100 — use 100+ for meaningful eval)")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable stay_under path filter — crawl entire domain for larger corpus")
    parser.add_argument("--recrawl",   action="store_true",
                        help="Delete cached _eval_pages.pkl and re-crawl")
    parser.add_argument("--output",    metavar="FILE", help="Save results to CSV")
    parser.add_argument("--verbose",   action="store_true")

    args = parser.parse_args()

    if args.quick:
        run_quick(args)
    elif args.full_ablation:
        run_full_ablation(args)
