"""
agent.py — retrieve relevant doc chunks, prompt Claude, return Swift code

Backend selection (automatic):
  DATABASE_URL set → pgvector (Postgres)
  DATABASE_URL unset → sdk_index.pkl (local pickle, original behaviour)

Tracing (optional):
  Set LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY to enable Langfuse tracing.
  Every generate_swift_code call becomes a traced generation with:
    - query, framework, top_k
    - retrieved chunks + similarity scores
    - Claude model, token usage, latency
  If keys are not set, tracing is silently skipped.
"""

import os
import time
import pickle
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import anthropic
import db

INDEX_PATH      = "sdk_index.pkl"
EMBED_MODEL     = "all-MiniLM-L6-v2"
RERANK_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # ~67 MB
TOP_K           = 8
RERANK_FACTOR   = 2   # retrieve TOP_K * RERANK_FACTOR candidates, re-rank to TOP_K
CLAUDE_MODEL    = "claude-sonnet-4-6"

SYSTEM_PROMPT = """\
You are a senior iOS Swift engineer who specialises in SDK integrations.
Given SDK documentation excerpts, generate clean, idiomatic, production-quality
Swift sample app code.

Guidelines:
- Swift 5.9+ syntax; prefer async/await over callbacks
- Default to SwiftUI unless the caller specifies UIKit
- Import only what the SDK docs mention — do not hallucinate types or methods
- Wrap SDK calls in a dedicated service/manager class (e.g. SDKManager)
- Add concise inline comments on every SDK-specific line
- Provide a realistic app structure: App entry point, main View, service layer
- If a method name is unclear from context, note it with a TODO comment
- Output ONLY Swift code blocks — no prose outside fenced code blocks
"""


# ── Langfuse (optional) ────────────────────────────────────────────────────────
def _get_langfuse():
    """Return a Langfuse client if credentials are set, else None."""
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY")
    sk = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    if not (pk and sk):
        return None
    try:
        from langfuse import Langfuse
        return Langfuse(public_key=pk, secret_key=sk, host=host)
    except ImportError:
        print("[agent] langfuse not installed — pip install langfuse to enable tracing")
        return None


@dataclass
class RetrievedChunk:
    text: str; source: str; page: int; score: float


@dataclass
class GenerationResult:
    code: str; sources: list[RetrievedChunk]; query: str
    retrieve_ms: float = 0.0
    generate_ms: float = 0.0
    input_tokens: int  = 0
    output_tokens: int = 0


def _rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> tuple[list[RetrievedChunk], float]:
    """Cross-encoder re-ranking — scores every (query, chunk) pair and returns top_k."""
    from sentence_transformers import CrossEncoder
    t0      = time.perf_counter()
    model   = CrossEncoder(RERANK_MODEL)
    pairs   = [(query, c.text) for c in chunks]
    scores  = model.predict(pairs)
    ranked  = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    result  = [c for _, c in ranked[:top_k]]
    return result, (time.perf_counter() - t0) * 1000


def retrieve(query: str, top_k: int = TOP_K, rerank: bool = True) -> tuple[list[RetrievedChunk], float]:
    """Returns (chunks, latency_ms). Fetches top_k * RERANK_FACTOR candidates then re-ranks."""
    t0         = time.perf_counter()
    candidates = top_k * RERANK_FACTOR if rerank else top_k
    model      = SentenceTransformer(EMBED_MODEL)
    q_emb      = model.encode([query], normalize_embeddings=True)[0]

    if db.using_postgres():
        results = db.similarity_search(q_emb, candidates)
        chunks  = [
            RetrievedChunk(text=r["text"], source=r["source"],
                           page=r["page"], score=r["score"])
            for r in results
        ]
    else:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError("Index not found. Run ingest.py or ingest_url.py first.")
        with open(INDEX_PATH, "rb") as f:
            index = pickle.load(f)
        scores  = index["embeddings"] @ q_emb
        top_idx = np.argsort(scores)[::-1][:candidates]
        chunks  = [
            RetrievedChunk(text=index["chunks"][i]["text"], source=index["chunks"][i]["source"],
                           page=index["chunks"][i]["page"], score=float(scores[i]))
            for i in top_idx
        ]

    embed_ms = (time.perf_counter() - t0) * 1000

    if rerank and len(chunks) > top_k:
        chunks, rerank_ms = _rerank(query, chunks, top_k)
        return chunks, embed_ms + rerank_ms

    return chunks, embed_ms


def generate_swift_code(query: str, use_swiftui: bool = True, top_k: int = TOP_K,
                        rerank: bool = True) -> GenerationResult:
    framework = "SwiftUI" if use_swiftui else "UIKit"
    langfuse  = _get_langfuse()

    # ── Trace root ────────────────────────────────────────────────────────────
    trace = None
    if langfuse:
        trace = langfuse.trace(
            name   = "generate_swift_code",
            input  = {"query": query, "framework": framework, "top_k": top_k, "rerank": rerank},
            tags   = ["rag", "swift-codegen"],
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    print(f"\n[agent] Retrieving top {top_k} chunks (rerank={rerank}) ...")
    chunks, retrieve_ms = retrieve(query, top_k=top_k, rerank=rerank)
    for c in chunks:
        print(f"  {c.score:.3f}  {c.source} p.{c.page}")

    if trace:
        trace.span(
            name   = "retrieval",
            input  = {"query": query, "top_k": top_k},
            output = [{"source": c.source, "page": str(c.page),
                       "score": round(c.score, 4), "snippet": c.text[:120]}
                      for c in chunks],
            metadata = {"latency_ms": round(retrieve_ms, 1),
                        "embed_model": EMBED_MODEL,
                        "rerank_model": RERANK_MODEL if rerank else None,
                        "rerank_factor": RERANK_FACTOR if rerank else None,
                        "backend": "postgres" if db.using_postgres() else "pickle"},
        )

    # ── Claude call ───────────────────────────────────────────────────────────
    context = "\n\n---\n\n".join(
        f"[Source: {c.source}, Page {c.page} | similarity: {c.score:.3f}]\n{c.text}"
        for c in chunks
    )
    prompt = (
        f"## SDK Documentation Context\n\n{context}\n\n---\n\n"
        f"## Task\n\n{query}\n\n"
        f"Generate a complete {framework} sample iOS app. Structure:\n"
        f"1. SDKManager.swift — service wrapper\n"
        f"2. ContentView.swift — main UI\n"
        f"3. AppEntry.swift — @main entry\n"
        f"Comment every SDK-specific call.\n"
    )

    generation = None
    if trace:
        generation = trace.generation(
            name   = "claude-codegen",
            model  = CLAUDE_MODEL,
            input  = [{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user",   "content": prompt}],
        )

    print(f"[agent] Calling Claude ({CLAUDE_MODEL}) ...")
    t0      = time.perf_counter()
    client  = anthropic.Anthropic()
    message = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=4096, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    generate_ms   = (time.perf_counter() - t0) * 1000
    code          = message.content[0].text
    input_tokens  = message.usage.input_tokens
    output_tokens = message.usage.output_tokens

    if generation:
        generation.end(
            output   = code,
            usage    = {"input": input_tokens, "output": output_tokens},
            metadata = {"latency_ms": round(generate_ms, 1)},
        )

    if trace:
        trace.update(
            output   = {"code_length": len(code), "framework": framework},
            metadata = {
                "total_latency_ms": round(retrieve_ms + generate_ms, 1),
                "retrieve_ms":      round(retrieve_ms, 1),
                "generate_ms":      round(generate_ms, 1),
                "input_tokens":     input_tokens,
                "output_tokens":    output_tokens,
            },
        )
        langfuse.flush()

    print(f"[agent] Done — retrieve {retrieve_ms:.0f}ms | generate {generate_ms:.0f}ms | "
          f"tokens in={input_tokens} out={output_tokens}")

    return GenerationResult(
        code=code, sources=chunks, query=query,
        retrieve_ms=retrieve_ms, generate_ms=generate_ms,
        input_tokens=input_tokens, output_tokens=output_tokens,
    )
