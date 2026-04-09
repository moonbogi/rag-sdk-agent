"""
agent.py — retrieve relevant doc chunks, prompt Claude, return Swift code

Backend selection (automatic):
  DATABASE_URL set → pgvector (Postgres)
  DATABASE_URL unset → sdk_index.pkl (local pickle, original behaviour)
"""

import os
import pickle
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import anthropic
import db

INDEX_PATH   = "sdk_index.pkl"
EMBED_MODEL  = "all-MiniLM-L6-v2"
TOP_K        = 8
CLAUDE_MODEL = "claude-sonnet-4-6"

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


@dataclass
class RetrievedChunk:
    text: str; source: str; page: int; score: float


@dataclass
class GenerationResult:
    code: str; sources: list[RetrievedChunk]; query: str


def retrieve(query: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
    model = SentenceTransformer(EMBED_MODEL)
    q_emb = model.encode([query], normalize_embeddings=True)[0]

    if db.using_postgres():
        results = db.similarity_search(q_emb, top_k)
        return [
            RetrievedChunk(text=r["text"], source=r["source"],
                           page=r["page"], score=r["score"])
            for r in results
        ]

    # Pickle fallback
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("Index not found. Run ingest.py or ingest_url.py first.")
    with open(INDEX_PATH, "rb") as f:
        index = pickle.load(f)
    scores  = index["embeddings"] @ q_emb
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [
        RetrievedChunk(text=index["chunks"][i]["text"], source=index["chunks"][i]["source"],
                       page=index["chunks"][i]["page"], score=float(scores[i]))
        for i in top_idx
    ]


def generate_swift_code(query: str, use_swiftui: bool = True, top_k: int = TOP_K) -> GenerationResult:
    framework = "SwiftUI" if use_swiftui else "UIKit"
    print(f"\n[agent] Retrieving top {top_k} chunks ...")
    chunks = retrieve(query, top_k=top_k)
    for c in chunks:
        print(f"  {c.score:.3f}  {c.source} p.{c.page}")

    context = "\n\n---\n\n".join(
        f"[Source: {c.source}, Page {c.page} | similarity: {c.score:.3f}]\n{c.text}"
        for c in chunks
    )
    prompt = f"""## SDK Documentation Context\n\n{context}\n\n---\n\n## Task\n\n{query}\n\n
Generate a complete {framework} sample iOS app. Structure:\n
1. SDKManager.swift — service wrapper\n2. ContentView.swift — main UI\n3. AppEntry.swift — @main entry\n
Comment every SDK-specific call.\n"""

    print(f"[agent] Calling Claude ({CLAUDE_MODEL}) ...")
    client  = anthropic.Anthropic()
    message = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=4096, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return GenerationResult(code=message.content[0].text, sources=chunks, query=query)
