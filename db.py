"""
db.py — pgvector-backed vector store, drop-in replacement for sdk_index.pkl

Schema:
  chunks(id, source, page, text, embedding vector(384))

Uses the same embedding shape as the pickle index (all-MiniLM-L6-v2 = 384 dims).
Falls back to pickle index automatically if DATABASE_URL is not set,
so local dev without Postgres still works.
"""

from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np

DATABASE_URL = os.environ.get("DATABASE_URL")  # set in docker-compose / Railway
PICKLE_PATH  = "sdk_index.pkl"
VECTOR_DIM   = 384   # all-MiniLM-L6-v2


# ── Connection pool (lazy) ─────────────────────────────────────────────────────
_pool = None

def _get_pool():
    global _pool
    if _pool is None:
        import psycopg2
        from psycopg2 import pool
        _pool = pool.SimpleConnectionPool(1, 10, DATABASE_URL)
    return _pool


def get_conn():
    return _get_pool().getconn()


def put_conn(conn):
    _get_pool().putconn(conn)


# ── Init schema ────────────────────────────────────────────────────────────────
def init_db():
    """Create the pgvector extension and chunks table if they don't exist."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id        SERIAL PRIMARY KEY,
                    source    TEXT NOT NULL,
                    page      TEXT NOT NULL,
                    text      TEXT NOT NULL,
                    embedding vector({VECTOR_DIM}) NOT NULL
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                ON chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            conn.commit()
    finally:
        put_conn(conn)


# ── Write ──────────────────────────────────────────────────────────────────────
def upsert_chunks(chunks: list[dict], embeddings: np.ndarray):
    """
    Insert chunks + embeddings into Postgres.
    Deduplicates by (source, page, text) — same logic as the pickle dedup.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            for chunk, emb in zip(chunks, embeddings):
                vec = emb.tolist()
                cur.execute("""
                    INSERT INTO chunks (source, page, text, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT DO NOTHING;
                """, (str(chunk["source"]), str(chunk["page"]), chunk["text"], vec))
        conn.commit()
    finally:
        put_conn(conn)


# ── Read ───────────────────────────────────────────────────────────────────────
def similarity_search(query_embedding: np.ndarray, top_k: int) -> list[dict]:
    """
    Cosine similarity search using pgvector's <=> operator.
    Returns list of {text, source, page, score} dicts.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            vec = query_embedding.tolist()
            cur.execute("""
                SELECT source, page, text,
                       1 - (embedding <=> %s::vector) AS score
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (vec, vec, top_k))
            rows = cur.fetchall()
        return [
            {"source": r[0], "page": r[1], "text": r[2], "score": float(r[3])}
            for r in rows
        ]
    finally:
        put_conn(conn)


def get_stats() -> dict:
    """Return chunk count and distinct sources."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*), ARRAY_AGG(DISTINCT source) FROM chunks;")
            count, sources = cur.fetchone()
        return {"chunks": count, "sources": sorted(sources or [])}
    finally:
        put_conn(conn)


def chunk_exists(text: str) -> bool:
    """Dedup check — same as pickle's md5 fingerprint approach but simpler."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM chunks WHERE text = %s LIMIT 1;", (text,))
            return cur.fetchone() is not None
    finally:
        put_conn(conn)


# ── Fallback: load from pickle ─────────────────────────────────────────────────
def load_pickle_index() -> dict:
    if not os.path.exists(PICKLE_PATH):
        return {"chunks": [], "embeddings": None}
    with open(PICKLE_PATH, "rb") as f:
        return pickle.load(f)


# ── Unified interface (auto-selects backend) ───────────────────────────────────
def using_postgres() -> bool:
    return bool(DATABASE_URL)
