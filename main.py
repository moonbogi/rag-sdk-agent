"""
main.py — FastAPI server
  POST /ingest        multipart PDF upload → index
  POST /ingest/url    crawl SDK doc site → index
  POST /generate      query → Swift sample app code
  GET  /health
  GET  /index/stats

Auth: pass X-API-Key header on all POST endpoints.
      Set API_KEY env var (defaults to "dev-key" for local dev).
"""

import os
import pickle
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

import db
from ingest     import ingest_pdf, INDEX_PATH
from ingest_url import ingest_url as ingest_from_url
from agent      import generate_swift_code

# ── API key auth ──────────────────────────────────────────────────────────────
API_KEY        = os.environ.get("API_KEY", "dev-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def require_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


app = FastAPI(
    title       = "SDK RAG Agent",
    description = "Upload SDK documentation PDFs → generate Swift iOS sample apps.",
    version     = "0.2.0",
)


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    if db.using_postgres():
        db.init_db()
        print("[startup] Postgres/pgvector ready")
    else:
        print("[startup] No DATABASE_URL — using local pickle index")


# ── Request / response models ─────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    query:       str  = Field(..., example="Show me how to initialize the SDK and authenticate a user")
    use_swiftui: bool = Field(True, description="True = SwiftUI output, False = UIKit")
    top_k:       int  = Field(8,    ge=1, le=20, description="Number of doc chunks to retrieve")


class SourceRef(BaseModel):
    source: str
    page:   str
    score:  float


class GenerateResponse(BaseModel):
    query:   str
    code:    str
    sources: list[SourceRef]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "backend": "postgres" if db.using_postgres() else "pickle"}


@app.get("/index/stats")
def index_stats():
    """Return chunk count and sources — works with both backends."""
    if db.using_postgres():
        stats = db.get_stats()
        return {"indexed": stats["chunks"] > 0, **stats}

    if not os.path.exists(INDEX_PATH):
        return {"indexed": False, "chunks": 0, "sources": []}
    with open(INDEX_PATH, "rb") as f:
        index = pickle.load(f)
    sources = sorted({c["source"] for c in index["chunks"]})
    return {"indexed": True, "chunks": len(index["chunks"]), "sources": sources}


class IngestUrlRequest(BaseModel):
    url:       str = Field(..., example="https://docs.stripe.com/sdks/ios")
    max_pages: int = Field(60, ge=1, le=200)


@app.post("/ingest/url", summary="Crawl an SDK doc website and ingest it",
          dependencies=[Security(require_api_key)])
def ingest_url_endpoint(req: IngestUrlRequest):
    try:
        n = ingest_from_url(req.url, max_pages=req.max_pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": f"Crawled '{req.url}'", "chunks_added": n}


@app.post("/ingest", summary="Upload an SDK documentation PDF",
          dependencies=[Security(require_api_key)])
async def ingest(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        n = ingest_pdf(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

    return {"message": f"Ingested '{file.filename}'", "chunks_added": n}


@app.post("/generate", response_model=GenerateResponse,
          summary="Generate a Swift sample app from the indexed SDK docs",
          dependencies=[Security(require_api_key)])
def generate(req: GenerateRequest):
    try:
        result = generate_swift_code(
            query       = req.query,
            use_swiftui = req.use_swiftui,
            top_k       = req.top_k,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return GenerateResponse(
        query   = result.query,
        code    = result.code,
        sources = [
            SourceRef(source=c.source, page=str(c.page), score=round(c.score, 4))
            for c in result.sources
        ],
    )
