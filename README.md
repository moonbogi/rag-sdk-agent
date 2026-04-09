# SDK RAG Agent

Ingests SDK documentation PDFs and generates Swift iOS sample apps using retrieval-augmented generation.

## Architecture

```
PDF ──► extract_pages() ──► chunk_pages() ──► SentenceTransformer embed
                                                        │
                                                sdk_index.pkl  (numpy array + metadata)
                                                        │
query ──────────────────────────────► cosine similarity search
                                                        │
                                              top-K chunks as context
                                                        │
                                           Claude claude-sonnet-4-6
                                                        │
                                            Swift sample app code  ◄──
```

**Embedding model:** `all-MiniLM-L6-v2` — runs fully local, ~80 MB, no API key needed.  
**Vector store:** a single `sdk_index.pkl` file (numpy array + chunk metadata). Appending more PDFs grows the index in place.

---

## Supported SDK Sources

### Stripe iOS SDK
Stripe's docs are web-only. Crawl the iOS SDK section directly:
```bash
python cli.py ingest-url https://docs.stripe.com/sdks/ios --max-pages 80
```
Then generate:
```bash
python cli.py generate "set up PaymentSheet and handle a card payment"
python cli.py generate "implement Apple Pay with STPApplePayContext"
```

### Spotify iOS SDK
Crawl Spotify's iOS developer docs:
```bash
python cli.py ingest-url https://developer.spotify.com/documentation/ios --max-pages 40
```
Then generate:
```bash
python cli.py generate "authenticate with Spotify and connect SPTAppRemote"
python cli.py generate "subscribe to player state and show now playing UI"
```

### PDF docs (any SDK)
```bash
python cli.py ingest path/to/sdk-docs.pdf
```

### Mix sources — same index
Both SDKs can coexist. Retrieval is scoped by cosine similarity:
```bash
python cli.py ingest-url https://docs.stripe.com/sdks/ios
python cli.py ingest-url https://developer.spotify.com/documentation/ios
python cli.py stats   # → shows both sources + chunk counts
```

---

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Usage

### CLI (no server)

```bash
# 1. Ingest your SDK docs
python cli.py ingest stripe-ios-sdk.pdf plaid-link-ios.pdf

# 2. Show what's in the index
python cli.py stats

# 3. Generate a SwiftUI sample app
python cli.py generate "initialize the SDK and handle authentication"

# 4. Generate UIKit instead
python cli.py generate "display a transaction list" --uikit

# 5. Save output to a file
python cli.py generate "card scanning flow" --output CardScanView.swift
```

### API Server

```bash
uvicorn main:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

**Ingest a PDF:**
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@stripe-ios-sdk.pdf"
```

**Generate Swift code:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "initialize the SDK and authenticate a user", "use_swiftui": true}'
```

**Check index stats:**
```bash
curl http://localhost:8000/index/stats
```

---

## Tuning

| Parameter | Location | Effect |
|---|---|---|
| `CHUNK_SIZE` | `ingest.py` | Larger = more context per chunk, but noisier retrieval |
| `CHUNK_OVERLAP` | `ingest.py` | Higher = fewer missed concepts at boundaries |
| `TOP_K` | `agent.py` | More chunks = richer context, higher token cost |
| `EMBED_MODEL` | both | Swap to `all-mpnet-base-v2` for better quality, slower speed |

## Extending

- **Persistent vector DB:** swap the pickle index in `ingest.py`/`agent.py` for ChromaDB or Qdrant — the chunk/embedding structure stays the same.
- **More output types:** add a `framework` field to `GenerateRequest` and extend the prompt in `agent.py`.
- **Re-ranking:** add a cross-encoder re-ranking pass after the initial cosine retrieval in `agent.py`.
