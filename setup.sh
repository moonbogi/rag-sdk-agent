#!/usr/bin/env bash
# setup.sh — one-shot environment setup for rag-sdk-agent
set -euo pipefail

echo "==> Creating virtual environment ..."
python3 -m venv venv

echo "==> Activating and upgrading pip ..."
source venv/bin/activate
pip install --upgrade pip --quiet

echo "==> Installing dependencies ..."
pip install -r requirements.txt

echo "==> Pre-downloading embedding models used in eval (avoids timeout during ablation) ..."
python - <<'PY'
from sentence_transformers import SentenceTransformer
for m in ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "BAAI/bge-small-en-v1.5"]:
    print(f"  Downloading {m} ...")
    SentenceTransformer(m)
print("  Models cached.")
PY

echo ""
echo "==> Setup complete."
echo ""
echo "Next steps:"
echo "  source venv/bin/activate"
echo "  export ANTHROPIC_API_KEY=sk-ant-..."
echo ""
echo "--- Ingest docs ---"
echo "  python cli.py ingest-url https://docs.stripe.com/sdks/ios --max-pages 80"
echo ""
echo "--- Evaluate retrieval quality ---"
echo "  # Quick eval on existing index (5 min):"
echo "  python eval.py --quick --output results.csv"
echo ""
echo "  # Full ablation — 27 configs, ranked leaderboard (15-30 min):"
echo "  python eval.py --full-ablation --url https://docs.stripe.com/sdks/ios --output ablation.csv"
echo ""
echo "--- Generate Swift code ---"
echo "  python cli.py generate 'set up PaymentSheet and handle a card payment'"
