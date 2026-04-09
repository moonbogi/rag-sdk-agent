#!/usr/bin/env python3
"""
cli.py — quick terminal interface (no server needed)

Examples:
  # Ingest one or more PDFs
  python cli.py ingest docs/stripe-ios-sdk.pdf docs/plaid-ios.pdf

  # Generate Swift code
  python cli.py generate "initialize the SDK and handle authentication errors"

  # Generate UIKit instead of SwiftUI
  python cli.py generate "display a transaction list" --uikit

  # Show index stats
  python cli.py stats
"""

import argparse
import sys
import textwrap
from ingest     import ingest_pdf, INDEX_PATH
from ingest_url import ingest_url
from agent      import generate_swift_code
import os, pickle


def cmd_ingest(args):
    for path in args.pdfs:
        n = ingest_pdf(path)
        print(f"✓ {path}  →  {n} new chunks added\n")


def cmd_ingest_url(args):
    n = ingest_url(args.url, max_pages=args.max_pages)
    print(f"✓ {args.url}  →  {n} new chunks added\n")


def cmd_generate(args):
    use_swiftui = not args.uikit
    framework   = "UIKit" if args.uikit else "SwiftUI"
    print(f"Framework: {framework} | Top-K: {args.top_k}")

    result = generate_swift_code(
        query       = args.query,
        use_swiftui = use_swiftui,
        top_k       = args.top_k,
    )

    print("\n" + "═" * 70)
    print("SOURCES RETRIEVED")
    print("═" * 70)
    for c in result.sources:
        print(f"  [{c.score:.3f}]  {c.source}  p.{c.page}")

    print("\n" + "═" * 70)
    print("GENERATED SWIFT CODE")
    print("═" * 70 + "\n")
    print(result.code)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result.code)
        print(f"\n✓ Saved to {args.output}")


def cmd_stats(_args):
    if not os.path.exists(INDEX_PATH):
        print("No index found. Run:  python cli.py ingest <pdf>")
        return
    with open(INDEX_PATH, "rb") as f:
        index = pickle.load(f)
    sources = sorted({c["source"] for c in index["chunks"]})
    print(f"Index: {INDEX_PATH}")
    print(f"Chunks: {len(index['chunks'])}")
    print(f"Sources ({len(sources)}):")
    for s in sources:
        count = sum(1 for c in index["chunks"] if c["source"] == s)
        print(f"  {s}  ({count} chunks)")


def main():
    parser = argparse.ArgumentParser(
        prog="sdk-rag",
        description="SDK documentation RAG agent → Swift sample app generator",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest PDF(s) into the local index")
    p_ingest.add_argument("pdfs", nargs="+", metavar="PDF")

    # ingest-url
    p_url = sub.add_parser("ingest-url", help="Crawl an SDK doc website and ingest it")
    p_url.add_argument("url", help="Start URL (e.g. https://docs.stripe.com/sdks/ios)")
    p_url.add_argument("--max-pages", type=int, default=60, help="Max pages to crawl (default 60)")

    # generate
    p_gen = sub.add_parser("generate", help="Generate Swift code for a task")
    p_gen.add_argument("query", help="What you want the sample app to do")
    p_gen.add_argument("--uikit",  action="store_true", help="Use UIKit instead of SwiftUI")
    p_gen.add_argument("--top-k",  type=int, default=8,  help="Number of chunks to retrieve (default 8)")
    p_gen.add_argument("--output", metavar="FILE",        help="Write code to this file")

    # stats
    sub.add_parser("stats", help="Show index statistics")

    args = parser.parse_args()

    dispatch = {"ingest": cmd_ingest, "ingest-url": cmd_ingest_url, "generate": cmd_generate, "stats": cmd_stats}
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
