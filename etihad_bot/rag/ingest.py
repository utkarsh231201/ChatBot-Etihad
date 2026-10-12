"""CLI: python -m etihad_bot.rag.ingest [--pdf file ...] [--url https://...]"""
import argparse

from etihad_bot.config import get_settings
from etihad_bot.rag.loaders import load_sources
from etihad_bot.rag.pipeline import RAGPipeline


def main():
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", nargs="*", default=None, help="PDF paths (default: the policy PDF)")
    ap.add_argument("--url", nargs="*", default=[])
    ap.add_argument("--chunk-size", type=int, default=s.chunk_size)
    ap.add_argument("--chunk-overlap", type=int, default=s.chunk_overlap)
    a = ap.parse_args()

    # Indexing flow: PDF loader -> text splitter -> BGE embeddings -> ChromaDB (policy PDF only)
    pdfs = a.pdf if a.pdf is not None else [s.policy_pdf]
    docs = load_sources(pdfs, a.url)
    pipe = RAGPipeline(chunk_size=a.chunk_size, chunk_overlap=a.chunk_overlap)
    n = pipe.ingest(docs)
    print(f"Loaded {len(docs)} sections -> {n} chunks into '{pipe.collection}' (total {pipe.count()})")


if __name__ == "__main__":
    main()
