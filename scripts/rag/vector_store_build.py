#!/usr/bin/env python3
from __future__ import annotations
import os
"""
Build a local Chroma vector store from a released PDF bundle for the RAG-Workflow project.

This script downloads the latest PDF bundle from the InterviewPrep GitHub release,
extracts the PDF files, derives a heading (H1) for each document based on its textual
content, and then embeds the content into a Chroma vector store.  The first non‑empty
line of each PDF is used as the H1 heading unless that line starts with "Contents"
(case insensitive), in which case the next non‑empty line becomes the H1.  Tags are
derived from the PDF file's path and the computed H1, similar to the original Markdown
workflow.

Usage example:
  ./scripts/bin/run_venv.sh scripts/rag/vector_store_build.py \
      --persist ./.chroma \
      --chunk-size 800 --chunk-overlap 120 \
      --local --model sentence-transformers/all-MiniLM-L6-v2 \
      --bundle-url https://github.com/brandon-benge/InterviewPrep/releases/download/latest/pdfs-bundle.tar.gz

This script requires the `pdftotext` utility from Poppler to be installed. On macOS,
you can install it via Homebrew (`brew install poppler`). On Debian/Ubuntu, use
`apt install poppler-utils`.  The script will abort with an error if `pdftotext`
is not available.
"""

import argparse
import sys
import shutil
import re
import tarfile
import urllib.request
from urllib.parse import urlparse, unquote
import tempfile
import subprocess
from pathlib import Path
from typing import List
from .build_utils import download_bundle, safe_extract, ensure_pdftotext
from .pdf_pipeline import extract_pdf_texts, extract_keywords_tfidf
from .split_pipeline import config_split_params, build_tokenizer, count_tokens_and_log, auto_cap
from .embed_pipeline import get_embedder, write_embeddings


def lazy_import():
    """Import vector store + splitter using Haystack components."""
    try:
        from haystack_integrations.document_stores.chroma import ChromaDocumentStore  # type: ignore
        from haystack import Document  # type: ignore
        from haystack.components.preprocessors import DocumentSplitter  # type: ignore
        return ChromaDocumentStore, Document, DocumentSplitter
    except ImportError as e:
        raise RuntimeError(f'Haystack modules missing ({e}); install haystack-ai and chroma-haystack.')

def embedding_fn(model: str):
    try:
        from haystack.components.embedders import SentenceTransformersDocumentEmbedder  # type: ignore
        return SentenceTransformersDocumentEmbedder(model=model or 'sentence-transformers/all-MiniLM-L6-v2')
    except ImportError:
        raise RuntimeError('SentenceTransformersDocumentEmbedder not available; install haystack-ai.')

TAG_TOKEN_RE = re.compile(r'[^a-z0-9]+')

def slugify(text: str) -> str:
    return TAG_TOKEN_RE.sub('-', text.lower()).strip('-')

def derive_tags(path: Path, h1: str | None) -> list[str]:
    """
    Derive lightweight tags from a PDF path and optional H1 heading.
    Strategy:
      - Include an H1 slug first (if present)
      - Up to first 3 directory names
      - File stem tokens (split on non‑alphanumerics)
    """
    parts = list(path.parts)
    dir_tags = [p.lower() for p in parts[:-1][:3]]
    stem_tokens = [t for t in TAG_TOKEN_RE.split(path.stem.lower()) if t]
    ordered: list[str] = []
    seen: set[str] = set()
    if h1:
        h1_slug = slugify(h1)[:60]
        if h1_slug and h1_slug not in seen:
            seen.add(h1_slug)
            ordered.append(h1_slug)
    for t in (*dir_tags, *stem_tokens):
        if t and t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered

def build(args):
    """
    Download the PDF bundle, extract documents, embed and persist for the RAG-Workflow.
    """
    ChromaDocumentStore, Document, DocumentSplitter = lazy_import()

    # Determine persistence directory and chunk settings
    persist_dir = Path(args.persist)
    if getattr(args, 'persist_by_model', False):
        # Create a filesystem-safe slug from the model name
        # Lowercase, replace non-alphanumerics with '-', and collapse repeats
        raw = str(args.model)
        slug = re.sub(r'[^a-zA-Z0-9]+', '-', raw).strip('-').lower()
        persist_dir = persist_dir / slug

    # Clean up existing persistence directory if --force is set
    if args.force and persist_dir.exists():
        shutil.rmtree(persist_dir)

    # Location of PDF bundle: always download (fail if unavailable)
    pdf_url = args.bundle_url

    # Temporary directory for extraction
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Derive the output filename from the URL path (fallback to a generic name)
        parsed = urlparse(pdf_url)
        fname = Path(unquote(parsed.path)).name or 'bundle.tar.gz'
        tar_path = tmp_path / fname

        # Always download the bundle; do not use any local tarball.
        print(f'Downloading PDF bundle from {pdf_url}...')
        try:
            download_bundle(pdf_url, tar_path)
        except Exception as e:
            print(f'ERROR: failed to download PDF bundle: {e}')
            return 1

        # Extract the tarball with a safe filter to avoid future deprecation warnings
        print('Extracting PDF bundle...')
        try:
            safe_extract(tar_path, tmp_path)
        except Exception as e:
            print(f'ERROR: failed to extract PDF bundle: {e}')
            return 1

        # Find all PDFs in the extracted directory (e.g. pdfs/)
        pdf_paths: List[Path] = list(tmp_path.rglob('*.pdf'))
        if not pdf_paths:
            print('No PDF files found in the extracted bundle.')
            return 1

        print(f'Found {len(pdf_paths)} PDF files. Parsing content...')

        # Ensure the pdftotext command is available before attempting to parse PDFs
        try:
            ensure_pdftotext()
        except Exception as e:
            print(f"ERROR: {e}")
            return 1

        docs = []
        pdf_texts = []
        pdf_h1s = []
        pdf_paths_for_meta = []

        texts, h1s, kept_paths = extract_pdf_texts(pdf_paths)
        pdf_texts.extend(texts)
        pdf_h1s.extend(h1s)
        pdf_paths_for_meta.extend(kept_paths)

        # Extract keywords for each PDF using TF-IDF (fail if unavailable or empty)
        if not pdf_texts:
            print('ERROR: No extracted text to tag from PDFs.')
            return 1
        try:
            top_n = getattr(args, 'tfidf_top_n', None)
            if top_n is None:
                top_n = 20
            keywords_lists = extract_keywords_tfidf(pdf_texts, top_n=top_n)
        except Exception as e:
            print(f'ERROR: keyword extraction failed: {e}')
            return 1
        if any(not kw for kw in keywords_lists):
            print('ERROR: One or more PDFs produced no tags; aborting build. Ensure scikit-learn is installed and PDFs contain extractable text.')
            return 1

        for text, h1, pdf_path, keywords in zip(pdf_texts, pdf_h1s, pdf_paths_for_meta, keywords_lists):
            meta = {
                'source': str(pdf_path),
                'tags': ','.join(keywords),
                'h1': h1
            }
            docs.append(Document(content=text, meta=meta))

        print(f'PDF processing complete. {len(docs)} documents prepared for embedding.')

        if not docs:
            print('No valid PDF documents to embed.')
            return 1

        # Configure splitting based on split_by with sensible defaults
        split_by = getattr(args, 'split_by', 'sentence')
        split_by, split_length, split_overlap = config_split_params(
            split_by, getattr(args, 'chunk_size', None), getattr(args, 'chunk_overlap', None)
        )

        print(f"Splitting by '{split_by}' with length={split_length}, overlap={split_overlap} ...")
        # Persist resolved values back to args for downstream references
        args.chunk_size = split_length
        args.chunk_overlap = split_overlap
        splitter = DocumentSplitter(split_by=split_by, split_length=split_length, split_overlap=split_overlap)
        splitter.warm_up()
        splits = splitter.run(documents=docs)["documents"]
        tokenizer, tokenizer_type = build_tokenizer(True, args.model)
        if tokenizer is not None:
            splits, limit, tok_counts, avg_tokens = count_tokens_and_log(splits, tokenizer, tokenizer_type)
            cap = getattr(args, 'max_tokens_per_chunk', None)
            splits, applied_cap = auto_cap(True, tokenizer, tokenizer_type, tok_counts, splits, cap)
        else:
            avg_len = sum(len(s.content or '') for s in splits) // max(len(splits), 1)
            print(f"Generated {len(splits)} chunks (avg ~{avg_len} chars).")

        # Initialize document store and embeddings
        document_store = ChromaDocumentStore(persist_path=str(persist_dir))
        embedder = get_embedder(args.model)
        print('Embedding & persisting to Chroma...')
        embedded_docs = write_embeddings(document_store, embedder, splits)

        # Sanitize metadata for Chroma (supports only str, int, float, bool)
        for d in embedded_docs:
            meta = dict(d.meta or {})
            for k in list(meta.keys()):
                v = meta[k]
                # Drop splitter internals or coerce when reasonable
                if k in {
                    "_split_overlap",
                    "_split_length",
                    "_split_id",
                    "_split_offset_start",
                    "_split_offset_end",
                    "_split_parent_id",
                }:
                    # Prefer dropping; alternatively, coerce overlap to configured int
                    if k == "_split_overlap":
                        try:
                            # Cast numpy scalars or sequences to int fallback
                            if hasattr(v, "item"):
                                meta[k] = int(v.item())
                            elif isinstance(v, (list, tuple)):
                                meta[k] = int(args.chunk_overlap)
                            elif isinstance(v, (str, int, float, bool)):
                                # keep as-is if already acceptable
                                pass
                            else:
                                meta[k] = int(args.chunk_overlap)
                        except Exception:
                            meta.pop(k, None)
                    else:
                        meta.pop(k, None)
                elif not isinstance(v, (str, int, float, bool)):
                    # Best-effort cast; otherwise drop
                    try:
                        if hasattr(v, "item"):
                            meta[k] = v.item()
                        else:
                            meta[k] = str(v)
                    except Exception:
                        meta.pop(k, None)
            d.meta = meta

        document_store.write_documents(embedded_docs)

        print(f'Done. Persist dir: {persist_dir}')
        return 0

def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--persist', required=True, help='Persistence directory (required)')
    ap.add_argument('--split-by', choices=['word','sentence','line','passage','page','document'], default='sentence', help='Unit to split by')
    ap.add_argument('--chunk-size', type=int, help='Chunk size (auto if omitted, depends on split_by)')
    ap.add_argument('--chunk-overlap', type=int, help='Chunk overlap (auto if omitted, depends on split_by)')
    ap.add_argument('--max-tokens-per-chunk', type=int, help='Optional token cap per chunk; auto-derived from model if omitted')
    ap.add_argument('--model', required=True, help='Embedding model name (required)')
    ap.add_argument('--bundle-url', required=True, help='Download this tar.gz bundle of PDFs (required)')
    ap.add_argument('--tfidf-top-n', type=int, help='Number of TF-IDF keywords to extract per PDF (default 20)')
    # Local-only embeddings
    ap.add_argument('--force', action='store_true', help='Rebuild even if directory exists')
    ap.add_argument('--persist-by-model', action='store_true', help='Persist under a model-derived subfolder')
    return ap.parse_args(argv)

if __name__ == '__main__':
    args = parse(sys.argv[1:])
    sys.exit(build(args))