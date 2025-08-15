from __future__ import annotations
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords_tfidf(texts: list[str], top_n: int = 5) -> list[list[str]]:
    """Extract top N keywords for each document using TF-IDF."""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    keywords_per_doc = []
    for doc_idx in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix.getrow(doc_idx)
        top_indices = row.toarray()[0].argsort()[-top_n:][::-1]
        keywords = [feature_names[i] for i in top_indices if row[0, i] > 0]
        keywords_per_doc.append(keywords)
    return keywords_per_doc
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords_tfidf(texts: list[str], top_n: int = 5) -> list[list[str]]:
    """Extract top N keywords for each document using TF-IDF."""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    keywords_per_doc = []
    for doc_idx in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix.getrow(doc_idx)
        top_indices = row.toarray()[0].argsort()[-top_n:][::-1]
        keywords = [feature_names[i] for i in top_indices if row[0, i] > 0]
        keywords_per_doc.append(keywords)
    return keywords_per_doc
#!/usr/bin/env python3
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
  ./scripts/bin/run_venv.sh scripts/rag/vector_store_build_rag_modified.py \
      --persist ./chroma_store \
      --chunk-size 800 --chunk-overlap 120 --local

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
import tempfile
import subprocess
from pathlib import Path
from typing import List


def lazy_import():
    """Import vector store + splitter using new package names with fallback."""
    try:
        from langchain_chroma import Chroma  # type: ignore
    except Exception:
        from langchain_community.vectorstores import Chroma  # type: ignore
    from langchain.docstore.document import Document  # type: ignore
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
    return Chroma, Document, RecursiveCharacterTextSplitter

def embedding_fn(local: bool, model: str):
    if local:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
        except Exception:
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
        return HuggingFaceEmbeddings(model_name=model or 'sentence-transformers/all-MiniLM-L6-v2')
    else:
        from langchain_openai import OpenAIEmbeddings  # type: ignore
        return OpenAIEmbeddings(model=model or 'text-embedding-3-small')

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
    Chroma, Document, TextSplitter = lazy_import()

    # Determine persistence directory and chunk settings
    persist_dir = Path(args.persist)

    # Clean up existing persistence directory if --force is set
    if args.force and persist_dir.exists():
        shutil.rmtree(persist_dir)

    # Location of PDF bundle: try local file first, else download
    pdf_url = "https://github.com/brandon-benge/InterviewPrep/releases/download/latest/pdfs-bundle.tar.gz"
    local_tar = Path('pdfs-bundle.tar.gz')

    # Temporary directory for extraction
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        tar_path = tmp_path / 'pdfs-bundle.tar.gz'

        # Attempt to use an existing local tarball if present
        if local_tar.exists():
            try:
                shutil.copy(local_tar, tar_path)
                print(f'Using local PDF bundle: {local_tar}')
            except Exception as e:
                print(f'ERROR: could not copy local PDF bundle: {e}')
                return 1
        else:
            # Download the tarball
            print(f'Downloading PDF bundle from {pdf_url}...')
            try:
                urllib.request.urlretrieve(pdf_url, tar_path)
            except Exception as e:
                print(f'ERROR: failed to download PDF bundle: {e}')
                return 1

        # Extract the tarball with a safe filter to avoid future deprecation warnings
        print('Extracting PDF bundle...')
        try:
            with tarfile.open(tar_path, 'r:gz') as tf:
                tf.extractall(tmp_path, filter='data')
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
        if shutil.which('pdftotext') is None:
            print("ERROR: 'pdftotext' command not found on your system.")
            print("Please install poppler-utils (e.g. `brew install poppler` on macOS) "
                  "or ensure that the 'pdftotext' binary is in your PATH.")
            return 1

        docs = []
        pdf_texts = []
        pdf_h1s = []
        pdf_paths_for_meta = []

        for pdf_path in pdf_paths:
            try:
                result = subprocess.run(
                    ['pdftotext', str(pdf_path), '-'],
                    capture_output=True,
                    text=True,
                    check=True
                )
            except Exception as e:
                print(f'WARN: failed to extract text from {pdf_path}: {e}')
                continue

            text = result.stdout or ''
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if not lines:
                continue
            first_line = lines[0]
            if first_line.lower().startswith('contents') and len(lines) > 1:
                h1 = lines[1]
            else:
                h1 = first_line

            pdf_texts.append(text)
            pdf_h1s.append(h1)
            pdf_paths_for_meta.append(pdf_path)

        # Extract keywords for each PDF using TF-IDF
        if pdf_texts:
            keywords_lists = extract_keywords_tfidf(pdf_texts, top_n=5)
        else:
            keywords_lists = [[] for _ in pdf_texts]

        for text, h1, pdf_path, keywords in zip(pdf_texts, pdf_h1s, pdf_paths_for_meta, keywords_lists):
            meta = {
                'source': str(pdf_path),
                'tags': ','.join(keywords),
                'h1': h1
            }
            docs.append(Document(page_content=text, metadata=meta))

        print(f'PDF processing complete. {len(docs)} documents prepared for embedding.')

        if not docs:
            print('No valid PDF documents to embed.')
            return 1

        # Use size-based splitting since PDFs lack section headings
        splitter = TextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        splits = splitter.split_documents(docs)
        avg_len = sum(len(s.page_content) for s in splits) // max(len(splits), 1)
        print(f'Generated {len(splits)} chunks (avg ~{avg_len} chars).')

        # Initialize embeddings and persist to Chroma
        emb = embedding_fn(args.local, args.model)
        print('Embedding & persisting to Chroma...')
        vs = Chroma.from_documents(splits, embedding=emb, persist_directory=args.persist)
        try:
            vs.persist()
        except Exception:
            pass
        print(f'Done. Persist dir: {args.persist}')
        return 0

def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--persist', default='.chroma', help='Persistence directory')
    ap.add_argument('--chunk-size', type=int, default=1000)
    ap.add_argument('--chunk-overlap', type=int, default=150)
    ap.add_argument('--force', action='store_true', help='Rebuild even if directory exists')
    ap.add_argument('--local', action='store_true', help='Use local HuggingFace embedding model')
    ap.add_argument('--model', default='', help='Embedding model (OpenAI embedding or local HF)')
    return ap.parse_args(argv)

if __name__ == '__main__':
    args = parse(sys.argv[1:])
    sys.exit(build(args))