# RAG-Workflow (Build-Only)

This repository builds and maintains a local Chroma vector store from a PDF bundle. It downloads the bundle, extracts and preprocesses documents, splits into chunks, embeds with either local or OpenAI embeddings, and persists to `.chroma/`.

Quiz generation has been moved to a separate project. Use the sibling `Quiz-Project` to generate quizzes using this vector store.

## What this repo does

- Download and extract a PDF bundle
- Split text into chunks and attach metadata
- Compute embeddings (local HF or OpenAI)
- Persist embeddings to Chroma at `./.chroma`

## Quick start

1) Ensure Ollama is installed and running if you need the local checks

2) Build the vector store

```bash
./scripts/bin/run_venv.sh ./master.py build
```

Configuration is read from `params.yaml` under the top-level `build` section. Dependencies are managed from the root `requirements.txt`.

### params.yaml (build section)

```yaml
build:
  enabled: true
  persist: .chroma

  # How to split documents before embedding:
  # sentence  - best general-purpose; coherent chunks for QA and retrieval
  # word      - very fine-grained windows; entities/short phrases; larger N, lower coherence
  # line      - structured text where newlines matter (code, logs, tables, markdown lists)
  # passage   - paragraph/section level; preserves local context; configurable size/overlap
  # page      - page-per-chunk (slides/forms); forces size/overlap to 1/0
  # document  - whole file; only when docs are tiny or indexed elsewhere; forces 1/0
  split_by: sentence

  # chunk_size/chunk_overlap are optional; auto-tuned per split_by if omitted.
  # Typical defaults -> word:(200,40), sentence:(6,1), line:(50,10), page/document:(1,0)
  # chunk_size: 6
  # chunk_overlap: 1

  # Embedding model (see guidance below). Changing models requires a rebuild (set force: true).
  model: sentence-transformers/all-MiniLM-L6-v2

  # Bundle to index
  bundle_url: https://github.com/brandon-benge/InterviewPrep/releases/download/latest/pdfs-bundle.tar.gz

  # Provider: exactly one must be true
  local: true
  openai: false

  # Rebuild from scratch (wipe .chroma) — recommended when you change model/splitting
  force: true
```

Split strategies and when to use them:
- sentence: best general-purpose choice; coherent chunks for QA and retrieval
- word: sliding windows for very fine-grained lookups (entities, short phrases)
- line: structured text where newlines are meaningful (code, logs, tables, bullet-heavy markdown)
- passage: paragraph/section level; configurable; good for preserving local context
- page: page-per-chunk; great for slide decks or forms; forces size/overlap to 1/0
- document: whole file; only when docs are tiny or indexing externally; forces size/overlap to 1/0

The builder reports the average tokens per chunk when possible (falls back to characters if tokenization isn’t available).

## Choosing an embedding model (and when to use)

You do not need to change the model when you change `split_by`, but it can help to align model capacity and domain with chunk length and content.

- Fast local baseline (short/medium chunks, general text)
  - sentence-transformers/all-MiniLM-L6-v2 (384 dims): very fast, compact index; great default for `sentence` or `line`.
- Higher quality local (better semantic recall, larger index)
  - sentence-transformers/all-mpnet-base-v2 (768 dims)
  - BAAI/bge-base-en-v1.5 (768 dims) or BAAI/bge-large-en-v1.5 (1024 dims)
- Code/diagrams/logs (line-split, fenced blocks, mermaid, code)
  - jina-embeddings-v2-base-code (code-oriented; stronger on structured/text-with-code)
- Cloud (very long chunks, mixed content, strong quality)
  - OpenAI text-embedding-3-small (1536 dims): lower cost, good quality
  - OpenAI text-embedding-3-large (3072 dims): highest quality, larger vectors and cost

Guidelines by split strategy:
- sentence: MiniLM is a great default; use MPNet/BGE for higher quality if index size is acceptable.
- line: for natural prose, same as sentence; for code/log-style data (including mermaid), prefer a code-focused model.
- passage: favor MPNet/BGE; keep chunks under ~512 tokens for most local models or consider OpenAI for longer paragraphs.
- page/document: local models will truncate if content is long; prefer OpenAI embeddings or reduce chunk size.

Other considerations:
- Vector size vs performance: larger dimensions improve recall but increase storage and query latency.
- Token limits and truncation: many local models effectively cap around ~512 tokens; OpenAI supports much longer inputs.
- Multilingual: consider multilingual-capable models (e.g., bge-m3, paraphrase-multilingual-mpnet-base-v2) if needed.
- Rebuild required on change: switching models or splitting strategies should be followed by a fresh build (`force: true`).

Example tweaks:
```yaml
# sentence-level, fast and compact:
model: sentence-transformers/all-MiniLM-L6-v2
split_by: sentence
# passage-level, higher quality:
# model: sentence-transformers/all-mpnet-base-v2
# split_by: passage
# page-level with long inputs (cloud):
# model: text-embedding-3-large
# local: false
# openai: true
```
