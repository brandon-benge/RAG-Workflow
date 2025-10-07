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
	chunk_size: 40
	chunk_overlap: 8
	model: sentence-transformers/all-MiniLM-L6-v2
	bundle_url: https://github.com/brandon-benge/InterviewPrep/releases/download/latest/pdfs-bundle.tar.gz
	local: true   # exactly one of local/openai must be true
	openai: false
	force: true
```

## Quiz generation (moved)

Use `Quiz-Project` in this repo's root for quiz workflows.

```bash
cd Quiz-Project
./scripts/bin/run_venv.sh
./master.py prepare  # uses ../.chroma by default
```

If you move `Quiz-Project`, update its `quiz.params` (rag_persist) to point to this repo's `./.chroma` directory.

