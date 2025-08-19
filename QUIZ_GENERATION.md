# Quiz Generation Workflow

## Flow Diagram & Summary

This workflow guides you through setting up your local AI quiz system, generating quizzes, and validating answers. The process is fully offline-first, using Ollama and a local vector database (Chroma) built from PDFs or other supported sources. OpenAI is an optional provider. All quiz context comes from the vector database; markdown files are no longer used for quiz generation.

```mermaid
flowchart TD
  %% ====== SETUP ======
  subgraph setup["Setup"]
    s1["Install Ollama (brew install ollama)"] --> s2["Pull Model (ollama pull mistral)"]
    s3["Check/Start Ollama (check_ollama.py)"] --> s4["Verify Params (quiz.params)"]
  end

  %% ====== BUILD (VECTOR STORE) ======
  subgraph build["Build Vector Store uses build"]
    b0{"build enabled?"}
    b0 -->|yes| b1["Download PDF Bundle (--bundle-url)"]
    b0 -->|no| p0
    b1 --> b2["Extract PDFs"]
    b2 --> b2a["Extract Text from PDFs"]
    b2a --> b3["Derive H1 (first meaningful line or title)"]
    b3 --> b2b["Generate Tags via extract_keywords_tfidf"]
    b2b --> b2c{"Tags present for all docs?"}
    b2c -->|no| b2e["Fail build (missing tags)"]
    b2c -->|yes| b4["Split to Chunks (chunk_size / chunk_overlap)"]
    b4 --> b4a["Attach metadata (h1, tags)"]
    b4a --> b5{"Embeddings Mode"}
    b5 -->|local=true| b6["Local Embeddings (HF model)"]
    b5 -->|openai=true| b7["OpenAI Embeddings (OPENAI_API_KEY)"]
    b6 --> b7a["Insert into metastore (Chroma)"]
    b7 --> b7a
    b7a --> b8["Persist to Chroma (persist dir)"]
  end

  %% ====== PREPARE (QUIZ GENERATION) ======
  subgraph prepare["Prepare Quiz uses prepare and prepare.rag"]
    p0["Load Params (provider/model, rag_k, files)"]
    p1{"Provider?"}
    p1 -->|ollama| p2["Connect Ollama"]
    p1 -->|ai| p3["Use OpenAI API"]
    p2 --> p4
    p3 --> p4
    p4["For each Question (count)"] --> p5{"Select Tag"}
    p5 -->|include_tags set| p6["Use provided tag(s)"]
    p5 -->|else| p7["Choose random tag per question"]
    p6 --> p8
    p7 --> p8
    p8["Retrieve top-k Chunks from Chroma by tag or H1"] --> p9["Compose Prompt"]
    p9 --> p10["LLM Generate Q/A/Expl"]
    p10 --> p11["Validate Structure (count, options, letter)"]
    p11 --> p12["Accumulate"]
  end

  %% ====== OUTPUTS & VALIDATION ======
  subgraph outputs["Outputs & Validation"]
    o1["Write quiz.json and answer_key.json"]
    o2{"Validation Path"}
    o2 -->|Answer Key| o3["./master.py validate"]
    o2 -->|Manual Marking| o4["./master.py export"] --> o5["Edit quiz.txt"] --> o6["./master.py parse"] --> o3
  end

  %% ====== WIRES ======
  setup --> b0
  b8 --> p0
  p12 --> o1 --> o2
```

---

## 1. Install Ollama & Build the Vector Store

The first step in the workflow is to **build the vector store** from your PDF bundle. The `vector_store_build.py` script downloads the latest PDF bundle from the GitHub releases page, extracts the PDF files, derives headings from the first meaningful line of each document, and stores the content in a Chroma vector store for retrieval. This step is only required when you first set up the system or when new content is added. **Quiz generation does not occur in this step.** All settings for building are read from the `[build]` section of `quiz.params`.

**Ollama Installation**
```bash
brew install ollama
ollama pull mistral
```

**Vector Store Build (Local Embeddings)**
```bash
./master.py build
```
This will build the vector store using the settings under the `[build]` section (e.g., embedding model, chunk size, persist directory). The vector store is written to the directory specified by `persist` (e.g., `.chroma`). Rebuilding (e.g., force, embedding mode, chunk sizes) is controlled by the `[build]` section.

**Vector Store Build (OpenAI Embeddings)**

To use OpenAI embeddings, set `openai=true` (and `local=false`) in the `[build]` section of `quiz.params`, export `OPENAI_API_KEY`, then run:
```bash
./master.py build
```
All embedding and build options for the vector store come from the `[build]` section of `quiz.params`.

---

## 2. Generate the Quiz

After the vector store has been built (using `./master.py build`), generate quiz questions using:

```bash
./master.py prepare
```

**This step is responsible for quiz generation only, using the vector store built in the previous step.** All quiz generation settings are read from the `[prepare]` section of `quiz.params`. The master script does not accept CLI flags for quiz generation.

**Primary (Offline, Ollama)**
```bash
./master.py prepare
```
This generates quiz questions using the vector store and the provider/model specified in `[prepare]` (e.g., `provider=ollama`, `model=mistral`). Question count, output file names, retrieval parameters, and freshness are all defined in `[prepare]`.

**Experimental (OpenAI)**

Set `provider=ai` and `model=<openai-model>` in `[prepare]` of `quiz.params`, export `OPENAI_API_KEY`, then run:
```bash
./master.py prepare
```

**Advanced Options**

- Change model: set `provider` and `model` in `[prepare]` (e.g., `provider=ollama`, `model=mistral`)
- Control novelty / avoid duplicates: set `avoid_recent_window=<N>` in `[prepare]` (required; N is the number of previous questions to avoid repeating)
- Retrieval depth: set `rag_k` in `[prepare]`
- Filtering: set `include_tags`, `include_h1`, and `restrict_sources` in `[prepare.rag]`

---

## 3. Validate the Quiz

After generating the quiz, you can validate the answer key or check your own answers.

**Interactive Validation with Answer Key**
```bash
./master.py validate
```
This will prompt you for each question and provide immediate feedback.

**Validate with User Answers**
```bash
./master.py validate
```

**Manual Marking Workflow**
1. Export plain text: `./master.py export`
2. Mark answers in `quiz.txt`
3. Parse answers: `./master.py parse`
4. Validate as above.

---

## Files Produced

- `quiz.json` – list of question objects (no answers)
- `answer_key.json` – mapping question id -> `{ answer, explanation }`
- `quiz.txt` – markable plain text export (created by 
`./master.py export` )

## Providers & Status

| Provider | Config Source | Status | Notes |
|----------|----------------|--------|-------|
| OpenAI   | `quiz.params`  | Experimental | Requires `OPENAI_API_KEY`; configured via `[prepare]` |
| Ollama   | `quiz.params`  | Primary      | Offline / fast iteration / zero API cost; configured via `[prepare]` |

> **Accuracy Note (Ollama):** Local models may occasionally produce mismatches (e.g. the answer letter not conceptually matching the best option, weak explanations, or subtly duplicated stems). Validate logically.  If a question looks off: (1) re‑run with `fresh=true` in params; (2) adjust retrieval filters; or (3) manually correct.  The validator checks structure, not semantic truth.  OpenAI path can yield different style but is optional/experimental.
> **Accuracy Note (Ollama):** Local models may occasionally produce mismatches (e.g. the answer letter not conceptually matching the best option, weak explanations, or subtly duplicated stems). Validate logically.  If a question looks off: (1) re‑run with a larger `avoid_recent_window` in params; (2) adjust retrieval filters; or (3) manually correct.  The validator checks structure, not semantic truth.  OpenAI path can yield different style but is optional/experimental.

## Offline‑First Philosophy

The system prioritizes *repeatable, air‑gapped study*.  Core guarantees:

1. Works with a **local PDF bundle** converted into a Chroma vector store plus Ollama and HuggingFace embeddings.
2. Never requires an internet call unless you opt into OpenAI.
3. Vector store auto‑build (`[build] enabled=true`) keeps friction low.
4. All quiz context comes from the vector database; template mode is deprecated.

## Ollama Setup & Validation

> Install: https://ollama.com
```bash
ollama pull mistral
./scripts/bin/run_venv.sh scripts/rag/check_ollama.py check
```
> Manage via tasks (Install / Start / Stop / Check) or CLI (`brew services start ollama`).


---

## Retrieval‑Augmented Generation (Always On)

The quiz pipeline now automatically retrieves context from a Chroma vector store before LLM prompting. All quiz context comes from the vector database. Markdown files are not used for quiz generation.

### Build vs. Prepare: Responsibilities

- The **build** step is for constructing the vector store, using all settings from the `[build]` section of `quiz.params`. This includes downloading PDFs, extracting, chunking, and embedding.
- The **prepare** step is for generating the quiz, using all settings from the `[prepare]` section of `quiz.params`. This step retrieves context from the vector store and creates questions.

### Preflight Validation (`master.py prepare`)

When you run `./master.py prepare`, the script will:
1. Build the vector store **only if** `[build] enabled=true` in `quiz.params`. (This is a convenience pre-check to ensure the vector store exists and is up to date.)
2. Validate provider connectivity as specified in `[prepare]` (OpenAI key or Ollama daemon).
3. Generate the quiz using retrieval and quiz parameters from `[prepare]` and `[prepare.rag]`.

### Vector Store Build (Manual / CI)

Prefer `./master.py prepare` for an automated workflow that will build the vector store if needed and then generate the quiz. For CI or manual control of only the build step, call the builder directly (all flags required):

```bash
./scripts/bin/run_venv.sh scripts/rag/vector_store_build.py \
  --persist ./.chroma \
  --chunk-size 800 \
  --chunk-overlap 120 \
  --local \  # or --openai
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --bundle-url https://github.com/brandon-benge/InterviewPrep/releases/download/latest/pdfs-bundle.tar.gz \
  --force
```

### quiz.params (minimal skeleton)

```ini
[build]
enabled=true               # Controls whether the vector store is (re)built before quiz generation
persist=.chroma            # Directory for the vector store
chunk_size=800
chunk_overlap=120
model=sentence-transformers/all-MiniLM-L6-v2
bundle_url=https://github.com/brandon-benge/InterviewPrep/releases/download/latest/pdfs-bundle.tar.gz
local=true
openai=false
force=true

[prepare]
provider=ollama            # Controls which LLM provider is used for quiz generation
model=mistral
count=5
quiz=quiz.json
answers=answer_key.json
rag_persist=.chroma        # Path to the vector store to use for retrieval
rag_k=3
avoid_recent_window=5
rag_local=true
rag_openai=false

[prepare.rag]
include_tags=team prioritization
include_h1=
restrict_sources=
```

### Source & Tag / H1 Filtering

> Set filters in `[prepare.rag]` inside `quiz.params` (e.g., `include_tags=caching consistency`, `include_h1=rate-limiting`, `restrict_sources=pdfs`). Then run `./master.py prepare`. These values affect only quiz generation and retrieval, not vector store building. No CLI flags are accepted by `master.py`.

### Recommended Settings

- Keep `rag_k` between 3–5.
- Rebuild the store (`[build] enabled=true` or manual build) after adding new PDFs to the release bundle.
- Prefer local embeddings for consistent offline workflow (`local=true`).
- Use OpenAI embeddings only if you explicitly need broader semantic recall (experimental path).

### Disabling RAG (Debug Only)

If you need to benchmark raw behavior or inspect model output without context, call `generate_quiz.py` directly with `--no-rag` (the master script always uses RAG and expects a vector store to exist).

### Maintenance Tips

- Delete & rebuild (`force=true`) after large document reorganizations or when a new PDF bundle is released.
- Monitor store size; extremely large stores may slow retrieval—consider pruning outdated notes.

## Troubleshooting

| Symptom | Cause | Resolution |
|---------|-------|------------|
| Vector store missing | Not built yet | Add `[build] enabled=true` or run build script manually |
| Vector store health check failed | Embedding mismatch / corrupt dir | Rebuild with matching flags (`local=true` vs OpenAI) |
| Ollama request error | Daemon not running | `./scripts/bin/run_venv.sh scripts/rag/check_ollama.py start` or launch app |
| 0 questions returned | Model output malformed | Re-run; ensure model supports instruction following |
| Validation failed: count mismatch | Model produced fewer items | Re-run; sometimes temperature / truncation issues |
| OPENAI_API_KEY error | Env var missing | Only needed for experimental OpenAI path; `export OPENAI_API_KEY=sk-...` |
| Question seems wrong / answer dubious (Ollama) | Model hallucination | Re-run with a larger `avoid_recent_window` in params; switch provider; manual edit |

## Example Snippet

```json
[
  {
    "id": "Q1",
    "question": "Which component handles rate limiting?",
    "options": ["API Gateway", "Object Store", "CDN Edge", "Log Indexer"],
    "topic": "Rate Limiter System Design",
    "difficulty": "medium"
  }
]
```

#### *Placeholder Heading (Fill Me)*
> Return to main README: [README.md](./README.md)