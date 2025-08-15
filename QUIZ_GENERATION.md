# Quiz Generation Workflow

## Flow Diagram & Summary

This workflow guides you through setting up your local AI quiz system, generating quizzes, and validating answers. The process is fully offline-first, using Ollama and a local vector database (Chroma) built from PDFs or other supported sources. OpenAI is an optional provider. All quiz context comes from the vector database; markdown files are no longer used for quiz generation.

```mermaid
flowchart TD
  %% ====== SETUP ======
  subgraph S[Setup]
    S1[Install Ollama\n(brew install ollama)] --> S2[Pull Model\n(ollama pull mistral)]
    S3[Check/Start Ollama\n(check_ollama.py)] --> S4[Verify Params\n(quiz.params)]
  end

  %% ====== BUILD (VECTOR STORE) ======
  subgraph B[Build Vector Store (uses [build])]
    B0{"[build] enabled?"}
    B0 -->|yes| B1[Download PDF Bundle\n(--bundle-url)]
    B0 -->|no| P0
    B1 --> B2[Extract PDFs]
    B2 --> B3[Derive H1\n(first real line or title)]
    B3 --> B4[Split to Chunks\n(chunk_size / chunk_overlap)]
    B4 --> B5{"Embeddings Mode"}
    B5 -->|local=true| B6[Local Embeddings\n(HF model)]
    B5 -->|openai=true| B7[OpenAI Embeddings\n(OPENAI_API_KEY)]
    B6 --> B8[Persist to Chroma\n(persist dir)]
    B7 --> B8
  end

  %% ====== PREPARE (QUIZ GENERATION) ======
  subgraph P[Prepare Quiz (uses [prepare] & [prepare.rag])]
    P0[Load Params\n(provider/model, rag_k, files)]
    P1{"Provider?"}
    P1 -->|ollama| P2[Connect Ollama]
    P1 -->|ai| P3[Use OpenAI API]
    P2 --> P4
    P3 --> P4
    P4[For each Question (count)] --> P5{"Select Tag"}
    P5 -->|include_tags set| P6[Use provided tag(s)]
    P5 -->|else| P7[Choose random tag per question]
    P6 --> P8
    P7 --> P8
    P8[Retrieve top-k Chunks\n(from Chroma by tag/H1)] --> P9[Compose Prompt]
    P9 --> P10[LLM Generate Q/A/Expl]
    P10 --> P11[Validate Structure\n(count, options, letter)]
    P11 --> P12[Accumulate]
  end

  %% ====== OUTPUTS & VALIDATION ======
  subgraph O[Outputs & Validation]
    O1[Write quiz.json & answer_key.json]
    O2{"Validation Path"}
    O2 -->|Answer Key| O3[./master.py validate]
    O2 -->|Manual Marking| O4[./master.py export] --> O5[Edit quiz.txt] --> O6[./master.py parse] --> O3
  end

  %% ====== WIRES ======
  S --> B0
  B8 --> P0
  P12 --> O1 --> O2
```