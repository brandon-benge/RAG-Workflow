#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime as _dt
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =========================
# Logging & Print Wrapper
# =========================

_orig_print = print
def log(level: str, msg: str) -> None:
    ts = _dt.now().isoformat(timespec='seconds')
    _orig_print(f"[{ts}] [{level}] {msg}")

def print(*args, **kwargs):  # type: ignore
    if args and isinstance(args[0], str):
        ts = _dt.now().isoformat(timespec='seconds')
        args = (f"[{ts}] {args[0]}",) + args[1:]
    return _orig_print(*args, **kwargs)

# =========================
# Data Model
# =========================

@dataclass
class Question:
    id: str
    question: str
    options: List[str]
    topic: str
    difficulty: str
    answer: str
    explanation: str
    raw_response: Optional[Any] = None  # dict or string

    def public_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "options": self.options,
            "topic": self.topic,
            "difficulty": self.difficulty,
        }

    def answer_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "explanation": self.explanation,
        }

# =========================
# Constants & CLI
# =========================


HISTORY_FILE = Path(".quiz_history.json")

@dataclass
class Config:
    count: int
    quiz: Path
    answers: Path
    sources: List[str]
    no_random_component: bool
    model: str
    ai: bool
    ollama: bool
    ollama_model: str
    ollama_temperature: Optional[float]
    ollama_num_predict: Optional[int]
    ollama_top_k: Optional[int]
    ollama_top_p: Optional[float]
    ollama_snippet_chars: int
    ollama_corpus_chars: int
    ollama_compact_json: bool
    debug_ollama_payload: bool
    dump_ollama_prompt: Optional[str]
    dump_ollama_payload: Optional[str]
    dump_llm_payload: Optional[str]
    template: bool
    seed: int
    fresh: bool
    verify: bool
    dry_run: bool
    # RAG
    rag_persist: str
    rag_k: int
    rag_queries: Optional[List[str]]
    rag_max_queries: Optional[int]
    rag_local: bool
    rag_embed_model: str
    no_rag: bool
    restrict_sources: Optional[List[str]]
    include_tags: Optional[List[str]]
    include_h1: Optional[List[str]]
    dump_rag_context: Optional[str]

def parse_args(argv: List[str]) -> Config:
    p = argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=5)
    p.add_argument('--quiz', default='quiz.json')
    p.add_argument('--answers', default='answer_key.json')
    # Removed --sources argument; all context comes from the vector database
    p.add_argument('--no-random-component', action='store_true')

    # Providers
    p.add_argument('--model', default='gpt-4o-mini')
    p.add_argument('--ai', action='store_true')
    p.add_argument('--ollama', action='store_true')
    p.add_argument('--ollama-model', default='mistral')

    # Ollama tuning
    p.add_argument('--ollama-temperature', type=float)
    p.add_argument('--ollama-num-predict', type=int)
    p.add_argument('--ollama-top-k', type=int)
    p.add_argument('--ollama-top-p', type=float)
    p.add_argument('--ollama-snippet-chars', type=int, default=-1)
    p.add_argument('--ollama-corpus-chars', type=int, default=-1)
    p.add_argument('--ollama-compact-json', action='store_true')
    p.add_argument('--debug-ollama-payload', action='store_true')
    p.add_argument('--dump-ollama-prompt')
    p.add_argument('--dump-ollama-payload')
    p.add_argument('--dump-llm-payload')

    # Flow
    p.add_argument('--template', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--fresh', action='store_true')
    p.add_argument('--verify', action='store_true')
    p.add_argument('--dry-run', action='store_true')

    # RAG
    p.add_argument('--rag-persist', default='.chroma')
    p.add_argument('--rag-k', type=int, default=4)
    p.add_argument('--rag-queries', nargs='+')
    p.add_argument('--rag-max-queries', type=int)
    p.add_argument('--rag-local', action='store_true')
    p.add_argument('--rag-embed-model', default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--no-rag', dest='no_rag', action='store_true')

    # RAG filters/obs
    p.add_argument('--restrict-sources', nargs='+')
    p.add_argument('--include-tags', nargs='+')
    p.add_argument('--include-h1', nargs='+')
    p.add_argument('--dump-rag-context')

    a = p.parse_args(argv)
    return Config(
        count=a.count,
        quiz=Path(a.quiz),
        answers=Path(a.answers),
        sources=a.sources,
        no_random_component=a.no_random_component,
        model=a.model,
        ai=a.ai,
        ollama=a.ollama,
        ollama_model=a.ollama_model,
        ollama_temperature=a.ollama_temperature,
        ollama_num_predict=a.ollama_num_predict,
        ollama_top_k=a.ollama_top_k,
        ollama_top_p=a.ollama_top_p,
        ollama_snippet_chars=a.ollama_snippet_chars,
        ollama_corpus_chars=a.ollama_corpus_chars,
        ollama_compact_json=a.ollama_compact_json,
        debug_ollama_payload=a.debug_ollama_payload,
        dump_ollama_prompt=a.dump_ollama_prompt,
        dump_ollama_payload=a.dump_ollama_payload,
        dump_llm_payload=a.dump_llm_payload,
        template=a.template,
        seed=a.seed,
        fresh=a.fresh,
        verify=a.verify,
        dry_run=a.dry_run,
        rag_persist=a.rag_persist,
        rag_k=a.rag_k,
        rag_queries=a.rag_queries,
        rag_max_queries=a.rag_max_queries,
        rag_local=a.rag_local,
        rag_embed_model=a.rag_embed_model,
        no_rag=a.no_rag,
        restrict_sources=a.restrict_sources,
        include_tags=a.include_tags,
        include_h1=a.include_h1,
        dump_rag_context=a.dump_rag_context,
    )

# =========================
# File IO Helpers
# =========================



def _pretty_and_parse_raw_response(key: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Normalize raw_response for readability:
      - If raw_response is a JSON string, parse it to dict.
      - If raw_response['response'] is a JSON string, parse it too.
      - Preserve all keys (including 'context').
    """
    for _, entry in key.items():
        raw = entry.get('raw_response', '')
        obj: Any = raw

        # Top-level: parse '{...}' string into dict if possible
        if isinstance(raw, str) and raw.strip().startswith('{'):
            try:
                obj = json.loads(raw)
            except Exception:
                obj = raw  # leave as-is

        # If dict, see if 'response' looks like JSON string
        if isinstance(obj, dict):
            # Remove 'context' key if present
            obj.pop('context', None)
            resp = obj.get('response')
            if isinstance(resp, str):
                text = resp.strip()
                if text.startswith('[') or text.startswith('{'):
                    try:
                        obj['response'] = json.loads(text)
                    except Exception:
                        obj['response'] = text.replace('\\n', '\n')
                else:
                    obj['response'] = text.replace('\\n', '\n')
        entry['raw_response'] = obj

    return key

def write_outputs(questions: List[Question], quiz_path: Path, answers_path: Path) -> None:
    # quiz.json
    quiz = [q.public_dict() for q in questions]
    quiz_path.write_text(json.dumps(quiz, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    # answer_key.json initial pass (store raw_response as object when possible)
    key: Dict[str, Dict[str, Any]] = {}
    for q in questions:
        entry = q.answer_dict()
        raw = q.raw_response

        if isinstance(raw, str) and raw.strip().startswith('{'):
            try:
                entry['raw_response'] = json.loads(raw)
            except Exception:
                entry['raw_response'] = raw
        else:
            entry['raw_response'] = raw

        key[q.id] = entry

    # Normalize & pretty-print
    key = _pretty_and_parse_raw_response(key)
    answers_path.write_text(json.dumps(key, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

# =========================
# Question Builders
# =========================

def template_questions(files: Dict[str, str], count: int, seed: int) -> List[Question]:
    pairs: List[Tuple[str, str, str]] = []
    seen = set()

    for path, text in files.items():
        topic = Path(path).stem.replace('-', ' ').title()

        for line in text.splitlines():
            line = line.strip().lstrip('-').lstrip('*').strip()
            if not line:
                continue
            if ':' in line and not line.lower().startswith('http'):
                name, desc = line.split(':', 1)
                name, desc = name.strip(), desc.strip()
                if 2 <= len(name) <= 60 and 5 <= len(desc) <= 200:
                    key = name.lower()
                    if key not in seen:
                        seen.add(key)
                        pairs.append((name, desc, topic))

        for line in text.splitlines():
            if line.startswith('##'):
                name = line.lstrip('#').strip()
                if 2 <= len(name) <= 60:
                    key = name.lower()
                    if key not in seen:
                        seen.add(key)
                        pairs.append((name, f"Concept related to {name}", topic))

    if not pairs:
        raise RuntimeError('No template candidates found.')

    rng = random.Random(seed)
    names = [p[0] for p in pairs]
    questions: List[Question] = []

    for idx in range(count):
        name, desc, topic = pairs[idx % len(pairs)]
        distractors = [n for n in names if n != name]
        while len(distractors) < 3:
            distractors.append(f'Placeholder {len(distractors)+1}')
        rng.shuffle(distractors)
        options = [name] + distractors[:3]
        rng.shuffle(options)
        answer_letter = chr(ord('A') + options.index(name))

        questions.append(Question(
            id=f"Q{idx+1}",
            question=f"Which component is described: '{desc[:140]}'?",
            options=options,
            topic=topic,
            difficulty=('easy' if len(desc) < 60 else 'medium'),
            answer=answer_letter,
            explanation=f"{name} matches the description."
        ))

    return questions

# =========================
# Robust JSON-from-LLM
# =========================

def _parse_model_questions(raw_json: str, provider: str) -> List[Question]:
    """Robustly parse LLM JSON for several shapes."""
    def _to_list(data: Any) -> List[Dict[str, Any]]:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if isinstance(data.get('questions'), list):
                return data['questions']
            if isinstance(data.get('data'), list):
                return data['data']
            if all(isinstance(v, dict) for v in data.values()):
                items = []
                for k, v in data.items():
                    vv = dict(v)
                    vv.setdefault('id', k)
                    items.append(vv)
                return items
            return [data]
        raise RuntimeError(f'{provider}: unexpected JSON shape')

    try:
        data = json.loads(raw_json)
    except Exception as e:
        m = re.search(r'(\[\s*{.*}\s*\])', raw_json, re.DOTALL)
        if not m:
            raise RuntimeError(f'{provider}: could not parse JSON: {e}')
        data = json.loads(m.group(1))

    items = _to_list(data)
    out: List[Question] = []

    for idx, obj in enumerate(items, start=1):
        if not isinstance(obj, dict):
            continue

        qid = str(obj.get('id') or f'Q{idx}')
        question = (obj.get('question') or '').strip() or f'Placeholder question {idx}'

        raw_opts = obj.get('options')
        if not isinstance(raw_opts, list):
            cand = []
            for key in ['a', 'b', 'c', 'd', 'A', 'B', 'C', 'D']:
                if key in obj:
                    cand.append(str(obj[key]))
            if not cand:
                cand = ['Option A', 'Option B', 'Option C', 'Option D']
            raw_opts = cand
        options = [str(o).strip() for o in raw_opts][:4]
        while len(options) < 4:
            options.append(f'Extra {len(options)+1}')

        # Prefer an explicit answer_letter if provided; otherwise map the answer.
        answer_letter: Optional[str] = None
        explicit_letter = str(obj.get('answer_letter', '')).strip().upper()
        if explicit_letter in ['A', 'B', 'C', 'D']:
            answer_letter = explicit_letter
        else:
            answer_raw = str(obj.get('answer', 'A')).strip()
            if answer_raw.upper() in ['A', 'B', 'C', 'D']:
                answer_letter = answer_raw.upper()
            else:
                # exact match
                for i, opt in enumerate(options):
                    if answer_raw.lower() == opt.lower():
                        answer_letter = chr(ord('A') + i)
                        break
                # prefix / loose match
                if answer_letter is None:
                    for i, opt in enumerate(options):
                        if opt.lower().startswith(answer_raw.lower()[:5]):
                            answer_letter = chr(ord('A') + i)
                            break

                # If still no match, replace a random option with the model's answer
                if answer_letter is None and answer_raw:
                    import random as _rand
                    idx_replace = _rand.randint(0, 3)
                    options[idx_replace] = answer_raw
                    answer_letter = chr(ord('A') + idx_replace)
                    # annotate explanation to make this transparent
                    note = (
                        f" [Note: original answer '{answer_raw}' was not in options; "
                        f"replaced option {answer_letter} accordingly]"
                    )
                    obj_expl = (obj.get('explanation') or '').strip()
                    obj['explanation'] = (obj_expl + note) if obj_expl else note

        topic = (obj.get('topic') or 'General').strip() or 'General'
        difficulty = (obj.get('difficulty') or 'medium').strip() or 'medium'
        explanation = (obj.get('explanation') or '').strip()

        out.append(Question(
            id=qid,
            question=question,
            options=options,
            topic=topic,
            difficulty=difficulty,
            answer=answer_letter,
            explanation=explanation
        ))

    if not out:
        raise RuntimeError(f'{provider}: no questions parsed from model output')
    return out

# =========================
# Providers
# =========================

class Providers:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._openai = None
        self._OpenAIClient = None
        self._openai_client = None
        self._requests = None

        try:
            import requests  # type: ignore
            self._requests = requests
        except Exception:
            pass

        try:
            import openai  # type: ignore
            self._openai = openai
            try:
                from openai import OpenAI  # type: ignore
                self._OpenAIClient = OpenAI
                try:
                    self._openai_client = OpenAI()
                except Exception:
                    self._openai_client = None
            except Exception:
                self._OpenAIClient = None
        except Exception:
            pass

    def _dump_payload(self, prompt: str, payload: dict) -> None:
        if self.cfg.dump_ollama_prompt:
            try:
                Path(self.cfg.dump_ollama_prompt).write_text(prompt, encoding='utf-8')
                log("debug", f"Wrote full LLM prompt -> {self.cfg.dump_ollama_prompt}")
            except Exception as e:
                log("warn", f"Could not write prompt: {e}")
        path = self.cfg.dump_llm_payload or self.cfg.dump_ollama_payload
        if path:
            try:
                Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
                log("debug", f"Wrote LLM payload -> {path}")
            except Exception as e:
                log("warn", f"Could not write payload: {e}")

    # --- OpenAI ---

    def openai_questions(self, files: Dict[str, str], count: int, model: str,
                         token: str, recent_norm: List[str], temperature: float, *, iteration: Optional[int] = None) -> List[Question]:
        if not self._openai:
            raise RuntimeError('openai package not available')
        if not os.getenv('OPENAI_API_KEY') and not self._openai_client:
            raise RuntimeError('OPENAI_API_KEY not set')

        # Build corpus (cap to avoid huge prompts)
        max_chars = 28000
        parts, total = [], 0
        for pth, txt in files.items():
            trimmed = txt[:2000]
            part = f"\n# FILE: {pth}\n{trimmed}\n"
            if total + len(part) > max_chars:
                continue
            parts.append(part)
            total += len(part)
        corpus = ''.join(parts)

        recent_clause = ("Avoid reusing these prior question phrasings: " + '; '.join(recent_norm[:30])) if recent_norm else ''
        system = (
            "You are an assistant that creates high-quality multiple-choice quiz questions for system design and devops. "
            "If citation lines like 'C<number>:' exist, you MUST base questions on them, pick a concise Title Case topic, "
            "and return STRICT JSON with: id, question, options(4), topic, difficulty, answer, explanation. "
            "The 'answer' MUST be a single letter A/B/C/D corresponding to the provided options; you may also include 'answer_letter' as the same letter."
        )
        user = (f"Uniqueness token: {token}. Generate {count} questions (IDs Q1..Q{count}). "
                f"{recent_clause} Source material: {corpus[:12000]}")

        start = time.time()
        if self._openai_client:
            resp = self._openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=temperature
            )
            content = resp.choices[0].message.content or ""
            raw_response: Any = json.loads(resp.model_dump_json())
        else:
            self._openai.api_key = os.getenv('OPENAI_API_KEY')
            resp = self._openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=temperature
            )
            content = resp['choices'][0]['message']['content'] or ""
            raw_response = json.loads(json.dumps(resp, default=str))
        duration = time.time() - start
        iter_str = f"[{(iteration or 0)+1}/{self.cfg.count}] " if iteration is not None else ""
        log("info", f"{iter_str}LLM response time (openai {model}): {duration:.2f}s")

        m = re.search(r"```json\n(.*)```", content, re.DOTALL)
        json_text = m.group(1) if m else content
        questions = _parse_model_questions(json_text, provider='openai')
        for q in questions:
            q.raw_response = raw_response
        return questions

    # --- Ollama ---

    def ollama_questions(self, files: Dict[str, str], count: int, model: str,
                         token: str, recent_norm: List[str], temperature: float,
                         *, snippet_chars: int, corpus_chars: int, num_predict: Optional[int],
                         top_k: Optional[int], top_p: Optional[float], compact_json: bool,
                         debug_payload: bool, iteration: Optional[int] = None, theme: Optional[str] = None) -> List[Question]:
        if not self._requests:
            raise RuntimeError('requests not installed')

        # Build corpus respecting limits
        parts, total = [], 0
        max_chars = None if corpus_chars == -1 else 28000
        for pth, txt in files.items():
            trimmed = txt if snippet_chars == -1 or pth == 'RAG_CONTEXT.md' else txt[:snippet_chars]
            part = f"\n# FILE: {pth}\n{trimmed}\n"
            if corpus_chars != -1 and total + len(part) > corpus_chars:
                break
            if max_chars is not None and total + len(part) > max_chars:
                break
            parts.append(part)
            total += len(part)
        corpus = ''.join(parts)

        recent_clause = ("Avoid reusing these prior question phrasings: " + '; '.join(recent_norm[:30])) if recent_norm else ''
        style_clause = 'Return STRICT COMPACT JSON array ONLY.' if compact_json else 'Return STRICT JSON as an array.'
        prompt = (
            f"Uniqueness token: {token}. Create {count} multiple choice questions (IDs Q1..Q{count}) "
            "about system design and devops using ONLY the provided notes.\n"
            f"{recent_clause}\n"
            "If a citation directory is present, ground each question in those sections, pick a concise Title Case topic, "
            "NEVER use 'RAG_CONTEXT' as a topic.\n"
            f"{style_clause} Keys: id, question, options, topic, difficulty, answer, explanation. "
            "The 'answer' MUST be a single letter A/B/C/D matching the options; you may also include 'answer_letter' as the same letter. Do NOT put option text in 'answer'.\n"
            "Source notes:\n" + (corpus if corpus_chars == -1 else corpus[:corpus_chars])
        )

        options: Dict[str, Any] = {'temperature': temperature}
        if num_predict is not None: options['num_predict'] = num_predict
        if top_k is not None: options['top_k'] = top_k
        if top_p is not None: options['top_p'] = top_p
        payload = {'model': model, 'prompt': prompt, 'stream': False, 'options': options}
        self._dump_payload(prompt, payload)

        if debug_payload:
            trunc = prompt[:600] + (f"... [truncated, total {len(prompt)} chars]" if len(prompt) > 600 else "")
            try:
                log("debug", "Ollama request payload (truncated prompt):")
                _orig_print(json.dumps({**payload, "prompt": trunc}, indent=2)[:4000])
            except Exception:
                pass

        start = time.time()
        resp = self._requests.post('http://localhost:11434/api/generate', json=payload, timeout=240)
        if resp.status_code != 200:
            raise RuntimeError(f'Ollama HTTP {resp.status_code}: {resp.text[:200]}')
        data = resp.json()
        content = data.get('response', '') or ''
        m = re.search(r"```json\n(.*)```", content, re.DOTALL)
        json_text = m.group(1) if m else content
        duration = time.time() - start
        iter_str = f"[{(iteration or 0)+1}/{self.cfg.count}] " if iteration is not None else ""
        theme_str = f", theme: {theme}" if theme else ""
        log("info", f"{iter_str}LLM response time (ollama {model}{theme_str}): {duration:.2f}s")

        questions = _parse_model_questions(json_text, provider='ollama')
        for q in questions:
            q.raw_response = data  # keep the entire dict (includes context if present)
        return questions

# =========================
# RAG
# =========================

class RAG:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._vs = None
        self._embedding = None
        self._init_done = False

    def _slugify(self, x: str) -> str:
        return re.sub(r'[^a-z0-9]+', '-', x.lower()).strip('-') if x else ''

    def _init(self) -> None:
        if self._init_done:
            return
        self._init_done = True
        try:
            try:
                from langchain_chroma import Chroma  # type: ignore
            except Exception:
                from langchain_community.vectorstores import Chroma  # type: ignore

            if self.cfg.rag_local:
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
                except Exception:
                    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
                self._embedding = HuggingFaceEmbeddings(model_name=self.cfg.rag_embed_model)
            else:
                from langchain_openai import OpenAIEmbeddings  # type: ignore
                self._embedding = OpenAIEmbeddings()

            if not os.path.isdir(self.cfg.rag_persist):
                log("warn", f"RAG store '{self.cfg.rag_persist}' not found; proceeding without RAG.")
                return

            self._vs = Chroma(persist_directory=self.cfg.rag_persist, embedding_function=self._embedding)
        except ModuleNotFoundError as e:
            log("warn", f"RAG modules missing ({e}); continuing without RAG.")
        except Exception as e:
            log("warn", f"RAG init error: {e}; continuing without RAG.")

    def _fetch_dynamic_h1_queries(self, db_path: str) -> List[str]:
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT string_value FROM embedding_metadata WHERE key='h1' GROUP BY string_value;")
            rows = cur.fetchall()
            conn.close()
            return [row[0][:80] for row in rows if row[0]]
        except Exception as e:
            log("warn", f"Could not fetch dynamic H1 queries: {e}")
            return []

    def build_context(self, files: Dict[str, str], *, k: int, count: int) -> Tuple[Dict[str, str], List[str]]:
        if self.cfg.no_rag:
            return files, []
        self._init()
        if not self._vs:
            return files, []

        db_path = os.path.join(self.cfg.rag_persist, 'chroma.sqlite3')
        dynamic_queries = self._fetch_dynamic_h1_queries(db_path)

        filter_override = any([self.cfg.restrict_sources, self.cfg.include_tags, self.cfg.include_h1])
        if dynamic_queries and not filter_override and not self.cfg.rag_queries:
            rng = random.Random(self.cfg.seed)
            selected_themes = rng.sample(dynamic_queries, min(count, len(dynamic_queries)))
            default_queries = selected_themes
            log("info", f"Using {len(selected_themes)} random dynamic H1 queries from SQLite.")
            log("info", "Theme mode enabled: using H1 headings as themes.")
        else:
            default_queries = dynamic_queries if dynamic_queries else [
                'caching strategies', 'load balancing', 'rate limiting', 'message queues', 'event driven architecture',
                'microservices observability', 'database replication', 'consistency tradeoffs', 'api gateway', 'circuit breaker'
            ]

        if self.cfg.rag_max_queries:
            queries = (self.cfg.rag_queries or default_queries)[:self.cfg.rag_max_queries]
        else:
            queries = (self.cfg.rag_queries or default_queries)[:max(count, 5)]

        # Retrieve
        all_docs = []
        seen_snips = set()
        for q in queries:
            try:
                over_k = k * (3 if (self.cfg.restrict_sources or self.cfg.include_tags or self.cfg.include_h1) else 1)
                docs = []
                try:
                    # Prefer filtering by H1 metadata (single-theme mode)
                    docs = self._vs.similarity_search(q, k=over_k, filter={"h1": q})
                    if not docs:
                        q_slug = self._slugify(q)
                        docs = self._vs.similarity_search(q, k=over_k, filter={"h1": q_slug})
                    if not docs:
                        docs = self._vs.similarity_search(q, k=over_k)
                except Exception as e:
                    log("warn", f"retrieval failed for '{q}': {e}")
                    continue
            except Exception as e:
                log("warn", f"retrieval failed for '{q}': {e}")
                continue
            for d in docs:
                snippet = d.page_content[:1000].strip()
                if snippet in seen_snips:
                    continue
                seen_snips.add(snippet)
                all_docs.append((q, d, snippet))

        # Filter cascade
        def _slug(s: str) -> str: return self._slugify(s)
        def _filter(docs_list):
            if not (self.cfg.restrict_sources or self.cfg.include_tags or self.cfg.include_h1):
                return docs_list
            out = []
            for q, d, snippet in docs_list:
                md = getattr(d, 'metadata', {}) or {}
                src = (md.get('source') or '').lower()
                tags = [t.strip().lower() for t in (md.get('tags') or '').split(',') if t.strip()]
                h1_slug = _slug(md.get('h1') or '')
                keep = True
                if self.cfg.restrict_sources:
                    import fnmatch
                    if not any((fnmatch.fnmatch(src, pat.lower()) if ('*' in pat or '?' in pat) else (pat.lower() in src))
                               for pat in self.cfg.restrict_sources):
                        keep = False
                if keep and self.cfg.include_tags:
                    # Support matching by TF-IDF keywords in tags
                    if not any(t in tags for t in [t.lower() for t in self.cfg.include_tags]):
                        keep = False
                if keep and self.cfg.include_h1:
                    if not h1_slug or all(ih not in h1_slug for ih in [_slug(h) for h in self.cfg.include_h1]):
                        keep = False
                if keep:
                    out.append((q, d, snippet))
            return out

        working = _filter(all_docs)
        if not working and (self.cfg.include_h1 or self.cfg.include_tags or self.cfg.restrict_sources):
            if self.cfg.include_h1:
                tmp_cfg = Config(**{**self.cfg.__dict__, "include_h1": None})
                self.cfg = tmp_cfg
                working = _filter(all_docs)
            if not working and self.cfg.include_tags:
                tmp_cfg = Config(**{**self.cfg.__dict__, "include_tags": None})
                self.cfg = tmp_cfg
                working = _filter(all_docs)
            if not working and self.cfg.restrict_sources:
                tmp_cfg = Config(**{**self.cfg.__dict__, "restrict_sources": None})
                self.cfg = tmp_cfg
                working = _filter(all_docs)

        # Build RAG_CONTEXT.md
        if not working:
            log("warn", "RAG retrieval produced no context; no data available for quiz generation.")
            return {}, queries

        header = [
            "# Retrieved Knowledge (Citations)",
            "Each question MUST be grounded in one or more cited sections. Do NOT invent facts.",
            "Guidance: Derive 'topic' from the PRIMARY cited section heading; keep it concise (1-4 words, Title Case). NEVER use 'RAG_CONTEXT'.",
            "Each retrieval batch is restricted to a single theme (H1) selected randomly."
        ]
        bodies = []
        citation_idx = 1
        for _q, d, snippet in working:
            heading = d.metadata.get('section_heading') or (snippet.split('\n', 1)[0][:80])
            source = (d.metadata.get('source') or d.metadata.get('rel_path') or d.metadata.get('path') or 'unknown')
            label = f"C{citation_idx}"; citation_idx += 1
            header.append(f"{label}: {source} :: {heading}")
            bodies.append(f"[{label}] (source: {source}, heading: {heading})\n{snippet}")

        content = "\n".join(header) + "\n\n---\n\n" + "\n\n".join(bodies)
        if self.cfg.dump_rag_context:
            try:
                Path(self.cfg.dump_rag_context).write_text(content, encoding='utf-8')
                log("debug", f"Wrote full RAG context -> {self.cfg.dump_rag_context}")
            except Exception as e:
                log("warn", f"Could not write RAG context: {e}")
        return {'RAG_CONTEXT.md': content}, queries

# =========================
# Quiz Orchestration
# =========================

class Quiz:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.providers = Providers(cfg)
        self.rag = RAG(cfg)

    def _recent_history(self) -> List[str]:
        if not self.cfg.fresh or not HISTORY_FILE.exists():
            return []
        try:
            loaded = json.loads(HISTORY_FILE.read_text(encoding='utf-8'))
            if isinstance(loaded, list):
                return [re.sub(r'\s+', ' ', q.lower()).strip() for q in loaded][-80:]
        except Exception:
            pass
        return []

    def _save_history(self, questions: List[Question]) -> None:
        if not self.cfg.fresh:
            return
        try:
            items = [re.sub(r'\s+', ' ', q.question.lower()).strip() for q in questions]
            if HISTORY_FILE.exists():
                try:
                    existing = json.loads(HISTORY_FILE.read_text(encoding='utf-8'))
                    if not isinstance(existing, list):
                        existing = []
                except Exception:
                    existing = []
            else:
                existing = []
            HISTORY_FILE.write_text(json.dumps((existing + items)[-100:], indent=2), encoding='utf-8')
        except Exception as e:
            log("warn", f"Could not update history: {e}")

    def validate(self, questions: List[Question], expected: int) -> Optional[str]:
        if len(questions) != expected:
            return f'Expected {expected} questions, got {len(questions)}'
        for q in questions:
            if q.answer.upper() not in ['A','B','C','D']:
                lower = q.answer.strip().lower()
                for idx,opt in enumerate(q.options):
                    if lower == opt.lower() or opt.lower().startswith(lower[:5]):
                        q.answer = chr(ord('A')+idx)
                        break
            if len(q.options) != 4:
                return f'Question {q.id} does not have 4 options'
        return None

    def _gen_one(self, files_for_q: Dict[str,str], provider: str, token: str,
                 recent_norm: List[str], temperature: float, iteration_index: int, theme: Optional[str] = None) -> List[Question]:
        if provider == 'ollama':
            return self.providers.ollama_questions(
                files_for_q, 1, self.cfg.ollama_model, token, recent_norm, temperature,
                snippet_chars=self.cfg.ollama_snippet_chars,
                corpus_chars=self.cfg.ollama_corpus_chars,
                num_predict=self.cfg.ollama_num_predict,
                top_k=self.cfg.ollama_top_k,
                top_p=self.cfg.ollama_top_p,
                compact_json=self.cfg.ollama_compact_json,
                debug_payload=self.cfg.debug_ollama_payload,
                iteration=iteration_index,
                theme=theme
            )
        else:
            return self.providers.openai_questions(
                files_for_q, 1, self.cfg.model, token, recent_norm, temperature,
                iteration=iteration_index
            )

    def run(self) -> Tuple[List[Question], Dict[str,str]]:
        if self.cfg.count < 1:
            raise RuntimeError('--count must be at least 1')

        # Only use the vector database for context
        files_ctx, queries = self.rag.build_context({}, k=self.cfg.rag_k, count=self.cfg.count)
        recent_norm = self._recent_history()

        provider = 'ollama' if self.cfg.ollama else 'openai'
        base_temp = 0.6 if self.cfg.fresh else 0.4
        temperature = self.cfg.ollama_temperature if (self.cfg.ollama and self.cfg.ollama_temperature is not None) else base_temp

        questions: List[Question] = []
        for idx in range(self.cfg.count):
            token = str(uuid.uuid4())
            theme = None

            # Per-Q small context using query (if available)
            files_single = files_ctx
            if queries and self.rag._vs:
                q = queries[idx % len(queries)]
                theme = q
                try:
                    docs = []
                    try:
                        q_slug = self.rag._slugify(q)
                        docs = self.rag._vs.similarity_search(q, k=self.cfg.rag_k, filter={"h1": q})
                        if not docs:
                            docs = self.rag._vs.similarity_search(q, k=self.cfg.rag_k, filter={"h1": q_slug})
                        if not docs:
                            docs = self.rag._vs.similarity_search(q, k=self.cfg.rag_k)
                    except Exception as e:
                        log("warn", f"Per-question retrieval failed: {e}")
                        docs = []
                    if docs:
                        seen, blocks = set(), []
                        for d in docs:
                            snippet = d.page_content[:1000].strip()
                            if snippet in seen: continue
                            seen.add(snippet)
                            heading = d.metadata.get('section_heading') or (snippet.split('\n',1)[0][:80])
                            source  = (d.metadata.get('source') or d.metadata.get('rel_path') or d.metadata.get('path') or 'unknown')
                            blocks.append(f"[C1] (source: {source}, heading: {heading})\n{snippet}")
                        if blocks:
                            header = "\n".join([
                                "# Retrieved Knowledge (Citations)",
                                f"Query: {q}",
                                "Each question MUST be grounded in one or more cited sections. Do NOT invent facts.",
                                "Guidance: Derive 'topic' from the PRIMARY cited section heading; keep it concise (1-4 words, Title Case). NEVER use 'RAG_CONTEXT'."
                            ])
                            files_single = {'RAG_CONTEXT.md': header + "\n\n---\n\n" + "\n\n".join(blocks)}
                except Exception as e:
                    log("warn", f"Per-question retrieval failed: {e}")
            elif queries:
                q = queries[idx % len(queries)]
                theme = q

            qlist = self._gen_one(files_single, provider, token, recent_norm, temperature, idx, theme=theme)
            if qlist:
                q = qlist[0]
                q.id = f"Q{idx+1}"
                questions.append(q)

        # Optional verify with Ollama
        if self.cfg.verify and provider == 'ollama' and self.providers._requests:
            corrected = 0
            for q in questions:
                verify_prompt = (
                    "Verify the correctness of the provided answer letter. "
                    "Return JSON {\"correct\":bool, \"correct_answer\":\"A-D\"}.\n"
                    f"Question: {q.question}\n"
                    f"A. {q.options[0]}\nB. {q.options[1]}\nC. {q.options[2]}\nD. {q.options[3]}\n"
                    f"Current answer: {q.answer}\n"
                )
                try:
                    resp = self.providers._requests.post(
                        'http://localhost:11434/api/generate',
                        json={'model': self.cfg.ollama_model, 'prompt': verify_prompt, 'stream': False, 'options': {'temperature': 0.0}},
                        timeout=90
                    )
                    if resp.status_code != 200:
                        continue
                    data = resp.json().get('response', '')
                    m = re.search(r"```json\n(.*)```", data, re.DOTALL)
                    payload = m.group(1) if m else data
                    js = json.loads(payload)
                    if isinstance(js, dict) and not js.get('correct', True):
                        new_ans = js.get('correct_answer')
                        if new_ans in ['A','B','C','D'] and new_ans != q.answer:
                            q.answer = new_ans
                            corrected += 1
                except Exception:
                    pass
            if corrected:
                log("info", f"Verification adjusted {corrected} answer(s).")

        return questions, files_ctx

# =========================
# Main
# =========================

def main(argv: List[str]) -> int:
    try:
        cfg = parse_args(argv)
        quiz = Quiz(cfg)
        questions, _ = quiz.run()

        err = quiz.validate(questions, cfg.count)
        if err:
            log("error", f"Validation failed: {err}")
            return 1

        if cfg.dry_run:
            log("info", "Dry run complete.")
            return 0

        write_outputs(questions, cfg.quiz, cfg.answers)
        quiz._save_history(questions)
        log("ok", f"Wrote {len(questions)} questions -> {cfg.quiz} and answer key -> {cfg.answers}")
        return 0
    except KeyboardInterrupt:
        _orig_print("\nInterrupted.")
        return 130
    except Exception as e:
        log("error", f"Generation failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))