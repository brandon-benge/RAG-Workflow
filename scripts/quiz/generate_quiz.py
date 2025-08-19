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

# --- Constants ---
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_HTTP_TIMEOUT = 240
VERIFY_HTTP_TIMEOUT = 90

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
    dump_llm_payload: Optional[str]
    dump_llm_response: Optional[str]
    template: bool
    seed: int
    avoid_recent_window: int
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
    p.add_argument('--dump-llm-payload')
    p.add_argument('--dump-llm-response')

    # Flow
    p.add_argument('--template', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--avoid-recent-window', type=int, required=True, help='Avoid reusing the last N questions (required int argument)')
    p.add_argument('--verify', action='store_true')
    p.add_argument('--dry-run', action='store_true')

    # RAG
    p.add_argument('--rag-persist', default='.chroma')
    p.add_argument('--rag-k', type=int, default=4)
    p.add_argument('--rag-queries', nargs='+')
    p.add_argument('--rag-max-queries', type=int)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--rag-local', dest='rag_local', action='store_true', default=True)
    group.add_argument('--rag-openai', dest='rag_local', action='store_false')
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
        dump_llm_payload=a.dump_llm_payload,
        dump_llm_response=a.dump_llm_response,
        template=a.template,
        seed=a.seed,
        avoid_recent_window=a.avoid_recent_window,
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
    avoid_recent_window: int

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
    # Only accept a list of dicts as the top-level JSON
    try:
        data = json.loads(raw_json)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and provider == 'ollama':
            items = [data]
        else:
            raise RuntimeError(f'{provider}: expected a list of questions, got {type(data)}')
        out: List[Question] = []

        def format_options(options):
            """
            Robustly handles options that may be:
            - a list of strings (each option)
            - a single string with embedded newlines
            - a list with one string containing embedded newlines
            Returns a list of clean option strings.
            """
            # Only accept a list of strings
            if isinstance(options, list) and all(isinstance(opt, str) for opt in options):
                return [opt.strip() for opt in options]
            raise RuntimeError('Options must be a list of strings')

        for idx, obj in enumerate(items, start=1):
            if not isinstance(obj, dict):
                continue

            qid = str(obj.get('id') or f'Q{idx}')
            question = (obj.get('question') or '').strip() or f'Placeholder question {idx}'

            raw_opts = obj.get('options')
            options = format_options(raw_opts)

            # Accept either 'answer_letter' or 'answer' as a single letter
            explicit_letter = str(obj.get('answer_letter', '')).strip().upper()
            answer_field = str(obj.get('answer', '')).strip().upper()
            if explicit_letter in ['A', 'B', 'C', 'D']:
                answer_letter = explicit_letter
            elif answer_field in ['A', 'B', 'C', 'D']:
                answer_letter = answer_field
            else:
                raise RuntimeError("answer or answer_letter must be one of A, B, C, D")

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
    except Exception as e:
        # Log the error before dumping the payload
        from pathlib import Path
        import inspect
        log("error", f"LLM output parsing failed: {e}")
        cfg = None
        # Try to get cfg from caller's frame
        for frame in inspect.stack():
            if 'self' in frame.frame.f_locals:
                self_obj = frame.frame.f_locals['self']
                if hasattr(self_obj, 'cfg'):
                    cfg = getattr(self_obj, 'cfg')
                    break
        path = None
        if cfg:
            path = getattr(cfg, 'dump_llm_payload', None)
        if path:
            try:
                Path(path).write_text(raw_json, encoding='utf-8')
            except Exception:
                pass
        raise

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

        import shutil
        import tempfile

        # Move existing dump files to temp directory on startup
        for attr in ['dump_llm_payload', 'dump_llm_response']:
            path = getattr(cfg, attr, None)
            if path and os.path.isfile(path):
                temp_dir = tempfile.gettempdir()
                base = os.path.basename(path)
                new_path = os.path.join(temp_dir, f"{base}.bak")
                try:
                    shutil.move(path, new_path)
                    log("info", f"Moved existing {base} to {new_path}")
                except Exception as e:
                    log("warn", f"Could not move {base}: {e}")

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

    def _dump_payload(self, prompt: str, payload: dict, response: Optional[Any] = None) -> None:
        if self.cfg.dump_ollama_prompt:
            try:
                Path(self.cfg.dump_ollama_prompt).write_text(prompt, encoding='utf-8')
                log("debug", f"Wrote full LLM prompt -> {self.cfg.dump_ollama_prompt}")
            except Exception as e:
                log("warn", f"Could not write prompt: {e}")
        path = self.cfg.dump_llm_payload
        if path:
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
                log("debug", f"Appended LLM payload -> {path}")
            except Exception as e:
                log("warn", f"Could not append payload: {e}")
        resp_path = self.cfg.dump_llm_response
        if resp_path and response is not None:
            try:
                try:
                    text = json.dumps(response, indent=2, ensure_ascii=False)
                except Exception:
                    text = str(response)
                with open(resp_path, "a", encoding="utf-8") as f:
                    f.write(text + "\n")
                log("debug", f"Appended LLM response -> {resp_path}")
            except Exception as e:
                log("warn", f"Could not append LLM response: {e}")

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

        window = self.cfg.avoid_recent_window
        recent_clause = ("Avoid reusing these prior question phrasings: " + '; '.join(recent_norm[-window:])) if recent_norm else ''
        system = (
            "You are an assistant that creates high-quality multiple-choice quiz questions for system design and devops. "
            "If citation lines like 'C<number>:' exist, you MUST base questions on them. "
            "Pick a concise Title Case topic. "
            "Return STRICT JSON with: id, question, options (list of 4), topic, difficulty, answer, explanation. "
            "The 'answer' MUST be a single letter A/B/C/D corresponding to the provided options; you may also include 'answer_letter' as the same letter. "
            "Do NOT put option text in 'answer'. "
            "Return ONLY a single strict JSON object for each question, no array, no markdown, no extra text."
        )
        user = (
            f"Uniqueness token: {token}. Generate {count} questions (IDs Q1..Q{count}). "
            f"{recent_clause} Source material: {corpus[:12000]}"
        )

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
        self._dump_payload(system + '\n' + user, {'model': model, 'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}], 'temperature': temperature}, raw_response)
        questions = _parse_model_questions(json_text, provider='openai')
        for q in questions:
            q.raw_response = raw_response
        return questions

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

        window = self.cfg.avoid_recent_window
        recent_clause = ("Avoid reusing these prior question phrasings: " + '; '.join(recent_norm[-window:])) if recent_norm else ''
        style_clause = 'Return STRICT COMPACT JSON array ONLY.' if compact_json else 'Return STRICT JSON as an array.'
        prompt = (
            "You are an assistant that creates high-quality multiple-choice quiz questions for system design and devops. "
            "If citation lines like 'C<number>:' exist, you MUST base questions on them. "
            "Pick a concise Title Case topic. "
            "NEVER use 'RAG_CONTEXT' as a topic. "
            f"{style_clause} Keys: id, question, options (list of 4), topic, difficulty, answer, explanation. "
            "Requirements: "
            "- 'options' MUST be a list of exactly 4 distinct answer choices. "
            "- 'answer' MUST be a single letter A/B/C/D matching the position in the options list (0=A, 1=B, 2=C, 3=D). "
            "- The explanation MUST clearly state which option is correct and why. "
            "- Double-check that the answer letter matches the correct option referenced in the explanation. "
            "- Do NOT put option text in 'answer'. "
            "Return ONLY a single strict JSON object for the question, no array, no markdown, no extra text.\n"
            f"\nUniqueness token: {token}. Create ONE multiple choice question (ID Q{iteration+1 if iteration is not None else 1}) about system design or devops using ONLY the provided notes.\n"
            f"{recent_clause}\n"
            "Source notes:\n" + (corpus if corpus_chars == -1 else corpus[:corpus_chars])
        )

        options: Dict[str, Any] = {'temperature': temperature}
        if num_predict is not None: options['num_predict'] = num_predict
        if top_k is not None: options['top_k'] = top_k
        if top_p is not None: options['top_p'] = top_p
        payload = {'model': model, 'prompt': prompt, 'stream': False, 'options': options}

        # Dump prompt/payload before the request (no response yet)
        self._dump_payload(prompt, payload, None)

        if debug_payload:
            trunc = prompt[:600] + (f"... [truncated, total {len(prompt)} chars]" if len(prompt) > 600 else "")
            try:
                log("debug", "Ollama request payload (truncated prompt):")
                _orig_print(json.dumps({**payload, "prompt": trunc}, indent=2)[:4000])
            except Exception:
                pass

        start = time.time()
        resp = self._requests.post(OLLAMA_URL, json=payload, timeout=DEFAULT_HTTP_TIMEOUT)
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

        # Optional verification step
        if self.cfg.verify:
            verify_prompt = (
                "Verify the correctness of the provided answer letter. "
                "Return JSON {\"correct\":bool, \"correct_answer\":\"A-D\"}.\n"
                "For each option, ensure it does NOT start with a letter (A, B, C, D) followed by punctuation (such as ':', ')', '.', '-', or whitespace). "
                "If any option starts this way, remove the letter and punctuation so only the answer text remains. "
                "If more than 4 options are present, remove one or more options that are NOT the correct answer so only 4 remain (the correct answer and the most plausible distractors). "
                "Each element in the array is a position in the options list (0=A, 1=B, 2=C, 3=D) but do not need to be labeled. "
                f"Raw LLM response: {json_text}\n"
            )
            try:
                verify_resp = self._requests.post(
                    OLLAMA_URL,
                    json={'model': model, 'prompt': verify_prompt, 'stream': False, 'options': {'temperature': 0.0}},
                    timeout=VERIFY_HTTP_TIMEOUT
                )
                if verify_resp.status_code == 200:
                    verify_data = verify_resp.json().get('response', '')
                    m2 = re.search(r"```json\n(.*)```", verify_data, re.DOTALL)
                    payload2 = m2.group(1) if m2 else verify_data
                    js = json.loads(payload2)
                    if isinstance(js, dict) and not js.get('correct', True):
                        new_ans = js.get('correct_answer')
                        if new_ans in ['A','B','C','D']:
                            json_text = re.sub(r'("answer"\s*:\s*")[A-D](")', f'"answer":"{new_ans}"', json_text)
                            log("info", f"Verification adjusted answer to {new_ans}.")
            except Exception:
                pass

        # Dump request/response after receiving data
        self._dump_payload(prompt, payload, data)

        questions = _parse_model_questions(json_text, provider='ollama')
        for q in questions:
            q.raw_response = data
        return questions


# =========================
# RAG
# =========================

class RAG:
    def get_blocks_for_theme(self, theme: str, k: int) -> Optional[Dict[str, str]]:
        if not self._vs:
            return None
        q = theme
        try:
            q_slug = self._slugify(q)
            docs = self._vs.similarity_search(q, k=k, filter={"h1": q})
            if not docs:
                docs = self._vs.similarity_search(q, k=k, filter={"h1": q_slug})
            if not docs:
                docs = self._vs.similarity_search(q, k=k)
        except Exception as e:
            log("warn", f"Per-question retrieval failed: {e}")
            docs = []
        if not docs:
            return None
        seen, blocks = set(), []
        for d in docs:
            snippet = d.page_content[:1000].strip()
            if snippet in seen:
                continue
            seen.add(snippet)
            heading = d.metadata.get('section_heading') or (snippet.split('\n',1)[0][:80])
            source  = (d.metadata.get('source') or d.metadata.get('rel_path') or d.metadata.get('path') or 'unknown')
            blocks.append(f"[C1] (source: {source}, heading: {heading})\n{snippet}")
        if not blocks:
            return None
        header = "\n".join([
            "# Retrieved Knowledge (Citations)",
            f"Query: {q}",
            "Each question MUST be grounded in one or more cited sections. Do NOT invent facts.",
            "Guidance: Derive 'topic' from the PRIMARY cited section heading; keep it concise (1-4 words, Title Case). NEVER use 'RAG_CONTEXT'."
        ])
        return {'RAG_CONTEXT.md': header + "\n\n---\n\n" + "\n\n".join(blocks)}

    def _fetch_unique_tags(self, db_path: str) -> List[str]:
        """Return a sorted list of unique tag tokens (lowercased) from embedding_metadata."""
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT string_value FROM embedding_metadata WHERE key='tags';")
            rows = cur.fetchall()
            conn.close()
            uniq: set[str] = set()
            for (val,) in rows:
                if not val:
                    continue
                for tok in str(val).split(','):
                    t = tok.strip().lower()
                    if t:
                        uniq.add(t)
            return sorted(uniq)
        except Exception as e:
            log("warn", f"Could not fetch tags: {e}")
            return []

    def get_blocks_for_tag(self, tag: str, k: int) -> Optional[Dict[str, str]]:
        """Fetch a small per-question context constrained by a single tag."""
        if not self._vs:
            return None
        q = tag.strip().lower()
        if not q:
            return None
        try:
            # Try exact/equality filter first, then fall back to similarity + post-filtering
            docs = []
            try:
                docs = self._vs.similarity_search(q, k=k, filter={"tags": q})
            except Exception:
                docs = []
            if not docs:
                docs = self._vs.similarity_search(q, k=k)
        except Exception as e:
            log("warn", f"Per-question retrieval by tag failed: {e}")
            docs = []
        # Post-filter by comma-separated tags in metadata
        kept = []
        for d in docs:
            md = getattr(d, 'metadata', {}) or {}
            tline = (md.get('tags') or '').lower()
            tag_list = [t.strip() for t in tline.split(',') if t.strip()]
            if q in tag_list or any((q == t or q in t) for t in tag_list):
                kept.append(d)
        docs = kept or docs
        if not docs:
            return None
        seen, blocks = set(), []
        for d in docs:
            snippet = d.page_content[:1000].strip()
            if snippet in seen:
                continue
            seen.add(snippet)
            heading = d.metadata.get('section_heading') or (snippet.split('\n',1)[0][:80])
            source  = (d.metadata.get('source') or d.metadata.get('rel_path') or d.metadata.get('path') or 'unknown')
            blocks.append(f"[C1] (source: {source}, heading: {heading})\n{snippet}")
        if not blocks:
            return None
        header = "\n".join([
            "# Retrieved Knowledge (Citations)",
            f"Tag: {q}",
            "Each question MUST be grounded in one or more cited sections. Do NOT invent facts.",
            "Guidance: Derive 'topic' from the PRIMARY cited section heading; keep it concise (1-4 words, Title Case). NEVER use 'RAG_CONTEXT'."
        ])
        return {'RAG_CONTEXT.md': header + "\n\n---\n\n" + "\n\n".join(blocks)}
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
        # New: collect unique tags for theme selection
        all_tags = self._fetch_unique_tags(db_path)

        # Build the tag pool (queries): if --include-tags provided, intersect and keep that order;
        # otherwise use all unique tags discovered in the store.
        if self.cfg.include_tags:
            req = [t.strip().lower() for t in self.cfg.include_tags]
            tag_pool = [t for t in all_tags if t in req]
        else:
            tag_pool = list(all_tags)

        if not tag_pool:
            log("warn", "No tags found in store; RAG will proceed with empty context.")
            return {}, []

        # We no longer build a large static context; each question will fetch its own blocks by tag.
        queries = tag_pool
        return files, queries

# =========================
# Quiz Orchestration
# =========================

class Quiz:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.providers = Providers(cfg)
        self.rag = RAG(cfg)

    def _recent_history(self) -> List[str]:
        if not HISTORY_FILE.exists():
            return []
        try:
            loaded = json.loads(HISTORY_FILE.read_text(encoding='utf-8'))
            if isinstance(loaded, list):
                # Only return the last N questions, where N is avoid_recent_window
                return [re.sub(r'\s+', ' ', q.lower()).strip() for q in loaded][-self.cfg.avoid_recent_window:]
        except Exception:
            pass
        return []

    def _save_history(self, questions: List[Question]) -> None:
        # Always save history for recent question avoidance
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
        base_temp = 0.4  # Use a fixed base temperature; remove fresh logic
        temperature = self.cfg.ollama_temperature if (self.cfg.ollama and self.cfg.ollama_temperature is not None) else base_temp

        questions: List[Question] = []
        for idx in range(self.cfg.count):
            token = str(uuid.uuid4())
            theme = None

            # Per-Q small context using query (if available)
            files_single = files_ctx
            if queries and self.rag._vs:
                if self.cfg.include_tags:
                    q = queries[idx % len(queries)]  # deterministic cycle through provided tags
                else:
                    import random as _r
                    q = _r.choice(queries)            # random tag per question
                theme = q
                maybe = self.rag.get_blocks_for_tag(q, self.cfg.rag_k)
                if maybe:
                    files_single = maybe
            elif queries:
                if self.cfg.include_tags:
                    q = queries[idx % len(queries)]
                else:
                    import random as _r
                    q = _r.choice(queries)
                theme = q

            qlist = self._gen_one(files_single, provider, token, recent_norm, temperature, idx, theme=theme)
            if qlist:
                q = qlist[0]
                q.id = f"Q{idx+1}"
                questions.append(q)
        # Verification now handled in Providers.ollama_questions before parsing.
        return questions, files_ctx

# =========================
# Main
# =========================

def main(argv: List[str]) -> int:
    try:
        cfg = parse_args(argv)
        if not cfg.rag_local and not os.environ.get('OPENAI_API_KEY'):
            log("error", "OpenAI embeddings selected (--rag-openai) but OPENAI_API_KEY is not set. Either set the key or use --rag-local (default).")
            return 1
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