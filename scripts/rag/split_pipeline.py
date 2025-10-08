from __future__ import annotations
from typing import List, Tuple, Optional

DEFAULTS = {
    "word": (200, 40),
    "sentence": (6, 1),
    "line": (40, 10),
    "passage": (1, 0),
    "page": (1, 0),
    "document": (1, 0),
}

def config_split_params(split_by: str, chunk_size: Optional[int], chunk_overlap: Optional[int]) -> Tuple[str, int, int]:
    if split_by not in DEFAULTS:
        split_by = 'sentence'
    d_len, d_ov = DEFAULTS[split_by]
    split_length = chunk_size if chunk_size is not None else d_len
    split_overlap = chunk_overlap if chunk_overlap is not None else d_ov
    if split_by in ("page", "document"):
        split_length, split_overlap = 1, 0
    else:
        if split_length < 1:
            split_length = d_len
        if split_overlap < 0:
            split_overlap = 0
        if split_overlap >= split_length:
            suggested = max(0, min(d_ov, split_length // 5))
            split_overlap = suggested
    return split_by, split_length, split_overlap

def build_tokenizer(local: bool, model: str):
    tokenizer = None
    tokenizer_type = 'none'
    if local:
        try:
            from transformers import AutoTokenizer  # type: ignore
            tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
            tokenizer_type = 'hf'
        except Exception:
            pass
    else:
        try:
            import tiktoken  # type: ignore
            tokenizer = tiktoken.get_encoding("cl100k_base")
            tokenizer_type = 'tiktoken'
        except Exception:
            pass
    return tokenizer, tokenizer_type

def count_tokens_and_log(splits, tokenizer, tokenizer_type: str) -> Tuple[List, int, list[int], Optional[int]]:
    total_tokens = 0
    # Suppress HF verbosity while encoding
    prev_hf_verbosity = None
    if tokenizer_type == 'hf':
        try:
            from transformers.utils import logging as hf_logging  # type: ignore
            prev_hf_verbosity = hf_logging.get_verbosity()
            hf_logging.set_verbosity_error()
        except Exception:
            prev_hf_verbosity = None
    for s in splits:
        txt = s.content or ""
        try:
            if tokenizer_type == 'hf':
                count = len(tokenizer.encode(txt, add_special_tokens=False))
            else:
                count = len(tokenizer.encode(txt))
        except Exception:
            count = 0
        s.meta = dict(s.meta or {})
        s.meta['token_count'] = int(count)
        total_tokens += count
    if tokenizer_type == 'hf' and prev_hf_verbosity is not None:
        try:
            from transformers.utils import logging as hf_logging  # type: ignore
            hf_logging.set_verbosity(prev_hf_verbosity)
        except Exception:
            pass
    tok_counts = [int((s.meta or {}).get('token_count', 0)) for s in splits]
    tok_counts.sort()
    n = len(tok_counts)
    avg_tokens = total_tokens // max(n, 1)
    p50 = tok_counts[int(0.50 * (n - 1))] if n else 0
    p95 = tok_counts[int(0.95 * (n - 1))] if n else 0
    tmax = tok_counts[-1] if n else 0
    limit = None
    if tokenizer_type == 'hf':
        try:
            limit = int(getattr(tokenizer, 'model_max_length', 0) or 0)
        except Exception:
            limit = None
    if limit and limit > 0:
        over = sum(1 for c in tok_counts if c > limit)
        print(f"Generated {n} chunks (avg ~{avg_tokens} tokens, p50={p50}, p95={p95}, max={tmax}, over_limit={over}/{n} @>{limit}).")
    else:
        print(f"Generated {n} chunks (avg ~{avg_tokens} tokens, p50={p50}, p95={p95}, max={tmax}).")
    return splits, (limit or 0), tok_counts, avg_tokens

def auto_cap(local: bool, tokenizer, tokenizer_type: str, tok_counts: list[int], splits, cap_override: Optional[int] = None):
    # Choose cap
    if cap_override and cap_override > 0:
        cap = cap_override
    else:
        if tokenizer_type == 'hf':
            try:
                cap = int(getattr(tokenizer, 'model_max_length', 0) or 0)
                if cap <= 0 or cap > 100000:
                    cap = 512
            except Exception:
                cap = 512
        else:
            cap = 8192
    if not cap or cap <= 0:
        return splits, cap
    # Suppress HF verbosity while encoding
    prev_hf_verbosity = None
    if tokenizer_type == 'hf':
        try:
            from transformers.utils import logging as hf_logging  # type: ignore
            prev_hf_verbosity = hf_logging.get_verbosity()
            hf_logging.set_verbosity_error()
        except Exception:
            prev_hf_verbosity = None
    overlap_tok = max(0, min(32, cap // 10))
    new_splits = []
    for s in splits:
        tc = int((s.meta or {}).get('token_count', 0))
        if tc <= cap:
            new_splits.append(s)
            continue
        text = s.content or ""
        try:
            if tokenizer_type == 'hf':
                ids = tokenizer.encode(text, add_special_tokens=False)
                decode = lambda toks: tokenizer.decode(toks, skip_special_tokens=True)
            else:
                ids = tokenizer.encode(text)
                decode = lambda toks: tokenizer.decode(toks)
        except Exception:
            new_splits.append(s)
            continue
        i = 0
        while i < len(ids):
            end = min(i + cap, len(ids))
            sub_ids = ids[i:end]
            sub_txt = decode(sub_ids)
            sub_meta = dict(s.meta or {})
            sub_meta['token_count'] = int(len(sub_ids))
            new_splits.append(type(s)(content=sub_txt, meta=sub_meta))
            if end >= len(ids):
                break
            i = end - overlap_tok if overlap_tok > 0 else end
    if tokenizer_type == 'hf' and prev_hf_verbosity is not None:
        try:
            from transformers.utils import logging as hf_logging  # type: ignore
            hf_logging.set_verbosity(prev_hf_verbosity)
        except Exception:
            pass
    # Log post-cap distribution
    tok_counts = [int((s.meta or {}).get('token_count', 0)) for s in new_splits]
    n = len(new_splits)
    if tok_counts:
        tok_counts.sort()
        avg_tokens = sum(tok_counts) // max(n, 1)
        p50 = tok_counts[int(0.50 * (n - 1))]
        p95 = tok_counts[int(0.95 * (n - 1))]
        tmax = tok_counts[-1]
        over = sum(1 for c in tok_counts if c > cap)
        print(f"After capping @{cap} tokens: {n} chunks (avg ~{avg_tokens}, p50={p50}, p95={p95}, max={tmax}, over_limit={over}/{n}).")
    return new_splits, cap
