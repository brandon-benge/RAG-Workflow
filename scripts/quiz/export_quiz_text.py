#!/usr/bin/env python3
from __future__ import annotations
import json, argparse
from pathlib import Path

def parse_args(argv):
    p=argparse.ArgumentParser(description='Export quiz JSON to markable text file')
    p.add_argument('--quiz',default='quiz.json'); p.add_argument('--out',default='quiz.txt'); p.add_argument('--force',action='store_true'); return p.parse_args(argv)

def main(argv=None):
    import sys
    args=parse_args(argv or sys.argv[1:]); quiz_path=Path(args.quiz); out_path=Path(args.out)
    if not quiz_path.exists(): print(f'[error] quiz file {quiz_path} not found'); return 1
    if out_path.exists() and not args.force: print(f'[warn] {out_path} already exists (use --force)'); return 1
    quiz=json.loads(quiz_path.read_text(encoding='utf-8'))
    lines=['# Quiz (mark answers by putting x inside the [ ] for exactly one option per question)', '# Example: [X] B. Explanation', '']
    for q in quiz:
        lines.append(f"{q['id']}: {q['question']} (Topic: {q.get('topic','')} | Difficulty: {q.get('difficulty','')})")
        for idx,opt in enumerate(q['options']):
            letter=chr(ord('A')+idx); lines.append(f'[ ] {letter}. {opt}')
        lines.append('')
    out_path.write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(f'[ok] Wrote markable quiz template -> {out_path}'); return 0

if __name__=='__main__':
    import sys; raise SystemExit(main())
