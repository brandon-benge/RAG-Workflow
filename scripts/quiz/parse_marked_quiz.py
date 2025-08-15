#!/usr/bin/env python3
from __future__ import annotations
import re, argparse, json
from pathlib import Path
from typing import Dict
QUESTION_RE=re.compile(r'^(Q\d+):'); OPTION_RE=re.compile(r'^\[(?P<mark>[xX ])\]\s+([A-D])\.\s')

def parse_args(argv):
    p=argparse.ArgumentParser(description='Parse marked quiz text to JSON answers'); p.add_argument('--in',dest='inp',default='quiz.txt'); p.add_argument('--out',dest='out',default='my_answers.json'); p.add_argument('--force',action='store_true'); return p.parse_args(argv)

def main(argv=None):
    import sys
    args=parse_args(argv or sys.argv[1:]); in_path=Path(args.inp); out_path=Path(args.out)
    if not in_path.exists(): print(f'[error] input file {in_path} not found'); return 1
    if out_path.exists() and not args.force: print(f'[warn] output file {out_path} exists (use --force)'); return 1
    answers: Dict[str,str]={}; current_qid=None; current_marks=[]
    def finalize():
        if current_qid and current_marks:
            for letter,marked in current_marks:
                if marked: answers[current_qid]=letter; break
    with in_path.open(encoding='utf-8') as f:
        for line in f:
            line=line.rstrip('\n'); qm=QUESTION_RE.match(line)
            if qm: finalize(); current_qid=qm.group(1); current_marks=[]; continue
            om=OPTION_RE.match(line)
            if om and current_qid:
                marked=om.group('mark').lower()=='x'; letter=line.split()[1].rstrip('.').upper(); current_marks.append((letter,marked))
        finalize()
    out_path.write_text(json.dumps(answers,indent=2)+'\n',encoding='utf-8'); print(f'[ok] Wrote answers JSON -> {out_path} ({len(answers)} answered)'); return 0

if __name__=='__main__':
    import sys; raise SystemExit(main())
