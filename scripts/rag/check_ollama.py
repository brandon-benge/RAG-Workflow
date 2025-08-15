#!/usr/bin/env python3
"""(moved to scripts/rag/) Ollama utility: check status, install, start/stop."""
from __future__ import annotations
import argparse, shutil, subprocess, sys, platform
try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore
API_URL = "http://localhost:11434/api/tags"

def _have_brew() -> bool: return shutil.which("brew") is not None

def cmd_install() -> int:
    if platform.system() != "Darwin": print("[error] Brew install only on macOS"); return 1
    if not _have_brew(): print("[error] Homebrew not found"); return 1
    res = subprocess.run(["brew","install","ollama"],text=True); return res.returncode

def _ensure_binary()->bool:
    if not shutil.which("ollama"): print("[fail] 'ollama' not on PATH"); return False
    return True

def cmd_start()->int:
    if platform.system()!="Darwin": print("[error] start only macOS"); return 1
    if not _have_brew(): print('[error] brew missing'); return 1
    if not _ensure_binary(): return 1
    return subprocess.run(["brew","services","start","ollama"],text=True).returncode

def cmd_stop()->int:
    if platform.system()!="Darwin": print("[error] stop only macOS"); return 1
    if not _have_brew(): print('[error] brew missing'); return 1
    return subprocess.run(["brew","services","stop","ollama"],text=True).returncode

def cmd_check()->int:
    if not _ensure_binary(): return 1
    if requests is None: print('[warn] requests not installed'); return 1
    try: r=requests.get(API_URL,timeout=2)
    except Exception as e: print(f"[fail] cannot reach daemon: {e}"); return 1
    if r.status_code!=200: print(f"[fail] HTTP {r.status_code}"); return 1
    try: data=r.json()
    except Exception: print('[fail] bad JSON'); return 1
    models=data.get('models') or []
    print('[ok]' if models else '[warn]', 'models:', ', '.join(m.get('name','?') for m in models[:5]))
    return 0

def parse_args(a):
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['check','install','start','stop']); return p.parse_args(a)

def main(argv):
    c=parse_args(argv).command
    return {'check':cmd_check,'install':cmd_install,'start':cmd_start,'stop':cmd_stop}[c]()

if __name__=='__main__': sys.exit(main(sys.argv[1:]))
