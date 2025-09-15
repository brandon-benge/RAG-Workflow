#!/usr/bin/env python3
import sys
import subprocess
import configparser
import datetime
from pathlib import Path
from typing import List, Optional

# Simple logger
def log(level: str, msg: str) -> None:
    print(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] [{level}] {msg}")

# Read config params
def load_params(params_path: Path) -> configparser.ConfigParser:
    log('info', f"Loading config params from: {params_path}")
    if not params_path.exists():
        log('error', f"Params file not found: {params_path}")
        sys.exit(1)
    cfg = configparser.ConfigParser()
    cfg.read(params_path)
    log('info', f"Config loaded. Sections found: {cfg.sections()}")
    return cfg

# Require keys from section
def require_keys(section: str, cfg: configparser.ConfigParser, keys: List[str]) -> dict:
    log('info', f"Checking for section [{section}] and required keys: {keys}")
    if section not in cfg:
        log('error', f"Missing section [{section}] in params file")
        sys.exit(1)
    out = {}
    for k in keys:
        if k not in cfg[section]:
            log('error', f"Missing key '{k}' in section [{section}]")
            sys.exit(1)
        log('info', f"Found key '{k}' in section [{section}]: {cfg[section][k]}")
        out[k] = cfg[section][k]
    log('info', f"All required keys found for section [{section}]. Returning values.")
    return out

# Bool parser
def bool_true(v: str) -> bool:
    return str(v).strip().lower() in ('1','true','yes','y')

# Dispatch based on subcommand
def dispatch(argv: List[str]) -> Optional[int]:
    log('info', f"Entered dispatch with argv: {argv}")
    if not argv:
        log('warning', 'No subcommand provided to dispatch.')
        return None

    repo_root = Path(__file__).resolve().parent
    log('info', f"Resolved repo_root: {repo_root}")
    params_path = repo_root / 'quiz.params'
    log('info', f"Using params_path: {params_path}")
    cfg = load_params(params_path)

    sub = argv[0].lower()
    if sub == 'build':
        # Only pre-build vector store
        if 'build' in cfg and bool_true(cfg['build'].get('enabled','false')):
            # --- Ollama preflight check ---
            ollama_check = repo_root / 'scripts' / 'rag' / 'check_ollama.py'
            runner = repo_root / 'scripts' / 'bin' / 'run_venv.sh'
            log('info', 'Checking if Ollama is running...')
            check_cmd = [str(runner), str(ollama_check), 'check'] if runner.exists() else [sys.executable, str(ollama_check), 'check']
            check_result = subprocess.run(check_cmd)
            if check_result.returncode != 0:
                log('warning', 'Ollama is not running. Attempting to start Ollama...')
                start_cmd = [str(runner), str(ollama_check), 'start'] if runner.exists() else [sys.executable, str(ollama_check), 'start']
                start_result = subprocess.run(start_cmd)
                if start_result.returncode != 0:
                    log('error', 'Failed to start Ollama. Please start it manually and retry.')
                    return 1
                else:
                    log('info', 'Ollama started successfully.')
            else:
                log('info', 'Ollama is already running.')
            # --- Continue with build ---
            build_script = repo_root / 'scripts' / 'rag' / 'vector_store_build.py'
            breq = require_keys('build', cfg, ['persist','chunk_size','chunk_overlap','model','bundle_url'])
            b_local = bool_true(cfg['build'].get('local','false'))
            b_openai = bool_true(cfg['build'].get('openai','false'))
            if b_local == b_openai:
                log('error', "[build] Exactly one of 'local' or 'openai' must be true")
                return 1
            build_args = [str(build_script)]
            if b_local:
                build_args.append('--local')
            else:
                build_args.append('--openai')
            if bool_true(cfg['build'].get('force','false')):
                build_args.append('--force')
            build_args += [
                '--persist', breq['persist'],
                '--chunk-size', breq['chunk_size'],
                '--chunk-overlap', breq['chunk_overlap'],
                '--model', breq['model'],
                '--bundle-url', breq['bundle_url'],
            ]
            cmd = [str(runner), *build_args] if runner.exists() else [sys.executable, *build_args]
            log('info', f"Executing command: {' '.join(cmd)}")
            rc = subprocess.run(cmd).returncode
            return rc
    # No further steps after build; exit after build step

    elif sub == 'prepare':
        # Directly run generate_quiz.py with its own section in quiz.params
        req = require_keys('prepare', cfg, [
            'count','quiz','answers','rag_persist','rag_k','provider','model',
            'avoid_recent_window','verify','rag_local','rag_openai',
            'dump_llm_payload','dump_llm_response','max_retries'
        ])
        rag_local = bool_true(req['rag_local'])
        rag_openai = bool_true(req['rag_openai'])
        if rag_local == rag_openai:
            log('error', "[prepare] Exactly one of 'rag_local' or 'rag_openai' must be true")
            return 1
        args = [
            str(repo_root / 'scripts' / 'quiz' / 'generate_quiz.py'),
            '--count', req['count'],
            '--quiz', req['quiz'],
            '--answers', req['answers'],
            '--rag-persist', req['rag_persist'],
            '--rag-k', req['rag_k'],
            '--' + req['provider'],
            '--' + req['provider'] + '-model', req['model'],
            '--avoid-recent-window', req['avoid_recent_window']
        ]
        # Only add --dump-llm-payload if set and not 'none' or 'false'
        dump_llm_payload = req.get('dump_llm_payload', '').strip().lower()
        if dump_llm_payload and dump_llm_payload not in ('none', 'false'):
            args += ['--dump-llm-payload', req['dump_llm_payload']]
        dump_llm_response = req.get('dump_llm_response', '').strip().lower()
        if dump_llm_response and dump_llm_response not in ('none', 'false'):
            args += ['--dump-llm-response', req['dump_llm_response']]
        # Add --max-retries if set and not 'none' or 'false'
        max_retries = req.get('max_retries', '').strip().lower()
        if max_retries and max_retries not in ('none', 'false'):
            args += ['--max-retries', req['max_retries']]
        if bool_true(req['verify']):
            args.append('--verify')
        if rag_openai:
            args.append('--rag-openai')
        else:
            args.append('--rag-local')
        runner = repo_root / 'scripts' / 'bin' / 'run_venv.sh'
        cmd = [str(runner), *args] if runner.exists() else [sys.executable, *args]
        log('info', f"Executing command: {' '.join(cmd)}")
        return subprocess.run(cmd).returncode
    elif sub == 'export':
        req = require_keys('export', cfg, ['quiz','out'])
        args = [str(repo_root / 'scripts' / 'quiz' / 'export_quiz_text.py'), '--quiz', req['quiz'], '--out', req['out']]
        runner = repo_root / 'scripts' / 'bin' / 'run_venv.sh'
        cmd = [str(runner), *args] if runner.exists() else [sys.executable, *args]
        log('info', f"Executing command: {' '.join(cmd)}")
        return subprocess.run(cmd).returncode
    elif sub == 'parse':
        req = require_keys('parse', cfg, ['in','out'])
        args = [str(repo_root / 'scripts' / 'quiz' / 'parse_marked_quiz.py'), '--in', req['in'], '--out', req['out']]
        runner = repo_root / 'scripts' / 'bin' / 'run_venv.sh'
        cmd = [str(runner), *args] if runner.exists() else [sys.executable, *args]
        log('info', f"Executing command: {' '.join(cmd)}")
        return subprocess.run(cmd).returncode
    elif sub == 'validate':
        req = require_keys('validate', cfg, ['quiz','answers'])
        args = [str(repo_root / 'scripts' / 'quiz' / 'validate_quiz_answers.py'), '--quiz', req['quiz'], '--answers', req['answers']]
        runner = repo_root / 'scripts' / 'bin' / 'run_venv.sh'
        cmd = [str(runner), *args] if runner.exists() else [sys.executable, *args]
        log('info', f"Executing command: {' '.join(cmd)}")
        return subprocess.run(cmd).returncode
    else:
        log('error', f"Unknown subcommand: {sub}")
        return 2

if __name__ == "__main__":
    dispatch(sys.argv[1:])