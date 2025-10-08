#!/usr/bin/env python3
import sys
import subprocess
import datetime
import json
from pathlib import Path
from typing import List, Optional, Any, Dict

# Best-effort import of PyYAML; if missing, try to install it automatically
def _import_yaml_with_bootstrap() -> Any:
    try:
        import yaml  # type: ignore
        return yaml
    except Exception:
        # Attempt to install PyYAML into the current interpreter environment
        try:
            args = [sys.executable, '-m', 'pip', 'install', 'PyYAML']
            # If outside a venv, prefer user install to avoid permission errors
            try:
                in_venv = (hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
            except Exception:
                in_venv = False
            if not in_venv:
                args.append('--user')
            subprocess.run(args, check=False)
            import yaml  # type: ignore
            return yaml
        except Exception:
            return None

yaml = _import_yaml_with_bootstrap()

# Simple logger
def log(level: str, msg: str) -> None:
    print(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] [{level}] {msg}")

# Check if params.yaml exists and log its status
def check_yaml_config(repo_root: Path) -> bool:
    yaml_path = repo_root / 'params.yaml'
    if yaml_path.exists():
        log('info', f"Found YAML config file: {yaml_path}")
        return True
    else:
        log('info', f"YAML config file not found: {yaml_path} (using defaults)")
        return False

# Read config params
def load_yaml(params_path: Path) -> Dict[str, Any]:
    log('info', f"Loading YAML params from: {params_path}")
    if not params_path.exists():
        log('error', f"Params file not found: {params_path}")
        sys.exit(1)
    if yaml is None:
        # Minimal fallback: try JSON parse if file happens to be JSON-compatible
        try:
            data = json.loads(params_path.read_text(encoding='utf-8'))
            return data if isinstance(data, dict) else {}
        except Exception:
            log('error', 'Could not load YAML. Please run ./scripts/bin/run_venv.sh to install dependencies or install PyYAML.')
            sys.exit(1)
    data = yaml.safe_load(params_path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        log('error', 'Invalid YAML structure: expected a top-level mapping')
        sys.exit(1)
    return data

# Require keys from section
def get_build(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if 'build' not in cfg:
        log('error', "Missing 'build' section in params.yaml")
        sys.exit(1)
    b = cfg['build'] or {}
    # Required keys; chunk size/overlap are optional now (auto-configured by split_by)
    must = ['persist','model','bundle_url']
    out = {}
    for k in must:
        if k not in b:
            log('error', f"Missing key '{k}' in build")
            sys.exit(1)
        out[k] = str(b[k])
        log('info', f"build[{k}] = {out[k]}")
    out['local'] = bool(str(b.get('local','true')).lower() in ('1','true','yes','y'))
    out['openai'] = bool(str(b.get('openai','false')).lower() in ('1','true','yes','y'))
    out['force'] = bool(str(b.get('force','false')).lower() in ('1','true','yes','y'))
    # Optional split controls
    out['split_by'] = str(b.get('split_by', 'sentence'))
    if 'chunk_size' in b:
        out['chunk_size'] = str(b.get('chunk_size'))
    if 'chunk_overlap' in b:
        out['chunk_overlap'] = str(b.get('chunk_overlap'))
    if 'max_tokens_per_chunk' in b:
        out['max_tokens_per_chunk'] = int(b.get('max_tokens_per_chunk'))
    # Optional: persist each model under a unique subfolder
    out['persist_by_model'] = bool(str(b.get('persist_by_model','false')).lower() in ('1','true','yes','y'))
    # Optional TF-IDF keywords per doc
    if 'tfidf_top_n' in b:
        try:
            out['tfidf_top_n'] = int(b.get('tfidf_top_n'))
        except Exception:
            out['tfidf_top_n'] = 20
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
    
    # Check for YAML configuration file
    check_yaml_config(repo_root)
    
    params_path = repo_root / 'params.yaml'
    log('info', f"Using params_path: {params_path}")
    cfg = load_yaml(params_path)

    sub = argv[0].lower()
    if sub == 'build':
        # Only pre-build vector store
        b_enabled = True
        if 'build' in cfg and isinstance(cfg['build'], dict):
            b_enabled = bool(str(cfg['build'].get('enabled','true')).lower() in ('1','true','yes','y'))
        if b_enabled:
            runner = repo_root / 'scripts' / 'bin' / 'run_venv.sh'
            # --- Continue with build ---
            build_script = repo_root / 'scripts' / 'rag' / 'vector_store_build.py'
            bvals = get_build(cfg)
            b_local = bvals['local']
            b_openai = bvals['openai']
            if b_local == b_openai:
                log('error', "[build] Exactly one of 'local' or 'openai' must be true")
                return 1
            # Only check/start Ollama when using local embeddings
            if b_local:
                ollama_check = repo_root / 'scripts' / 'rag' / 'check_ollama.py'
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
            # Run builder as a module so relative imports work (requires scripts/ and scripts/rag/ to be packages)
            build_module = 'scripts.rag.vector_store_build'
            build_args = ['-m', build_module]
            if b_local:
                build_args.append('--local')
            else:
                build_args.append('--openai')
            if bvals['force']:
                build_args.append('--force')
            build_args += [
                '--persist', bvals['persist'],
                '--model', bvals['model'],
                '--bundle-url', bvals['bundle_url'],
            ]
            # Pass split settings
            if bvals.get('split_by'):
                build_args += ['--split-by', bvals['split_by']]
            if 'chunk_size' in bvals:
                build_args += ['--chunk-size', bvals['chunk_size']]
            if 'chunk_overlap' in bvals:
                build_args += ['--chunk-overlap', bvals['chunk_overlap']]
            if 'max_tokens_per_chunk' in bvals:
                build_args += ['--max-tokens-per-chunk', str(bvals['max_tokens_per_chunk'])]
            if 'tfidf_top_n' in bvals:
                build_args += ['--tfidf-top-n', str(bvals['tfidf_top_n'])]
            if bvals.get('persist_by_model'):
                build_args += ['--persist-by-model']
            cmd = [str(runner), *build_args] if runner.exists() else [sys.executable, *build_args]
            log('info', f"Executing command: {' '.join(cmd)}")
            rc = subprocess.run(cmd).returncode
            return rc
    # No further steps after build; exit after build step

    elif sub in ('prepare','export','parse','validate'):
        log('warning', "This repository is build-only now. Use the Quiz-Project for quiz generation and validation.")
        qp = repo_root / 'Quiz-Project'
        log('info', f"See: {qp}/README.md (ensure rag_persist points to {(repo_root / '.chroma')})")
        return 2
    else:
        log('error', f"Unknown subcommand: {sub}")
        return 2

if __name__ == "__main__":
    dispatch(sys.argv[1:])