from __future__ import annotations
from pathlib import Path
import tarfile
import urllib.request

def download_bundle(url: str, out_path: Path) -> None:
    urllib.request.urlretrieve(url, out_path)

def safe_extract(tar_gz: Path, dest: Path) -> None:
    with tarfile.open(tar_gz, 'r:gz') as tf:
        tf.extractall(dest, filter='data')

def ensure_pdftotext() -> None:
    import shutil
    if shutil.which('pdftotext') is None:
        raise RuntimeError("'pdftotext' not found. Install poppler-utils (e.g. brew install poppler).")
