from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import subprocess

def extract_pdf_texts(pdf_paths: List[Path]) -> Tuple[List[str], List[str], List[Path]]:
    texts: List[str] = []
    h1s: List[str] = []
    keep_paths: List[Path] = []
    for pdf_path in pdf_paths:
        try:
            result = subprocess.run(
                ['pdftotext', str(pdf_path), '-'],
                capture_output=True,
                text=True,
                check=True
            )
        except Exception:
            continue
        text = result.stdout or ''
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            continue
        first_line = lines[0]
        if first_line.lower().startswith('contents') and len(lines) > 1:
            h1 = lines[1]
        else:
            h1 = first_line
        texts.append(text)
        h1s.append(h1)
        keep_paths.append(pdf_path)
    return texts, h1s, keep_paths

def extract_keywords_tfidf(texts: List[str], top_n: int = 20) -> List[List[str]]:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    keywords_per_doc: List[List[str]] = []
    for doc_idx in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix.getrow(doc_idx)
        top_indices = row.toarray()[0].argsort()[-top_n:][::-1]
        keywords = [feature_names[i] for i in top_indices if row[0, i] > 0]
        keywords_per_doc.append(keywords)
    return keywords_per_doc
