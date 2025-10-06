import re
from typing import List

def simple_text_clean(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    text = simple_text_clean(text)
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks
