import os, json, time
from typing import List, Dict, Optional
import numpy as np
import faiss
from llama_cpp import Llama

INDEX_DIR = os.environ.get('INDEX_LOCAL_DIR','index_files')
META_FILENAME = 'index_meta.json'

class Indexer:
    def __init__(self, model_path: str = None):
        model_path = model_path or os.environ.get('MODEL_PATH','models/model.gguf')
        self.model = Llama(model_path=model_path, embedding=True)
        self.dim = None

    def embed_texts(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for t in batch:
                r = self.model.create_embedding(t)
                emb = np.array(r['data'][0]['embedding'], dtype=np.float32)
                out.append(emb)
        arr = np.vstack(out).astype(np.float32)
        if self.dim is None:
            self.dim = arr.shape[1]
        return arr

    def incremental_add(self, vectors: np.ndarray, index_path: str):
        os.makedirs(os.path.dirname(index_path) or '.', exist_ok=True)
        if os.path.exists(index_path):
            idx = faiss.read_index(index_path)
        else:
            d = vectors.shape[1]
            idx = faiss.IndexFlatIP(d)
        faiss.normalize_L2(vectors)
        idx.add(vectors)
        faiss.write_index(idx, index_path)
        return index_path

    def save_meta_json(self, meta: List[Dict], out_path: str):
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path,'w') as f:
            json.dump(meta, f, indent=2)

def ingest_documents(docs: List[Dict], out_index_dir: str = INDEX_DIR):
    from src.chunker import chunk_text
    idx = Indexer()
    all_chunks = []
    meta_rows = []
    for doc in docs:
        doc_id = doc.get('id')
        text = doc.get('text','')
        chunks = chunk_text(text)
        for i,c in enumerate(chunks):
            chunk_id = f"{doc_id}__{i}"
            all_chunks.append(c)
            meta_rows.append({'chunk_id':chunk_id,'doc_id':doc_id,'text':c,'ingested_at':int(time.time())})
    if not all_chunks:
        print('No chunks to process.')
        return
    vectors = idx.embed_texts(all_chunks, batch_size=16)
    index_path = os.path.join(out_index_dir,'faiss.index')
    os.makedirs(out_index_dir, exist_ok=True)
    idx.incremental_add(vectors, index_path)
    meta_path = os.path.join(out_index_dir, META_FILENAME)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            existing = json.load(f)
        existing.extend(meta_rows)
        idx.save_meta_json(existing, meta_path)
    else:
        idx.save_meta_json(meta_rows, meta_path)
    return index_path, meta_path
