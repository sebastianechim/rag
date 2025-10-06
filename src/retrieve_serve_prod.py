import os, json, time
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import numpy as np
import faiss
from llama_cpp import Llama

APP = FastAPI(title='RAG Retrieval Service (prod)')
MODEL_PATH = os.environ.get('MODEL_PATH','/app/models/model.gguf')
INDEX_LOCAL_DIR = os.environ.get('INDEX_LOCAL_DIR','/app/index_files')
TOP_K = int(os.environ.get('TOP_K','4'))

class Query(BaseModel):
    query: str
    top_k: int | None = None
    user_id: str | None = None

llm = None
index = None
meta = []

@APP.on_event('startup')
def startup_event():
    global llm, index, meta
    llm = Llama(model_path=MODEL_PATH, embedding=True)
    idx_path = os.path.join(INDEX_LOCAL_DIR,'faiss.index')
    meta_path = os.path.join(INDEX_LOCAL_DIR,'index_meta.json')
    if os.path.exists(idx_path):
        index = faiss.read_index(idx_path)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

def embed_query(text: str):
    r = llm.create_embedding(text)
    v = np.array(r['data'][0]['embedding'], dtype=np.float32)
    faiss.normalize_L2(v.reshape(1, -1))
    return v

def do_retrieval(q: str, k: int):
    if index is None:
        raise RuntimeError('Index not loaded')
    v = embed_query(q).reshape(1, -1)
    D, I = index.search(v, k)
    hits = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0:
            continue
        entry = {}
        if meta:
            try:
                entry = meta[idx].copy()
            except Exception:
                entry = {}
        entry['_score'] = float(score)
        hits.append(entry)
    return hits

@APP.get('/health')
def health():
    ok = (llm is not None) and (index is not None)
    return {'status': 'ok' if ok else 'loading'}

@APP.post('/query')
def query(q: Query, background_tasks: BackgroundTasks):
    t0 = time.time()
    k = q.top_k or TOP_K
    hits = do_retrieval(q.query, k)
    context = '\n\n'.join([h.get('text','') for h in hits])
    prompt = f"""Use the context to answer. If unknown, say I don't know.

    Context:
    {context}

    User: {q.query}
    Answer:"""
    resp = llm.create_completion(prompt, max_tokens=256, temperature=0.2)
    answer = resp.get('choices', [{}])[0].get('text', '').strip()
    latency = (time.time() - t0) * 1000.0
    return {'answer': answer, 'retrieved': hits, 'latency_ms': latency}
