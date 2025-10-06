import os, math, numpy as np, faiss
from typing import List

def train_ivfpq(vectors: np.ndarray, nlist: int = 1024, m: int = 64, nbits: int = 8, index_path: str = 'trained_ivfpq.index'):
    d = vectors.shape[1]
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
    index.train(vectors)
    index_ivf_pq = faiss.IndexIDMap(index)
    faiss.write_index(index_ivf_pq, index_path)
    return index_path

def create_sharded_indices(vectors: np.ndarray, ids: np.ndarray, shard_size: int, trained_index_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    n = vectors.shape[0]
    shards = math.ceil(n / shard_size)
    for i in range(shards):
        start = i * shard_size
        end = min(n, (i+1) * shard_size)
        vchunk = vectors[start:end]
        idchunk = ids[start:end].astype('int64')
        idx = faiss.read_index(trained_index_path)
        faiss.normalize_L2(vchunk)
        idx.add_with_ids(vchunk, idchunk)
        outp = os.path.join(out_dir, f'index_shard_{i}.index')
        faiss.write_index(idx, outp)
    return out_dir
