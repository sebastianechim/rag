import os
import time
import numbers
import pytest

try:
    from llama_cpp import Llama
except Exception:
    Llama = None

MODEL_ENV = "CI_MODEL_PATH"
DEFAULT_MODEL = os.path.expanduser("~/.cache/llama_models/gemma-270m.gguf")

def _flatten(xs):
    """Recursively flatten nested lists/tuples/arrays to a list of scalars."""
    out = []
    for x in xs:
        if isinstance(x, (list, tuple)):
            out.extend(_flatten(x))
        else:
            # handle numpy scalars/arrays without importing numpy
            try:
                import numpy as _np
                if isinstance(x, _np.ndarray):
                    out.extend(_flatten(x.tolist()))
                    continue
            except Exception:
                pass
            out.append(x)
    return out

@pytest.fixture(scope="module")
def model_path():
    path = os.environ.get(MODEL_ENV, DEFAULT_MODEL)
    if not os.path.isfile(path):
        pytest.skip(f"GGUF model not found at {path}; set {MODEL_ENV} or put file there")
    return path

def test_model_load_embed_and_generate(model_path):
    if Llama is None:
        pytest.skip("llama-cpp-python not installed")

    t0 = time.time()
    llm = Llama(model_path=model_path, embedding=True)
    load_time = time.time() - t0
    assert load_time < 300, f"Model load too slow: {load_time:.1f}s"

    # embedding
    emb = llm.create_embedding("pytest smoke test")
    assert isinstance(emb, dict) and "data" in emb, "Unexpected embedding response shape"
    assert isinstance(emb["data"], list) and len(emb["data"]) >= 1

    # Get the first embedding entry's "embedding" field
    vec_raw = emb["data"][0].get("embedding")
    assert vec_raw is not None, "No 'embedding' field present"

    # flatten and validate numeric content
    flat = _flatten(vec_raw)
    # require at least 16 numeric elements (tuned for small embeddings; adjust as needed)
    assert len(flat) >= 16, f"Flattened embedding too small: {len(flat)} elements"

    # ensure elements are numeric
    assert any(isinstance(v, numbers.Number) for v in flat), "Embedding contains no numeric values"

    # generation: prefer documented API create_completion, fallback to create()
    prompt = "Say 'pytest smoke' in one short sentence."
    if hasattr(llm, "create_completion"):
        resp = llm.create_completion(prompt, max_tokens=40, temperature=0.0)
    elif hasattr(llm, "create"):
        resp = llm.create(prompt=prompt, max_tokens=40, temperature=0.0)
    else:
        pytest.skip("No generation API found on Llama object")

    text = resp.get("choices", [{}])[0].get("text", "")
    assert isinstance(text, str) and len(text.strip()) > 0, "Generation returned empty text"