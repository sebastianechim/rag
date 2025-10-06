import os
import time
import pytest

try:
    from llama_cpp import Llama
except Exception:
    Llama = None


MODEL_ENV = "CI_MODEL_PATH"  # CI sets this to the downloaded model path
DEFAULT_MODEL = os.path.expanduser("~/.cache/llama_models/gemma-270m.gguf")


@pytest.fixture(scope="module")
def model_path():
    path = os.environ.get(MODEL_ENV, DEFAULT_MODEL)
    if not os.path.isfile(path):
        # If model missing, skip the test (so CI can explicitly fail if you want)
        pytest.skip(f"GGUF model not found at {path} - set {MODEL_ENV} or put file there")
    return path


@pytest.mark.timeout(120)  # requires pytest-timeout plugin if present, optional
def test_model_load_embed_and_generate(model_path):
    """Smoke test: load GGUF with llama-cpp-python, embed and generate one completion."""
    if Llama is None:
        pytest.skip("llama-cpp-python not installed in environment")

    t0 = time.time()
    llm = Llama(model_path=model_path, embedding=True)
    load_time = time.time() - t0
    assert load_time < 300, f"Model load too slow: {load_time:.1f}s"

    # embedding
    emb = llm.create_embedding("pytest smoke test")
    assert "data" in emb and isinstance(emb["data"], list)
    assert len(emb["data"]) >= 1
    vec = emb["data"][0].get("embedding")
    assert isinstance(vec, (list, tuple)), "embedding not returned as list/tuple"
    assert len(vec) > 4

    # generation
    if hasattr(llm, "create_completion"):
        resp = llm.create_completion("Say 'pytest smoke' in one short sentence.", max_tokens=40, temperature=0.0)
    elif hasattr(llm, "create"):
        resp = llm.create(prompt="Say 'pytest smoke' in one short sentence.", max_tokens=40, temperature=0.0)
    else:
        pytest.skip("No generation API found on Llama object")

    text = resp.get("choices", [{}])[0].get("text", "")
    assert isinstance(text, str)
    assert len(text.strip()) > 0, "Generation returned empty text"