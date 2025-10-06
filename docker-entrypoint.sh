set -e
MODEL_PATH=${MODEL_PATH:-/app/models/model.gguf}
if [ ! -f "$MODEL_PATH" ]; then
  echo "Model not found at $MODEL_PATH — downloading from HF..."
  mkdir -p "$(dirname $MODEL_PATH)"
  python /app/scripts/download_model.py \
    --url "https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-UD-Q8_K_XL.gguf" \
    --out "$MODEL_PATH"
fi
exec uvicorn src.retrieve_serve_prod:APP --host 0.0.0.0 --port 8080