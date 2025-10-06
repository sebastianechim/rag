FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN apt-get update && apt-get install -y build-essential libgomp1 curl && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV MODEL_PATH=/app/models/model.gguf
ENV INDEX_LOCAL_DIR=/app/index_files
EXPOSE 8080
CMD ["uvicorn", "src.retrieve_serve_prod:APP", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
