import argparse
from pathlib import Path
import requests

def download(url, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # simple streamed download (works for public HF repo file)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print("Downloaded to", out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="Direct HF resolve URL for the gguf file")
    p.add_argument("--out", required=True, help="Output path")
    args = p.parse_args()
    download(args.url, args.out)
