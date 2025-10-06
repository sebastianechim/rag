import os
import argparse
import requests
import time
from pathlib import Path

def download_with_retries(url, out_path, token=None, attempts=3, delay=5):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".download")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for attempt in range(1, attempts + 1):
        try:
            with requests.get(url, stream=True, headers=headers, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            tmp.replace(out)
            size = out.stat().st_size
            print(f"Downloaded {out} ({size} bytes)")
            return out
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < attempts:
                print(f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="Direct URL to the GGUF file")
    p.add_argument("--out", required=True, help="Local output path (file)")
    p.add_argument("--token", default=None, help="Optional Hugging Face token")
    args = p.parse_args()
    download_with_retries(args.url, args.out, token=args.token, attempts=4, delay=5)