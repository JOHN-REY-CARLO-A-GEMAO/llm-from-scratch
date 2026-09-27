#!/usr/bin/env python3
"""Download a compact instruct GGUF from Hugging Face (stdlib only)."""
import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face repo ID")
    ap.add_argument("--file", default=None, help="exact GGUF filename (default: Q4_K_M then first GGUF)")
    ap.add_argument("--output-dir", default="models", help="destination directory")
    args = ap.parse_args()
    api = "https://huggingface.co/api/models/" + urllib.parse.quote(args.repo, safe="/")
    try:
        req = urllib.request.Request(api, headers={"User-Agent": "llm-from-scratch/1.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            metadata = json.load(response)
    except (OSError, urllib.error.URLError) as exc:
        ap.error(f"could not query Hugging Face repo {args.repo}: {exc}")
    files = [s["rfilename"] for s in metadata.get("siblings", [])
             if s.get("rfilename", "").lower().endswith(".gguf")]
    if args.file:
        if args.file not in files:
            ap.error(f"{args.file!r} not found; available GGUF files: {', '.join(files)}")
        filename = args.file
    else:
        filename = next((f for f in files if "q4_k_m" in f.lower()), files[0] if files else None)
    if not filename:
        ap.error(f"no .gguf files found in {args.repo}")
    destination = Path(args.output_dir) / Path(filename).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{args.repo}/resolve/main/{urllib.parse.quote(filename, safe='/')}?download=true"
    request = urllib.request.Request(url, headers={"User-Agent": "llm-from-scratch/1.0"})
    print(f"Downloading {args.repo}/{filename} -> {destination}", flush=True)
    try:
        with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as out:
            total = int(response.headers.get("Content-Length", "0"))
            downloaded = 0
            while chunk := response.read(1024 * 1024):
                out.write(chunk); downloaded += len(chunk)
                if total:
                    print(f"\r{downloaded / total:.0%} ({downloaded / 1e6:.1f}/{total / 1e6:.1f} MB)", end="", flush=True)
    except (OSError, urllib.error.URLError) as exc:
        destination.unlink(missing_ok=True)
        ap.error(f"download failed: {exc}")
    print(f"\nSaved {destination} ({destination.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
