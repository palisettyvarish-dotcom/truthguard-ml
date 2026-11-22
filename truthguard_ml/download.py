"""Dataset downloader. NOT RUN by the author (no network in the build sandbox).

URLs are from memory and may have moved: check each official page first. The command
refuses to run unless --accept-licenses is given, and it writes a checksum manifest.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time, urllib.request, zipfile, tarfile
from pathlib import Path
from . import licenses

URLS = {  # verify against the official pages in licenses.DATASETS
    "fever/train.jsonl": "https://fever.ai/download/fever/train.jsonl",
    "fever/shared_task_dev.jsonl": "https://fever.ai/download/fever/shared_task_dev.jsonl",
    "fever/wiki-pages.zip": "https://fever.ai/download/fever/wiki-pages.zip",
    "scifact/data.tar.gz": "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
    "liar/liar_dataset.zip": "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip",
}


def fetch(url: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            f.write(chunk); h.update(chunk)
    return h.hexdigest()


def export_vitaminc(raw: Path) -> None:
    from datasets import load_dataset  # pip install datasets
    ds = load_dataset("tals/vitaminc")
    (raw / "vitaminc").mkdir(parents=True, exist_ok=True)
    for split, d in ds.items():
        d.to_json(str(raw / "vitaminc" / f"{'dev' if split == 'validation' else split}.jsonl"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--only", nargs="*", help="subset of: fever scifact liar vitaminc")
    ap.add_argument("--skip-wiki", action="store_true", help="skip the large FEVER wiki dump")
    ap.add_argument("--accept-licenses", action="store_true")
    a = ap.parse_args(argv)
    print(licenses.license_banner(), file=sys.stderr)
    if not a.accept_licenses:
        sys.exit("Read the licenses above, then re-run with --accept-licenses.")
    want = set(a.only or ["fever", "scifact", "liar", "vitaminc"])
    manifest = {}
    for rel, url in URLS.items():
        ds = rel.split("/")[0]
        if ds not in want or (a.skip_wiki and "wiki-pages" in rel):
            continue
        dest = a.raw / rel
        manifest[rel] = {"url": url, "sha256": fetch(url, dest), "fetched": time.strftime("%F %T")}
        if rel.endswith(".zip"):
            with zipfile.ZipFile(dest) as z:
                z.extractall(dest.parent / ("wiki-pages" if "wiki" in rel else ""))
        if rel.endswith(".tar.gz"):
            with tarfile.open(dest) as t:
                t.extractall(dest.parent, filter="data")
    if "vitaminc" in want:
        export_vitaminc(a.raw); manifest["vitaminc"] = {"source": "hf:tals/vitaminc"}
    (a.raw / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("Downloaded. Verify the layout against pipeline.py's docstring (SciFact tarball unpacks into data/).")


if __name__ == "__main__":
    main()
