"""Download official benchmark datasets used by Study Agent evaluation.

Currently supports BEIR datasets such as SciFact.
Usage:
    python scripts/download_dataset.py scifact
"""

from __future__ import annotations

import hashlib
import shutil
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

import certifi


DATASETS = {
    "scifact": {
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "md5": "5f7d1de60b170fc8027bb7898e2efca1",
    },
}


def download(name: str) -> Path:
    if name not in DATASETS:
        raise SystemExit(
            f"Unknown dataset {name!r}. Available: {', '.join(DATASETS)}"
        )

    root = Path(__file__).resolve().parents[1]
    data_root = root / "data" / "benchmarks"
    data_root.mkdir(parents=True, exist_ok=True)

    spec = DATASETS[name]
    archive = data_root / f"{name}.zip"
    target = data_root / name

    print(f"Downloading {name} from the official BEIR source...")
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    request = urllib.request.Request(
        spec["url"],
        headers={"User-Agent": "Study-Agent/1.0"},
    )
    with urllib.request.urlopen(request, context=ssl_context) as response:
        archive.write_bytes(response.read())

    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    if digest != spec["md5"]:
        archive.unlink(missing_ok=True)
        raise RuntimeError(
            f"MD5 mismatch for {archive.name}: expected {spec['md5']}, got {digest}"
        )

    if target.exists():
        shutil.rmtree(target)

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(data_root)

    print(f"Downloaded and extracted to: {target}")
    return target


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/download_dataset.py scifact")
    download(sys.argv[1])
