"""Download the UCI Heart Disease dataset into data/raw/.

Usage:
    python scripts/download_data.py                 # download from UCI
    python scripts/download_data.py --from-zip heart+disease.zip   # offline: use a local copy

Only the Python standard library is used so this works before any install step.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
UCI_LEGACY_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)
WANTED_FILES = [
    "processed.cleveland.data",
    "processed.hungarian.data",
    "processed.switzerland.data",
    "processed.va.data",
    "heart-disease.names",
]
PRIMARY_FILE = "processed.cleveland.data"
EXPECTED_ROWS = 303

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("download_data")


def _fetch(url: str, timeout: int = 60) -> bytes:
    log.info("Downloading %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "heart-disease-mlops/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted, fixed URL)
        return resp.read()


def _extract(zip_bytes: bytes, dest: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = {Path(n).name: n for n in zf.namelist()}
        for wanted in WANTED_FILES:
            if wanted in names:
                (dest / wanted).write_bytes(zf.read(names[wanted]))
                log.info("Extracted %s", wanted)


def _verify(path: Path) -> None:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    log.info("%s: %d rows, sha256=%s", path.name, len(lines), sha)
    if len(lines) != EXPECTED_ROWS:
        raise RuntimeError(f"Unexpected row count {len(lines)} (expected {EXPECTED_ROWS})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-zip", type=Path, help="Use a locally downloaded UCI zip file")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / PRIMARY_FILE

    if target.exists() and not args.force and not args.from_zip:
        log.info("%s already present - skipping download (use --force to refresh)", target)
        _verify(target)
        return 0

    try:
        if args.from_zip:
            _extract(args.from_zip.read_bytes(), RAW_DIR)
        else:
            try:
                _extract(_fetch(UCI_ZIP_URL), RAW_DIR)
            except Exception as exc:  # network / format issue -> try legacy endpoint
                log.warning("Zip download failed (%s); trying legacy URL", exc)
                target.write_bytes(_fetch(UCI_LEGACY_URL))
        _verify(target)
    except Exception as exc:
        log.error("Dataset acquisition failed: %s", exc)
        return 1
    log.info("Raw data available in %s", RAW_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
