"""Send realistic traffic to the API so monitoring dashboards have data.

    python scripts/generate_traffic.py --url http://localhost:8000 --n 200
    python scripts/generate_traffic.py --url http://localhost:8080 --n 100   # Kubernetes

Patients are sampled from the cleaned dataset; a fraction of requests is deliberately
invalid to demonstrate 4xx monitoring. Status 0 in the summary means a connection error.
Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "heart_disease_clean.csv"
INT_FIELDS = {"age", "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"}


def load_patients() -> list[dict]:
    with DATA.open() as fh:
        rows = list(csv.DictReader(fh))
    patients = []
    for row in rows:
        row.pop("target", None)
        patient = {}
        for key, value in row.items():
            if value == "":
                continue  # optional field left out -> imputed by the model
            patient[key] = int(float(value)) if key in INT_FIELDS else float(value)
        patients.append(patient)
    return patients


def post(url: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"{url}/predict",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 (user supplied URL)
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, {}
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        return 0, {}


def wait_until_healthy(url: str, timeout: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=5) as resp:  # noqa: S310
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(2)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--invalid-ratio", type=float, default=0.1)
    parser.add_argument("--delay", type=float, default=0.1, help="seconds between requests")
    args = parser.parse_args()
    url = args.url.rstrip("/")

    if not wait_until_healthy(url):
        raise SystemExit(f"API at {url} is not healthy - is the container/server running?")

    patients = load_patients()
    statuses, labels = Counter(), Counter()
    for i in range(args.n):
        payload = dict(random.choice(patients))
        if random.random() < args.invalid_ratio:
            payload["cp"] = 9  # invalid code -> 422
        status, body = post(url, payload)
        statuses[status] += 1
        if body:
            labels[body["label"]] += 1
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{args.n} sent - status counts {dict(statuses)}")
        time.sleep(args.delay)
    print(f"Done. HTTP statuses: {dict(statuses)}; predictions: {dict(labels)}")


if __name__ == "__main__":
    main()
