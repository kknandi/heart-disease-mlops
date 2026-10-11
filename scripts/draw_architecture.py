"""Render the system architecture diagram to reports/figures/architecture.png.

python scripts/draw_architecture.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports" / "figures" / "architecture.png"

LANES = [
    ("Data & EDA", "#e8f1f8"),
    ("Training & Tracking", "#eef7ee"),
    ("CI/CD - GitHub Actions", "#fdf3e6"),
    ("Serving - Docker / Kubernetes", "#f3eefa"),
    ("Monitoring & Logging", "#fdecec"),
]
BOX_W, BOX_H = 2.3, 0.9


def box(ax, x, y, text, color="#ffffff"):
    ax.add_patch(
        FancyBboxPatch(
            (x - BOX_W / 2, y - BOX_H / 2),
            BOX_W,
            BOX_H,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            fc=color,
            ec="#333333",
            lw=1.1,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=8.5, wrap=True)
    return (x, y)


def side(b, where):
    """Anchor point on a box edge: 'l', 'r', 't' or 'b'."""
    x, y = b
    return {
        "l": (x - BOX_W / 2, y),
        "r": (x + BOX_W / 2, y),
        "t": (x, y + BOX_H / 2),
        "b": (x, y - BOX_H / 2),
    }[where]


def arrow(ax, a, b, text=None):
    """Straight arrow between box centres (horizontal or vertical neighbours)."""
    (x1, y1), (x2, y2) = a, b
    if abs(x2 - x1) > abs(y2 - y1):
        p1, p2 = (side(a, "r"), side(b, "l")) if x2 > x1 else (side(a, "l"), side(b, "r"))
    else:
        p1, p2 = (side(a, "t"), side(b, "b")) if y2 > y1 else (side(a, "b"), side(b, "t"))
    path(ax, [p1, p2], text)


def path(ax, points, text=None):
    """Poly-line through `points` with an arrow head on the last segment."""
    for (x1, y1), (x2, y2) in zip(points[:-2], points[1:-1], strict=True):
        ax.plot([x1, x2], [y1, y2], color="#444444", lw=1.2)
    ax.add_patch(
        FancyArrowPatch(
            points[-2], points[-1], arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#444444"
        )
    )
    if text:
        (x1, y1), (x2, y2) = points[-2], points[-1]
        horizontal = abs(x2 - x1) > abs(y2 - y1)
        ax.text(
            (x1 + x2) / 2 + (0 if horizontal else 0.08),
            (y1 + y2) / 2 + (0.1 if horizontal else 0),
            text,
            ha="center" if horizontal else "left",
            va="bottom" if horizontal else "center",
            fontsize=7.5,
            color="#555",
        )


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10.4)
    ax.axis("off")

    lane_h = 2.0
    for i, (name, color) in enumerate(LANES):
        y0 = 10 - (i + 1) * lane_h
        ax.add_patch(plt.Rectangle((0.1, y0 + 0.05), 15.8, lane_h - 0.1, fc=color, ec="none"))
        ax.text(0.25, y0 + lane_h - 0.25, name, fontsize=10, fontweight="bold", va="top")

    ys = [10 - (i + 0.55) * lane_h for i in range(len(LANES))]
    xs = [1.8, 4.6, 7.4, 10.2, 13.0]

    # Lane 1 - data
    uci = box(ax, xs[0], ys[0], "UCI ML Repository\nheart+disease.zip")
    dl = box(ax, xs[1], ys[0], "scripts/download_data.py\n(sha256 + row check)")
    clean = box(ax, xs[2], ys[0], "heart_disease.data\nvalidate / clean / binarise")
    csv = box(ax, xs[3], ys[0], "data/processed/\nheart_disease_clean.csv")
    eda = box(ax, xs[4], ys[0], "EDA notebook +\nheart_disease.eda figures")
    for a, b in [(uci, dl), (dl, clean), (clean, csv), (csv, eda)]:
        arrow(ax, a, b)

    # Lane 2 - training
    feat = box(ax, xs[0], ys[1], "Pipeline: ClinicalFeatureAdder\n-> impute/scale/one-hot")
    gs = box(ax, xs[1], ys[1], "GridSearchCV x3 families\nLR / RF / GB (5-fold CV)")
    mlf = box(
        ax, xs[2], ys[1], "MLflow Tracking\nparams, metrics, plots,\nnested candidate runs", "#fff"
    )
    reg = box(ax, xs[3], ys[1], "MLflow Model Registry\nheart-disease-classifier\n@champion")
    art = box(ax, xs[4], ys[1], "models/model.joblib\n+ metadata.json")
    gap12 = 10 - lane_h
    path(ax, [side(csv, "b"), (xs[3], gap12), (xs[0], gap12), side(feat, "t")])
    for a, b in [(feat, gs), (gs, mlf), (mlf, reg), (reg, art)]:
        arrow(ax, a, b)

    # Lane 3 - CI/CD
    push = box(ax, xs[0], ys[2], "git push / PR\n(GitHub)")
    lint = box(ax, xs[1], ys[2], "Lint\nruff check + format")
    test = box(ax, xs[2], ys[2], "Unit tests (pytest)\ncoverage + JUnit artifacts")
    trn = box(ax, xs[3], ys[2], "Train + quality gate\n(test ROC-AUC >= 0.85)\nMLflow artifacts")
    dock = box(ax, xs[4], ys[2], "Docker build + smoke test\n/health, /predict\npush -> GHCR")
    for a, b in [(push, lint), (lint, test), (test, trn), (trn, dock)]:
        arrow(ax, a, b)

    # Lane 4 - serving
    img = box(ax, xs[0], ys[3], "Docker image\npython:3.13-slim\nnon-root, healthcheck")
    dep = box(ax, xs[1], ys[3], "K8s Deployment\n2 replicas, probes,\nrolling update")
    api = box(ax, xs[2], ys[3], "FastAPI pods\nPOST /predict  GET /health\n/model-info  /metrics")
    svc = box(ax, xs[3], ys[3], "Service (LoadBalancer :8080)\n+ Ingress (nginx)")
    cli = box(ax, xs[4], ys[3], "Client\ncurl / Swagger UI /\nnotebook")
    gap34 = 10 - 3 * lane_h
    path(ax, [side(dock, "b"), (xs[4], gap34), (xs[0], gap34), side(img, "t")], text="")
    ax.text(xs[2], gap34 + 0.08, "image (GHCR / local)", ha="center", fontsize=7.5, color="#555")
    arrow(ax, img, dep)
    arrow(ax, dep, api, text="runs")
    arrow(ax, svc, api)
    arrow(ax, cli, svc, text="JSON")

    # Lane 5 - monitoring
    logs = box(
        ax, xs[1], ys[4], "Structured JSON logs\n(request_id, latency,\nprediction - no PII)"
    )
    prom = box(ax, xs[2], ys[4], "Prometheus\nscrapes /metrics (5s)")
    graf = box(
        ax, xs[3], ys[4], "Grafana dashboard\nRPS, p95 latency, errors,\nclass mix, prob. drift"
    )
    traffic = box(ax, xs[4], ys[4], "scripts/\ngenerate_traffic.py")
    arrow(ax, api, prom, text="scrape /metrics")
    arrow(ax, prom, graf)
    arrow(ax, dep, logs, text="stdout")
    arrow(ax, traffic, cli, text="load")

    ax.set_title(
        "Heart Disease Risk Prediction - MLOps Architecture", fontsize=15, fontweight="bold"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
