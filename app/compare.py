"""Side-by-side comparison of forecasting models on the same rolling backtest windows.

Reports MAE, RMSE and R2 at jurisdiction-day level plus WAPE (daily and weekly), relative to the naive baseline.
Optionally trains every ML model twice (raw target and winsorized target) to show whether outlier capping helps.

Usage:
    python -m app.compare                      # all models
    python -m app.compare --winsorize 98       # also the winsorized variant of every ML model
    python -m app.compare --models random_forest xgboost --folds 2

Writes reports/model_comparison.json and reports/model_comparison.md (git-ignored: derived from real data).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import config
from .features import build_wide, load_catalog
from .modeling import MODEL_LABELS, backtest, backtest_windows

REPORT_DIR = config.ROOT / "reports"
ALL_MODELS = ["baseline_media_7d", "baseline_semana_anterior", "ridge", "knn", "random_forest", "extra_trees",
              "hist_gradient_boosting", "xgboost", "ensemble_rf_es"]
BASELINE = "baseline_media_7d"
METRICS = ["mae_jurisdiccion", "rmse_jurisdiccion", "r2_jurisdiction", "wape_jurisdiccion_pct", "wape_semana_jurisdiccion_pct"]


def evaluate(kind: str, wide: pd.DataFrame, windows, winsorize_pct: float | None = None) -> dict:
    """Backtest `kind` on every window and average the metrics."""
    t0 = time.time()
    folds = [backtest(kind, wide, a, b, winsorize_pct=winsorize_pct)[0] for a, b in windows]
    out = {k: float(np.mean([f[k] for f in folds])) for k in METRICS}
    out["mae_by_window"] = [round(f["mae_jurisdiccion"], 3) for f in folds]
    out["seconds"] = round(time.time() - t0, 1)
    return out


def add_relative(rows: list[dict]) -> list[dict]:
    """Adds `vs_baseline_pct` (negative = better than the baseline MAE)."""
    base = next((r["mae_jurisdiccion"] for r in rows if r["model"] == BASELINE), None)
    for r in rows:
        r["vs_baseline_pct"] = None if base is None else round(100 * (r["mae_jurisdiccion"] - base) / base, 2)
    return rows


def to_markdown(rows: list[dict], windows) -> str:
    lines = [
        "# Model comparison", "",
        f"Rolling backtest: {len(windows)} windows of {config.BACKTEST_DAYS} days ({windows[0][0].date()} -> {windows[-1][1].date()}), "
        f"{config.BACKTEST_HORIZON}-day horizon, metrics at jurisdiction-day level (mean of windows).", "",
        "| Model | MAE | RMSE | R2 | WAPE day | WAPE week | vs baseline | MAE per window |", "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: x["mae_jurisdiccion"]):
        rel = "" if r["vs_baseline_pct"] is None else f"{r['vs_baseline_pct']:+.1f} %"
        lines.append(f"| {r['label']} | {r['mae_jurisdiccion']:.2f} | {r['rmse_jurisdiccion']:.2f} | {r['r2_jurisdiction']:.3f} | "
                     f"{r['wape_jurisdiccion_pct']:.1f} % | {r['wape_semana_jurisdiccion_pct']:.1f} % | {rel} | {r['mae_by_window']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=list(MODEL_LABELS))
    ap.add_argument("--winsorize", type=float, default=None, metavar="PCT", help="also run ML models with the target capped at this percentile")
    ap.add_argument("--folds", type=int, default=config.BACKTEST_FOLDS)
    a = ap.parse_args()

    catalog = load_catalog()
    wide = build_wide(pd.read_csv(config.PANEL_CSV, parse_dates=["fecha"]), catalog)
    windows = backtest_windows(pd.Timestamp(catalog["ultima_fecha"]), folds=a.folds)
    print("windows:", ", ".join(f"{x.date()}->{y.date()}" for x, y in windows))

    rows: list[dict] = []
    for kind in a.models:
        variants = [None] if kind.startswith("baseline") or a.winsorize is None else [None, a.winsorize]
        for w in variants:
            label = MODEL_LABELS[kind] + (f" + winsor P{w:g}" if w else "")
            try:
                r = evaluate(kind, wide, windows, w)
            except ImportError as e:
                print(f"skipped {label}: {e}")
                continue
            rows.append({"model": kind, "winsorize_pct": w, "label": label, **r})
            print(f"{label:<44} MAE={r['mae_jurisdiccion']:.2f}  RMSE={r['rmse_jurisdiccion']:.2f}  R2={r['r2_jurisdiction']:.3f}  ({r['seconds']}s)")

    rows = add_relative(rows)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "model_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md = to_markdown(rows, windows)
    (REPORT_DIR / "model_comparison.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
