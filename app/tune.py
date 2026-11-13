"""Hyper-parameter search that respects time (the time-series version of GridSearchCV).

A random K-fold is wrong for forecasting: it would train on days that come *after* the days it is scored on. Here every
candidate is scored with the same rolling-origin backtest used everywhere else:

* the OLDER windows are the *tuning* windows: each parameter combination is trained on the data before a window and scored
  on it (mean MAE at jurisdiction-day level);
* the NEWEST window is a *holdout* that the search never sees. The best combination and the project defaults are both
  scored on it, so the reported gain is not inflated by selection (the README warned that earlier tuning was optimistic).

Usage:
    python -m app.tune                                      # default models
    python -m app.tune --models random_forest xgboost --max-combos 6

Writes reports/tuning.json and reports/tuning.md (git-ignored). Nothing in production changes: adopt a result by editing
`_base_model` in app/modeling.py (or passing `params=` to `fit`).
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import pandas as pd

from . import config
from .features import build_wide, load_catalog
from .modeling import MODEL_LABELS, backtest, backtest_windows

REPORT_DIR = config.ROOT / "reports"
GRIDS: dict[str, dict[str, list]] = {
    "random_forest": {"min_samples_leaf": [5, 15, 30], "max_features": [0.3, 0.5, 0.8]},
    "hist_gradient_boosting": {"learning_rate": [0.03, 0.05, 0.1], "max_iter": [200, 400], "min_samples_leaf": [20, 50]},
    "xgboost": {"max_depth": [3, 5, 7], "learning_rate": [0.03, 0.05], "min_child_weight": [5, 10]},
    "ridge": {"ridge__alpha": [1.0, 10.0, 100.0, 1000.0]},
    "knn": {"kneighborsregressor__n_neighbors": [10, 25, 50]},
}


def expand_grid(grid: dict[str, list]) -> list[dict]:
    """Cartesian product of a parameter grid -> list of parameter dicts (deterministic order)."""
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def split_windows(windows: list) -> tuple[list, tuple]:
    """Older windows for tuning, the newest one as holdout."""
    if len(windows) < 2:
        raise ValueError("need at least 2 windows: one to tune on and one to hold out")
    return windows[:-1], windows[-1]


def mean_mae(kind: str, wide: pd.DataFrame, windows: list, params: dict | None) -> float:
    return float(np.mean([backtest(kind, wide, a, b, params=params)[0]["mae_jurisdiccion"] for a, b in windows]))


def pick_best(scored: list[tuple[dict, float]]) -> tuple[dict, float]:
    """Lowest MAE wins; ties keep the earliest (i.e. the first grid entry)."""
    return min(scored, key=lambda t: t[1])


def tune_model(kind: str, wide: pd.DataFrame, windows: list, grid: dict[str, list], max_combos: int | None = None,
               seed: int = config.RANDOM_STATE) -> dict:
    tune_w, holdout = split_windows(windows)
    combos = expand_grid(grid)
    if max_combos and len(combos) > max_combos:                         # random subset, reproducible
        pick = np.random.default_rng(seed).choice(len(combos), max_combos, replace=False)
        combos = [combos[i] for i in sorted(pick)]
    t0 = time.time()
    default_tune = mean_mae(kind, wide, tune_w, None)
    scored = [(c, mean_mae(kind, wide, tune_w, c)) for c in combos]
    best, best_tune = pick_best(scored)
    default_hold = mean_mae(kind, wide, [holdout], None)
    best_hold = mean_mae(kind, wide, [holdout], best)
    return {
        "model": kind, "label": MODEL_LABELS[kind], "combinations": len(combos), "best_params": best,
        "default_mae_tuning": round(default_tune, 4), "best_mae_tuning": round(best_tune, 4),
        "default_mae_holdout": round(default_hold, 4), "best_mae_holdout": round(best_hold, 4),
        "holdout_gain_pct": round(100 * (best_hold - default_hold) / default_hold, 2),
        "holdout_window": [holdout[0].strftime("%Y-%m-%d"), holdout[1].strftime("%Y-%m-%d")],
        "all": [{"params": c, "mae_tuning": round(m, 4)} for c, m in sorted(scored, key=lambda t: t[1])],
        "seconds": round(time.time() - t0, 1),
    }


def to_markdown(results: list[dict]) -> str:
    lines = ["# Hyper-parameter tuning (time-aware)", "",
             "Tuning on the older windows, scoring on a holdout window the search never saw. Negative gain = better than the defaults.", "",
             "| Model | Combinations | Best parameters | MAE defaults (holdout) | MAE best (holdout) | Holdout gain |", "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['label']} | {r['combinations']} | `{json.dumps(r['best_params'])}` | {r['default_mae_holdout']:.3f} | "
                     f"{r['best_mae_holdout']:.3f} | {r['holdout_gain_pct']:+.1f} % |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["random_forest", "hist_gradient_boosting"], choices=list(GRIDS))
    ap.add_argument("--max-combos", type=int, default=None, help="evaluate a random subset of each grid")
    ap.add_argument("--folds", type=int, default=config.BACKTEST_FOLDS)
    a = ap.parse_args()

    catalog = load_catalog()
    wide = build_wide(pd.read_csv(config.PANEL_CSV, parse_dates=["fecha"]), catalog)
    windows = backtest_windows(pd.Timestamp(catalog["ultima_fecha"]), folds=a.folds)
    print("tuning windows:", ", ".join(f"{x.date()}->{y.date()}" for x, y in windows[:-1]), "| holdout:", f"{windows[-1][0].date()}->{windows[-1][1].date()}")

    results = []
    for kind in a.models:
        try:
            r = tune_model(kind, wide, windows, GRIDS[kind], a.max_combos)
        except ImportError as e:
            print(f"skipped {kind}: {e}")
            continue
        results.append(r)
        print(f"{r['label']:<30} best={r['best_params']}  holdout MAE {r['default_mae_holdout']:.3f} -> {r['best_mae_holdout']:.3f} ({r['holdout_gain_pct']:+.1f} %)  [{r['seconds']}s]")

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "tuning.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (REPORT_DIR / "tuning.md").write_text(to_markdown(results), encoding="utf-8")
    print("\n" + to_markdown(results))


if __name__ == "__main__":
    main()
