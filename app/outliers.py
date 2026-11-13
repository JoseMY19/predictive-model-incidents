"""Outlier analysis (IQR rule) and winsorization of daily counts.

Same idea as the usual "IQR report + winsorize at P98" cleaning step, adapted to count time series:

* the fences are computed **per series** (jurisdiction x incident group), because a count of 30 is normal for one
  series and extreme for another;
* winsorization is only applied to the **training target** and its cap is computed from the training rows alone, so no
  information from the evaluation period leaks into the model. Evaluation always uses the raw, un-capped counts.

Usage:  python -m app.outliers          # prints the IQR report for the current panel
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SERIES_KEYS = ["jurisdiccion_id", "grupo"]


def iqr_bounds(values: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Tukey fences (Q1 - k*IQR, Q3 + k*IQR) ignoring NaN."""
    v = values.dropna()
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def iqr_report(wide: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """One row per series: how many days exceed the upper IQR fence and how large the extremes are."""
    rows = []
    for col in wide.columns:
        s = wide[col].dropna()
        lo, hi = iqr_bounds(s, k)
        over = s[s > hi]
        rows.append({
            "jurisdiccion_id": col[0], "grupo": col[1], "days": len(s), "mean": s.mean(), "p98": s.quantile(0.98),
            "upper_fence": hi, "outlier_days": int(len(over)), "outlier_share_pct": 100 * len(over) / max(len(s), 1),
            "max": s.max(),
        })
    return pd.DataFrame(rows)


def winsorize_caps(df: pd.DataFrame, pct: float = 98.0, target: str = "y") -> pd.Series:
    """Per-series upper cap = the `pct`-th percentile of `target` (index: (jurisdiccion_id, grupo))."""
    if not 50.0 < pct <= 100.0:
        raise ValueError("pct must be in (50, 100]")
    return df.groupby(SERIES_KEYS)[target].quantile(pct / 100.0)


def winsorize_target(df: pd.DataFrame, pct: float = 98.0, target: str = "y") -> pd.DataFrame:
    """Return a copy of `df` where `target` is capped at each series' `pct`-th percentile (computed on `df` only).
    pct = 100 is a no-op. Intended for TRAINING rows."""
    out = df.copy()
    if pct >= 100.0:
        return out
    caps = winsorize_caps(out, pct, target)
    cap_per_row = pd.MultiIndex.from_frame(out[SERIES_KEYS]).map(caps.to_dict().get).to_numpy(dtype=float)
    out[target] = np.minimum(out[target].to_numpy(dtype=float), cap_per_row)
    return out


def main() -> None:
    from . import config
    from .features import build_wide, load_catalog

    wide = build_wide(pd.read_csv(config.PANEL_CSV, parse_dates=["fecha"]), load_catalog())
    rep = iqr_report(wide)
    pd.options.display.float_format = "{:.2f}".format
    print(rep.sort_values("outlier_share_pct", ascending=False).head(15).to_string(index=False))
    total = rep["days"].sum()
    print(f"\n{int(rep['outlier_days'].sum())} outlier series-days of {int(total)} ({100 * rep['outlier_days'].sum() / total:.1f} %) above the IQR fence")


if __name__ == "__main__":
    main()
