"""Exploratory data analysis (univariate, bivariate, multivariate) and data-driven insights.

Works only on aggregated data (daily counts per jurisdiction/group and anonymous coordinates), never on personal fields.
Every sentence in the insights is generated from computed numbers, so the report cannot drift from the data.

Usage:  python -m app.eda
Writes reports/eda/*.png, reports/eda/insights.json and reports/eda/EDA.md  (git-ignored: derived from real data).
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .features import build_wide, holidays, load_catalog
from .spatial import cell_keys

OUT_DIR = config.ROOT / "reports" / "eda"
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
HOUR_TRUST_FROM = pd.Timestamp("2026-04-01")      # earlier loads were stamped with a default hour (see README, audit)


# ------------------------------------------------------------------ computations (pure, unit-tested)
def daily_total(wide: pd.DataFrame) -> pd.Series:
    """Total incidents per day over days with reliable data."""
    return wide.dropna(how="all").sum(axis=1, min_count=1).dropna()


def overview(wide: pd.DataFrame) -> dict:
    total = daily_total(wide)
    valid = wide.loc[total.index]
    return {
        "days": int(len(total)), "series": int(wide.shape[1]), "incidents": int(total.sum()),
        "mean_per_day": float(total.mean()), "median_per_day": float(total.median()), "std_per_day": float(total.std()),
        "iqr_per_day": float(total.quantile(0.75) - total.quantile(0.25)), "skew": float(total.skew()),
        "zero_share_pct": float(100 * (valid.to_numpy() == 0).mean()),
        "dispersion_index": float(valid.to_numpy().var(ddof=1) / max(valid.to_numpy().mean(), 1e-9)),   # variance/mean; 1 = Poisson
    }


def weekday_profile(total: pd.Series) -> pd.Series:
    """Mean incidents per weekday divided by the overall mean (1.0 = average day)."""
    return (total.groupby(total.index.dayofweek).mean() / total.mean()).reindex(range(7))


def share_by(panel: pd.DataFrame, key: str) -> pd.Series:
    s = panel.groupby(key)["n"].sum()
    return (100 * s / s.sum()).sort_values(ascending=False)


def group_correlation(wide: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between the daily totals of the incident groups."""
    return wide.dropna(how="all").T.groupby(level=1).sum().T.corr()


def autocorrelation(total: pd.Series, lags=(1, 7, 14)) -> dict[int, float]:
    full = total.asfreq("D")
    return {k: float(full.corr(full.shift(k))) for k in lags}


def holiday_effect(total: pd.Series) -> float:
    """% difference between the mean on national holidays and on non-holiday days of the same weekday mix."""
    hol = holidays(range(total.index.year.min(), total.index.year.max() + 1))
    is_h = pd.Series([d.date() in hol for d in total.index], index=total.index)
    if not is_h.any():
        return float("nan")
    weekday_mean = total[~is_h].groupby(total[~is_h].index.dayofweek).mean()
    expected = total[is_h].index.dayofweek.map(weekday_mean).to_numpy()
    return float(100 * (total[is_h].to_numpy().mean() / expected.mean() - 1))


def hourly_profile(geo: pd.DataFrame) -> pd.Series:
    """Incidents per hour of day (only rows whose hour is trustworthy), as a share of the total in %."""
    g = geo[pd.to_datetime(geo["fecha"]) >= HOUR_TRUST_FROM]
    c = g.groupby(g["hora"].astype(int)).size().reindex(range(24), fill_value=0)
    return 100 * c / max(c.sum(), 1)


def spatial_concentration(geo: pd.DataFrame, top_share: float = 0.05) -> dict:
    """Share of incidents that fall in the busiest `top_share` of occupied 200 m cells."""
    g = geo.dropna(subset=["lat", "lon"])
    counts = pd.Series(cell_keys(g["lat"].to_numpy(), g["lon"].to_numpy())).value_counts()
    k = max(1, int(round(top_share * len(counts))))
    return {"cells": int(len(counts)), "top_cells": k, "top_share_pct": float(100 * counts.iloc[:k].sum() / counts.sum())}


def recent_trend(total: pd.Series, weeks: int = 4) -> float:
    """% change of the last `weeks` weeks vs the `weeks` before (continuous tail only)."""
    t = total.asfreq("D")
    cont = t.iloc[np.where(t.isna())[0][-1] + 1:] if t.isna().any() else t
    n = 7 * weeks
    if len(cont) < 2 * n:
        return float("nan")
    return float(100 * (cont.iloc[-n:].mean() / cont.iloc[-2 * n:-n].mean() - 1))


def build_insights(wide: pd.DataFrame, panel: pd.DataFrame, geo: pd.DataFrame, catalog: dict) -> list[dict]:
    total = daily_total(wide)
    prof = weekday_profile(total)
    jur_name = {j["id"]: j["nombre"] for j in catalog["jurisdicciones"]}
    jur_share, grp_share = share_by(panel, "jurisdiccion_id"), share_by(panel, "grupo")
    hours = hourly_profile(geo)
    sp = spatial_concentration(geo)
    ac = autocorrelation(total)
    gc = group_correlation(wide)
    off = gc.where(~np.eye(len(gc), dtype=bool)).stack()
    pair = off.idxmax()
    best3 = max(range(24), key=lambda h: hours.reindex([(h + i) % 24 for i in range(3)]).sum())
    hi, lo = prof.idxmax(), prof.idxmin()
    top2 = jur_share.iloc[:2]
    out = [
        {"title": "The week has a clear rhythm",
         "text": f"{WEEKDAYS[hi]} is the busiest day ({100 * (prof[hi] - 1):+.0f} % vs the average day) and {WEEKDAYS[lo]} the quietest "
                 f"({100 * (prof[lo] - 1):+.0f} %). Daily totals correlate {ac[7]:.2f} with the same day one week earlier "
                 f"(lag 1 day: {ac[1]:.2f}): both the recent level and the weekly pattern carry signal.",
         "evidence": {"weekday_index": {WEEKDAYS[i]: round(float(v), 3) for i, v in prof.items()}, "autocorr": ac}},
        {"title": "A few jurisdictions concentrate the workload",
         "text": f"{jur_name[top2.index[0]]} and {jur_name[top2.index[1]]} account for {top2.sum():.0f} % of all incidents "
                 f"({jur_share.iloc[-1]:.1f} % for the smallest, {jur_name[jur_share.index[-1]]}). Errors in the big ones dominate any aggregate metric.",
         "evidence": {"share_pct": {jur_name[i]: round(float(v), 1) for i, v in jur_share.items()}}},
        {"title": "Incident mix is dominated by a couple of types",
         "text": f"{grp_share.index[0]} ({grp_share.iloc[0]:.0f} %) and {grp_share.index[1]} ({grp_share.iloc[1]:.0f} %) make up "
                 f"{grp_share.iloc[:2].sum():.0f} % of the volume. The strongest co-movement between types is {pair[0]} / {pair[1]} "
                 f"(r = {off.max():.2f}); correlations between raw daily totals include the weekly and seasonal pattern they share.",
         "evidence": {"share_pct": {k: round(float(v), 1) for k, v in grp_share.items()}}},
        {"title": "Incidents peak in a specific part of the day",
         "text": f"The busiest 3-hour window is {best3:02d}:00-{(best3 + 3) % 24:02d}:00, holding {hours.reindex([(best3 + i) % 24 for i in range(3)]).sum():.0f} % "
                 f"of incidents (a flat day would give 12.5 %). Hours are only reliable from {HOUR_TRUST_FROM.date()} (earlier loads used a default hour).",
         "evidence": {"hour_share_pct": {int(h): round(float(v), 2) for h, v in hours.items()}}},
        {"title": "Incidents are spatially concentrated",
         "text": f"The busiest {sp['top_cells']} of {sp['cells']} occupied 200 m cells ({100 * sp['top_cells'] / sp['cells']:.0f} %) hold "
                 f"{sp['top_share_pct']:.0f} % of all geolocated incidents. This concentration is what makes hotspot ranking worthwhile.",
         "evidence": sp},
    ]
    trend, hol = recent_trend(total), holiday_effect(total)
    extra = []
    if not np.isnan(trend):
        extra.append(f"The last 4 weeks are {trend:+.1f} % vs the 4 weeks before (this can reflect more registration effort, not only more incidents)")
    if not np.isnan(hol):
        extra.append(f"national holidays run {hol:+.0f} % vs a normal day of the same weekday")
    if extra:
        out.append({"title": "Short-term context", "text": "; ".join(extra) + ".", "evidence": {"trend_4w_pct": trend, "holiday_effect_pct": hol}})
    return out


# ------------------------------------------------------------------ plots
def _save(fig, name: str) -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=130)
    plt.close(fig)
    return name


def make_plots(wide: pd.DataFrame, panel: pd.DataFrame, geo: pd.DataFrame, catalog: dict) -> list[str]:
    total = daily_total(wide)
    jur_name = {j["id"]: j["nombre"].title() for j in catalog["jurisdicciones"]}
    files = []

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].hist(total, bins=30, color="#4c78a8"); ax[0].set_title("Daily incidents - distribution"); ax[0].set_xlabel("incidents / day")
    ax[1].boxplot(total, vert=False); ax[1].set_title("Daily incidents - boxplot"); ax[1].set_yticks([])
    files.append(_save(fig, "01_univariate_daily_total.png"))

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    js = share_by(panel, "jurisdiccion_id").rename(index=jur_name)
    ax[0].barh(js.index[::-1], js.values[::-1], color="#59a14f"); ax[0].set_title("Share by jurisdiction (%)")
    gs = share_by(panel, "grupo")
    ax[1].barh(gs.index[::-1], gs.values[::-1], color="#f28e2b"); ax[1].set_title("Share by incident group (%)")
    files.append(_save(fig, "02_univariate_shares.png"))

    prof = weekday_profile(total)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(WEEKDAYS, prof.values, color="#e15759"); ax.axhline(1, color="grey", lw=1, ls="--")
    ax.set_title("Bivariate: weekday vs incidents (1.0 = average day)")
    files.append(_save(fig, "03_bivariate_weekday.png"))

    hours = hourly_profile(geo)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(hours.index, hours.values, color="#76b7b2"); ax.set_title("Bivariate: hour of day vs incidents (% of total)"); ax.set_xlabel("hour")
    files.append(_save(fig, "04_bivariate_hour.png"))

    gc = group_correlation(wide)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(gc.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(gc))); ax.set_xticklabels(gc.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(gc))); ax.set_yticklabels(gc.index, fontsize=8)
    for i in range(len(gc)):
        for j in range(len(gc)):
            ax.text(j, i, f"{gc.iat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(gc.iat[i, j]) > 0.6 else "black")
    ax.set_title("Multivariate: correlation between incident groups"); fig.colorbar(im, ax=ax, shrink=0.8)
    files.append(_save(fig, "05_multivariate_group_correlation.png"))

    jt = wide.dropna(how="all").T.groupby(level=0).sum().T.rename(columns=jur_name)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    jt.rolling(7, min_periods=1).mean().plot(ax=ax, lw=1.2); ax.set_title("7-day mean of incidents by jurisdiction"); ax.legend(fontsize=7, ncol=4)
    files.append(_save(fig, "06_trend_by_jurisdiction.png"))
    return files


def main() -> None:
    catalog = load_catalog()
    panel = pd.read_csv(config.PANEL_CSV, parse_dates=["fecha"])
    geo = pd.read_csv(config.GEO_CSV)
    wide = build_wide(panel, catalog)
    ov, insights = overview(wide), build_insights(wide, panel, geo, catalog)
    files = make_plots(wide, panel, geo, catalog)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "insights.json").write_text(json.dumps({"overview": ov, "insights": insights}, indent=2), encoding="utf-8")
    md = ["# Exploratory data analysis", "", "## Overview", "",
          f"- {ov['incidents']:,} incidents over {ov['days']} reliable days, {ov['series']} series (jurisdiction x group)",
          f"- Per day: mean {ov['mean_per_day']:.0f}, median {ov['median_per_day']:.0f}, std {ov['std_per_day']:.0f}, IQR {ov['iqr_per_day']:.0f}, skew {ov['skew']:.2f}",
          f"- {ov['zero_share_pct']:.0f} % of series-days are zero; variance/mean = {ov['dispersion_index']:.1f} (over-dispersed: 1 would be Poisson)", "",
          "## Insights", ""]
    md += [f"{i}. **{x['title']}.** {x['text']}" for i, x in enumerate(insights, 1)]
    md += ["", "## Charts", ""] + [f"![{f}]({f})" for f in files]
    (OUT_DIR / "EDA.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:12 + len(insights)]))
    print(f"\nWritten to {OUT_DIR}")


if __name__ == "__main__":
    main()
