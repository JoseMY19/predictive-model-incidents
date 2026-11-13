"""Backtest del modelo espacial y elección de parámetros.

Uso:  python -m app.spatial_eval
Para cada semana de las últimas ventanas: usa solo datos anteriores, pronostica los próximos 7 días (modelo
temporal + reparto por celdas) y mide qué parte de las incidencias REALES cayó en las celdas marcadas.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from . import config
from .features import build_wide, load_catalog
from .modeling import fit, predict
from .spatial import SpatialModel, select_top

HALF_LIVES = [None, 14.0, 21.0, 30.0, 60.0]   # None = todo el historial con el mismo peso (baseline estático)
BLENDS = [0.0, 0.3, 0.6, 1.0]
H = 7
TOPK = 20


def main() -> None:
    catalog = load_catalog()
    wide = build_wide(pd.read_csv(config.PANEL_CSV, parse_dates=["fecha"]), catalog)
    last = pd.Timestamp(catalog["ultima_fecha"])
    grupos = catalog["grupos"]
    kind = json.loads(config.METRICS_JSON.read_text(encoding="utf-8"))["modelo"]
    geo = pd.read_csv(config.GEO_CSV, parse_dates=["fecha"])
    sm = SpatialModel.build(geo, grupos, last)
    n = len(sm.lat)
    kpat = grupos.index("PATRIMONIO")
    print(f"Celdas activas: {n} | {sm.stats}")

    # (origen, lam por (jur,grupo) para los 7 días siguientes), con el modelo entrenado solo con datos previos a cada ventana
    cases = []
    for i in range(2):  # las 2 últimas ventanas de BACKTEST_DAYS días
        te = last - pd.Timedelta(days=config.BACKTEST_DAYS * i)
        ts = te - pd.Timedelta(days=config.BACKTEST_DAYS - 1)
        model, cols, _ = fit(kind, wide, train_end=ts - pd.Timedelta(days=1))
        o = ts - pd.Timedelta(days=1)
        while o + pd.Timedelta(days=H) <= te:
            fc = predict(kind, model, cols, wide, o, H)
            cases.append((o, {c: float(fc[c].sum()) for c in fc.columns}))
            o += pd.Timedelta(days=H)
    print(f"Orígenes semanales evaluados: {len(cases)}")

    k5, k10 = int(0.05 * n), int(0.10 * n)
    rows = []
    for hl, bl in itertools.product(HALF_LIVES, BLENDS):
        if hl is None and bl not in (0.0, 0.3):
            continue
        acc = {"all5": 0.0, "all10": 0.0, "allTop": 0.0, "tot": 0.0, "pat5": 0.0, "pat10": 0.0, "patTop": 0.0, "ptot": 0.0}
        for o, lam in cases:
            oi = (o - sm.days[0]).days
            A = sm.M[oi + 1: oi + 1 + H].sum(axis=0)                  # real (celdas, grupos)
            L = sm.expected(o, lam, {"half_life": hl, "blend": bl})
            sa, aa = L.sum(axis=1), A.sum(axis=1)
            acc["tot"] += aa.sum()
            order = np.argsort(-sa)
            acc["all5"] += aa[order[:k5]].sum(); acc["all10"] += aa[order[:k10]].sum()
            acc["allTop"] += aa[select_top(sa, sm.lat, sm.lon, TOPK)].sum()
            sp, ap = L[:, kpat], A[:, kpat]
            acc["ptot"] += ap.sum()
            order = np.argsort(-sp)
            acc["pat5"] += ap[order[:k5]].sum(); acc["pat10"] += ap[order[:k10]].sum()
            acc["patTop"] += ap[select_top(sp, sm.lat, sm.lon, TOPK)].sum()
        rows.append({"half_life": hl, "blend": bl,
                     "todos_top5pct": acc["all5"] / acc["tot"], "todos_top10pct": acc["all10"] / acc["tot"],
                     "todos_top20_puntos": acc["allTop"] / acc["tot"],
                     "patrimonio_top5pct": acc["pat5"] / acc["ptot"], "patrimonio_top10pct": acc["pat10"] / acc["ptot"],
                     "patrimonio_top20_puntos": acc["patTop"] / acc["ptot"]})
    df = pd.DataFrame(rows)
    pd.options.display.float_format = "{:.3f}".format
    print(df.to_string(index=False))

    cand = df[df["half_life"].notna()].copy()
    # promedio de 4 métricas (todos/robos-hurtos x top 5%/10%): más estable que optimizar una sola
    cand["score"] = cand[["todos_top5pct", "todos_top10pct", "patrimonio_top5pct", "patrimonio_top10pct"]].mean(axis=1)
    best = cand.sort_values("score", ascending=False).iloc[0]
    static = df[(df["half_life"].isna()) & (df["blend"] == 0.0)].iloc[0]
    params = {"half_life": float(best["half_life"]), "blend": float(best["blend"])}
    print(f"\nParámetros elegidos: {params}")
    out = {
        "parametros": params, "celdas": n, "celda_m": 200, "horizonte_dias": H, "origenes": len(cases),
        "azar": {"top5pct": 0.05, "top10pct": 0.10},
        "elegido": {k: float(best[k]) for k in df.columns if k not in ("half_life", "blend")},
        "baseline_estatico": {k: float(static[k]) for k in df.columns if k not in ("half_life", "blend")},
        "limpieza": sm.stats,
        "grilla": json.loads(df.to_json(orient="records")),
    }
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.SPATIAL_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Guardado: {config.SPATIAL_JSON}")


if __name__ == "__main__":
    main()
