"""Construcción del panel ancho y de las features (misma lógica para entrenar y pronosticar)."""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
from dateutil.easter import easter

from . import config

LAGS = [1, 2, 3, 7, 14, 21, 28]
ROLLS = [7, 14, 28]
Series = tuple[int, str]  # (jurisdiccion_id, grupo)


def load_catalog() -> dict:
    return json.loads(config.CATALOG_JSON.read_text(encoding="utf-8"))


def series_list(catalog: dict) -> list[Series]:
    return [(j["id"], g) for j in catalog["jurisdicciones"] for g in catalog["grupos"]]


def build_wide(panel: pd.DataFrame, catalog: dict) -> pd.DataFrame:
    """Matriz fecha x serie con conteos. Días sin dato fiable (fuera de rangos válidos) = NaN;
    días válidos sin incidencias = 0."""
    series = series_list(catalog)
    full_idx = pd.date_range(panel["fecha"].min(), catalog["ultima_fecha"], freq="D")
    wide = pd.DataFrame(0.0, index=full_idx, columns=pd.MultiIndex.from_tuples(series))
    p = panel.copy()
    p["fecha"] = pd.to_datetime(p["fecha"])
    for (jur, grp), g in p.groupby(["jurisdiccion_id", "grupo"]):
        if (jur, grp) in wide.columns:
            wide.loc[g["fecha"].values, (jur, grp)] = g["n"].values
    valid = pd.Series(False, index=full_idx)
    for start, end in catalog["rangos_validos"]:
        valid.loc[start:end] = True
    wide.loc[~valid.values, :] = np.nan
    return wide


def holidays(years: range) -> set[date]:
    """Feriados nacionales del Perú (fijos + Semana Santa)."""
    fixed = [(1, 1), (5, 1), (6, 7), (6, 29), (7, 23), (7, 28), (7, 29), (8, 6), (8, 30),
             (10, 8), (11, 1), (12, 8), (12, 9), (12, 25)]
    out: set[date] = set()
    for y in years:
        out |= {date(y, m, d) for m, d in fixed}
        e = easter(y)
        out |= {date.fromordinal(e.toordinal() - 3), date.fromordinal(e.toordinal() - 2)}  # Jue/Vie Santo
    return out


def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    hol = holidays(range(dates.year.min(), dates.year.max() + 1))
    dow = dates.dayofweek
    f = pd.DataFrame(index=dates)
    for d in range(7):
        f[f"dow_{d}"] = (dow == d).astype(float)
    f["dom"] = dates.day.astype(float)
    f["fin_de_mes"] = (dates.day >= 28).astype(float)
    f["feriado"] = np.array([d.date() in hol for d in dates], dtype=float)
    return f


def make_features(wide: pd.DataFrame) -> pd.DataFrame:
    """Tabla larga (una fila por fecha x serie) con features y `y`.
    Solo usa información estrictamente anterior a cada fecha (shift >= 1)."""
    dates, cols = wide.index, list(wide.columns)
    T, S = wide.shape
    shifted = wide.shift(1)
    feats: dict[str, np.ndarray] = {}
    for k in LAGS:
        feats[f"lag_{k}"] = wide.shift(k).to_numpy().ravel()
    for w in ROLLS:
        feats[f"media_{w}d"] = shifted.rolling(w, min_periods=w).mean().to_numpy().ravel()
    feats["media_mismo_dia_4s"] = ((wide.shift(7) + wide.shift(14) + wide.shift(21) + wide.shift(28)) / 4).to_numpy().ravel()

    # Nivel agregado (menos ruidoso que cada serie): total de la jurisdicción y total del grupo.
    jur_tot = wide.T.groupby(level=0).sum(min_count=1).T          # fecha x jurisdicción
    grp_tot = wide.T.groupby(level=1).sum(min_count=1).T          # fecha x grupo
    for w in (7, 28):
        jm = jur_tot.shift(1).rolling(w, min_periods=w).mean()
        gm = grp_tot.shift(1).rolling(w, min_periods=w).mean()
        feats[f"jur_total_media_{w}d"] = jm[[j for j, _ in cols]].to_numpy().ravel()
        feats[f"grp_total_media_{w}d"] = gm[[g for _, g in cols]].to_numpy().ravel()
    ju, gr = [j for j, _ in cols], [g for _, g in cols]
    feats["jur_total_lag1"] = jur_tot.shift(1)[ju].to_numpy().ravel()
    feats["jur_total_lag7"] = jur_tot.shift(7)[ju].to_numpy().ravel()
    feats["grp_total_lag1"] = grp_tot.shift(1)[gr].to_numpy().ravel()
    # Tendencia: nivel de la última semana frente al último mes.
    feats["ratio_7_28"] = feats["media_7d"] / (feats["media_28d"] + 0.05)
    feats["ratio_jur_7_28"] = feats["jur_total_media_7d"] / (feats["jur_total_media_28d"] + 0.05)

    cal = calendar_features(dates)
    cal_long = pd.DataFrame(np.repeat(cal.to_numpy(), S, axis=0), columns=cal.columns)

    jur_ids = sorted({j for j, _ in cols})
    grp_names = list(dict.fromkeys(g for _, g in cols))
    jur_of = np.tile([j for j, _ in cols], T)
    grp_of = np.tile([g for _, g in cols], T)
    onehot = {f"jur_{j}": (jur_of == j).astype(float) for j in jur_ids}
    onehot.update({f"grp_{g}": (grp_of == g).astype(float) for g in grp_names})

    out = pd.DataFrame({**feats, **onehot})
    out = pd.concat([out, cal_long], axis=1)
    out.insert(0, "fecha", np.repeat(dates.to_numpy(), S))
    out.insert(1, "jurisdiccion_id", jur_of)
    out.insert(2, "grupo", grp_of)
    out["y"] = wide.to_numpy().ravel()
    return out


META_COLS = ["fecha", "jurisdiccion_id", "grupo", "y"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filas completas (features y objetivo sin NaN): excluye huecos y ventanas que los cruzan."""
    return df.dropna(subset=feature_columns(df) + ["y"])
