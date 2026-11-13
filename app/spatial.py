"""Modelo espacial: reparte el pronóstico temporal (jurisdicción x tipo x día) entre celdas de ~200 m.

Idea:  Λ[celda, tipo] = λ[jurisdicción, tipo, día] x participación[celda | jurisdicción, tipo]
La participación sale del patrón RECIENTE de incidencias (conteos con decaimiento exponencial), mezclando el
patrón del tipo con el de todos los tipos (los tipos raros, p.ej. robos/hurtos, son muy ralos por sí solos).
Parámetros (semivida, mezcla) elegidos por backtest:  python -m app.spatial
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

CELL_M = 200
LAT0, LON0 = -12.05, -77.05
DLAT = CELL_M / 111_000
DLON = CELL_M / (111_000 * np.cos(np.radians(12)))
PLACEHOLDER_MIN_REPEATS = 100          # coordenada repetida >100 veces = punto por defecto, no un lugar real
BBOX = (-12.05, -11.90, -77.05, -76.93)  # lat_min, lat_max, lon_min, lon_max (SJL con margen)
TURNOS = [("madrugada (00-05 h)", 0, 5), ("mañana (06-11 h)", 6, 11), ("tarde (12-17 h)", 12, 17), ("noche (18-23 h)", 18, 23)]
DEFAULT_PARAMS = {"half_life": 21.0, "blend": 0.3}


def cell_keys(lat, lon) -> np.ndarray:
    r = np.floor((np.asarray(lat, float) - LAT0) / DLAT).astype(int)
    c = np.floor((np.asarray(lon, float) - LON0) / DLON).astype(int)
    return r * 100_000 + c


def clean_geo(geo: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n0 = len(geo)
    rep = geo.groupby(["lat", "lon"])["fecha"].transform("size")
    g = geo[rep <= PLACEHOLDER_MIN_REPEATS]
    n1 = len(g)
    g = g[g["lat"].between(BBOX[0], BBOX[1]) & g["lon"].between(BBOX[2], BBOX[3])]
    return g.reset_index(drop=True), {"total": n0, "punto_por_defecto": n0 - n1, "fuera_de_sjl": n1 - len(g), "usadas": len(g)}


@dataclass
class SpatialModel:
    days: pd.DatetimeIndex
    grupos: list[str]
    M: np.ndarray                 # (días, celdas, grupos) conteos
    lat: np.ndarray               # centroide real de las incidencias de cada celda
    lon: np.ndarray
    jur: np.ndarray               # jurisdicción de cada celda (moda)
    turno: np.ndarray             # (celdas, 4) incidencias con hora real por turno
    turno_jur: dict               # jurisdicción -> (4,)
    stats: dict
    keys: np.ndarray = None       # clave cruda de cada celda (fila*100000+columna), estable entre reconstrucciones

    @classmethod
    def build(cls, geo: pd.DataFrame, grupos: list[str], last: pd.Timestamp) -> "SpatialModel":
        g, stats = clean_geo(geo)
        days = pd.date_range("2025-01-01", last)
        g = g[g["fecha"] <= last]
        r = np.floor((g["lat"].to_numpy() - LAT0) / DLAT).astype(int)
        c = np.floor((g["lon"].to_numpy() - LON0) / DLON).astype(int)
        cell_raw = r * 100_000 + c
        uniq, cidx = np.unique(cell_raw, return_inverse=True)
        n = len(uniq)
        di = (g["fecha"] - days[0]).dt.days.to_numpy()
        gi = g["grupo"].map({k: i for i, k in enumerate(grupos)}).to_numpy()
        M = np.zeros((len(days), n, len(grupos)), np.float32)
        np.add.at(M, (di, cidx, gi), 1)
        lat = pd.Series(g["lat"].to_numpy()).groupby(cidx).mean().to_numpy()
        lon = pd.Series(g["lon"].to_numpy()).groupby(cidx).mean().to_numpy()
        jur = pd.Series(g["jurisdiccion_id"].to_numpy()).groupby(cidx).agg(lambda s: s.mode().iloc[0]).to_numpy()
        # turnos con hora real
        T = np.zeros((n, len(TURNOS)), np.float32)
        h = g["hora"].to_numpy()
        ok = ~np.isnan(h)
        for k, (_, a, b) in enumerate(TURNOS):
            sel = ok & (h >= a) & (h <= b)
            np.add.at(T[:, k], cidx[sel], 1)
        turno_jur = {int(j): T[jur == j].sum(axis=0) for j in np.unique(jur)}
        stats.update(celdas=n, celda_m=CELL_M, con_hora_real=int(ok.sum()))
        return cls(days, grupos, M, lat, lon, jur, T, turno_jur, stats, uniq)

    # ---- participación espacial ------------------------------------------------------------
    def shares(self, origin: pd.Timestamp, half_life: float | None, blend: float) -> np.ndarray:
        """(celdas, grupos): fracción de cada (jurisdicción, tipo) que corresponde a cada celda."""
        oi = (origin - self.days[0]).days
        age = oi - np.arange(oi + 1)
        w = 0.5 ** (age / half_life) if half_life else np.ones(oi + 1)
        S = np.tensordot(w, self.M[: oi + 1], axes=(0, 0))          # (celdas, grupos)
        out = np.zeros_like(S)
        for j in np.unique(self.jur):
            idx = np.where(self.jur == j)[0]
            s_all = S[idx].sum(axis=1)
            p_all = s_all / s_all.sum() if s_all.sum() > 0 else np.full(len(idx), 1 / len(idx))
            for k in range(S.shape[1]):
                sg = S[idx, k]
                p_g = sg / sg.sum() if sg.sum() > 0 else p_all
                out[idx, k] = (1 - blend) * p_all + blend * p_g
        return out

    def expected(self, origin: pd.Timestamp, lam_window: dict[tuple[int, str], float], params: dict) -> np.ndarray:
        """Λ (celdas, grupos) para una ventana, dado el total esperado por (jurisdicción, tipo) en ella."""
        sh = self.shares(origin, params["half_life"], params["blend"])
        L = np.zeros_like(sh)
        for (j, g), v in lam_window.items():
            k = self.grupos.index(g)
            m = self.jur == j
            L[m, k] = v * sh[m, k]
        return L

    # ---- perfil horario ----------------------------------------------------------------------
    def turno_probable(self, i: int) -> tuple[str, float] | None:
        prior = self.turno_jur.get(int(self.jur[i]))
        if prior is None or prior.sum() == 0:
            return None
        p = self.turno[i] + 8 * prior / prior.sum()                 # suavizado hacia el perfil de la jurisdicción
        k = int(np.argmax(p))
        return TURNOS[k][0], float(p[k] / p.sum())


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12_742_000 * np.arcsin(np.sqrt(a))


def select_top(score: np.ndarray, lat: np.ndarray, lon: np.ndarray, k: int, min_dist_m: float = 350) -> list[int]:
    """Los k mejores puntos, evitando duplicados vecinos (supresión de no-máximos)."""
    chosen: list[int] = []
    for i in np.argsort(-score):
        if score[i] <= 0 or len(chosen) >= k:
            break
        if not chosen or haversine_m(lat[i], lon[i], lat[chosen], lon[chosen]).min() >= min_dist_m:
            chosen.append(int(i))
    return chosen


def load_cameras() -> list[dict]:
    """Cámaras conocidas: las municipales reales (geo/camaras_municipales.json, se genera con `python -m app.camaras_remote`),
    (como respaldo, si no hay municipales, la capa 'Coordenadas nuevas') y, opcionalmente, CAMARAS_JSON (lista de objetos con latitud/longitud, o
    {clave: [..]}). Un punto a ≤ 10 m de uno de OTRA fuente se considera la misma cámara y no se repite (varias cámaras en el mismo sitio de una misma fuente sí se conservan)."""
    # Si no hay municipales importadas se usa, como respaldo, la capa «Coordenadas nuevas» (34 puntos).
    files = [config.GEO_DIR / "camaras_municipales.json"]
    if not files[0].exists():
        files.append(config.GEO_DIR / "camaras_nuevas.json")
    if os.environ.get("CAMARAS_JSON"):
        files.append(Path(os.environ["CAMARAS_JSON"]))
    out: list[dict] = []
    for f in files:
        if not f.exists():
            continue
        previas = list(out)                          # solo se comparan contra cámaras de OTRAS fuentes
        data = json.loads(f.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else next(v for v in data.values() if isinstance(v, list))
        for i, r in enumerate(rows):
            la = r.get("latitud", r.get("latitude", r.get("lat")))
            lo = r.get("longitud", r.get("longitude", r.get("lng", r.get("lon"))))
            if la is None or lo is None:
                continue
            la, lo = float(la), float(lo)
            if previas and haversine_m(la, lo, np.array([c["lat"] for c in previas]), np.array([c["lon"] for c in previas])).min() <= 10:
                continue
            cam = {"id": r.get("nombre") or f"{f.stem}-{i + 1}", "lat": la, "lon": lo, "fuente": f.stem}
            for k in ("tipo", "oculta"):
                if r.get(k) is not None:
                    cam[k] = r[k]
            out.append(cam)
    return out
