"""API REST (FastAPI).  Ejecutar:  uvicorn app.api:app --port 8000"""
from __future__ import annotations

import json
import unicodedata
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .live import Live
from .spatial import SpatialModel, haversine_m, load_cameras, select_top

state: dict = {}

GRUPO_LABEL = {
    "PATRIMONIO": "robos y hurtos (patrimonio)", "PERSONA": "agresiones / contra la persona",
    "ORDENANZAS": "infracciones a ordenanzas municipales", "AUXILIO": "auxilio y rescate",
    "ALCOHOL_COSTUMBRES": "consumo de alcohol / buenas costumbres", "TRANQUILIDAD": "alteración de tranquilidad y orden",
    "TRANSITO": "accidentes e infracciones de tránsito", "OTROS": "otros incidentes",
}
CAMERA_NEAR_M = 200       # a esta distancia o menos se considera "cámara en la zona"
SPATIAL_META_KEYS = ("parametros", "celdas", "celda_m", "horizonte_dias", "origenes", "azar", "elegido", "baseline_estatico")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().upper().strip()
    return s[3:] if s.startswith("LA ") else s


@asynccontextmanager
async def lifespan(_: FastAPI):
    needed = (config.CATALOG_JSON, config.MODEL_FILE, config.METRICS_JSON, config.SPATIAL_JSON)
    for f in needed:
        if not f.exists():
            raise RuntimeError(f"Falta {f}. Ejecuta: python -m app.restore/clean/etl, python -m app.train, python -m app.spatial_eval")
    live = Live()
    live.start()                                        # sincroniza con PostgreSQL y construye el estado inicial
    catalog = live.snap["catalog"]
    jur_by_norm = {_norm(j["nombre"]): j for j in catalog["jurisdicciones"]}
    gj = json.loads((config.GEO_DIR / "juridiccion.geojson").read_text(encoding="utf-8"))
    for f in gj["features"]:
        j = jur_by_norm.get(_norm(f["properties"]["name"]))
        f["properties"] = {"nombre": f["properties"]["name"], "jurisdiccion_id": j["id"] if j else None}
    state.update(live=live, spatial=json.loads(config.SPATIAL_JSON.read_text(encoding="utf-8")), cameras=load_cameras(),
                 poligonos=gj, jur_name={j["id"]: j["nombre"] for j in catalog["jurisdicciones"]},
                 metrics=json.loads(config.METRICS_JSON.read_text(encoding="utf-8")))
    yield
    live.stop()
    state.clear()


app = FastAPI(title="Predicción de incidencias", version="0.3.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["GET"], allow_headers=["*"])


def snap() -> dict:
    return state["live"].snap


@app.get("/api/health")
def health():
    return {"status": "ok", "ultima_fecha": snap()["origin"].strftime("%Y-%m-%d"), "version": state["live"].version}


@app.get("/api/meta")
def meta():
    m = state["metrics"]
    bt = m["backtest"]
    res = bt["resultados"]
    s = snap()
    return {
        "catalogo": {**s["catalog"], "ultima_fecha": s["origin"].strftime("%Y-%m-%d")},
        "horizonte_max": config.HORIZON_MAX,
        "modelo": {
            "nombre": m["modelo_nombre"], "entrenado": m["entrenado"], "filas": m["filas_entrenamiento"],
            "backtest_desde": bt["desde"], "backtest_hasta": bt["hasta"],
            "descripcion_backtest": bt["descripcion"],
            "mejora_vs_baseline_pct": m.get("mejora_vs_baseline_pct"),
            "resultados": [
                {"id": k, "nombre": v["nombre"], "mae_jurisdiccion_dia": v["mae_jurisdiccion"],
                 "rmse_jurisdiccion_dia": v["rmse_jurisdiccion"],
                 "wape_dia_pct": v["wape_jurisdiccion_pct"], "wape_semana_pct": v["wape_semana_jurisdiccion_pct"]}
                for k, v in res.items()],
            "importancias": m["importancias"],
        },
        "espacial": {k: state["spatial"][k] for k in SPATIAL_META_KEYS},
    }


@app.get("/api/history")
def history(days: int = Query(90, ge=7, le=400)):
    """Conteos diarios reales por jurisdicción y grupo de los últimos `days` días con dato fiable."""
    wide: pd.DataFrame = snap()["wide"].dropna(how="all").iloc[-days:]
    rows = [{"fecha": d.strftime("%Y-%m-%d"), "jurisdiccion_id": int(j), "grupo": g, "n": int(v)}
            for d, r in wide.iterrows() for (j, g), v in r.items()]
    return {"rows": rows}


@app.get("/api/forecast")
def forecast(horizon: int = Query(14, ge=1, le=config.HORIZON_MAX)):
    """Pronóstico (esperanza de incidencias/día) por jurisdicción y grupo para los próximos `horizon` días."""
    s = snap()
    fc: pd.DataFrame = s["forecast"].iloc[:horizon]
    rows = [{"fecha": d.strftime("%Y-%m-%d"), "jurisdiccion_id": int(j), "grupo": g, "prediccion": round(float(v), 2)}
            for d, r in fc.iterrows() for (j, g), v in r.items()]
    return {"origen": s["origin"].strftime("%Y-%m-%d"), "horizonte": horizon, "rows": rows, "version": state["live"].version}


@app.get("/api/capas")
def capas():
    """Polígonos de jurisdicción y cámaras conocidas, para dibujar el mapa."""
    return {"jurisdicciones": state["poligonos"], "camaras": state["cameras"],
            "camaras_nota": "Solo cámaras de la capa 'Coordenadas nuevas' (y CAMARAS_JSON si se configuró)."}


@app.get("/api/live/status")
def live_status():
    """Estado de la sincronización con la base y verificación en vivo de las zonas marcadas."""
    return state["live"].status()


@app.get("/api/live/recientes")
def live_recientes(limit: int = Query(400, ge=1, le=3000)):
    """Incidencias llegadas desde que arrancó la API (con coordenada) y si cayeron en una zona marcada."""
    return {"rows": state["live"].recientes(limit), "version": state["live"].version}


@app.get("/api/hotspots")
def hotspots(dias: int = Query(7, ge=1, le=config.HORIZON_MAX), grupo: str = "TODOS",
             top: int = Query(20, ge=1, le=60), jurisdiccion_id: int | None = None):
    """Zonas (celdas de ~200 m) con mayor incidencia esperada en los próximos `dias` días."""
    s = snap()
    sm: SpatialModel = s["sm"]
    catalog = s["catalog"]
    if grupo != "TODOS" and grupo not in catalog["grupos"]:
        raise HTTPException(422, f"grupo inválido: {grupo}")
    origin: pd.Timestamp = s["origin"]
    fc: pd.DataFrame = s["forecast"].iloc[:dias]
    lam = {c: float(fc[c].sum()) for c in fc.columns}
    L = sm.expected(origin, lam, state["spatial"]["parametros"])           # (celdas, grupos)
    score_all = L.sum(axis=1) if grupo == "TODOS" else L[:, sm.grupos.index(grupo)]
    # Nivel de riesgo relativo al resto de las celdas (para el mismo filtro de tipo).
    # Si se filtra por jurisdicción, el nivel es relativo a las celdas de ESA jurisdicción (no a todo SJL).
    scope = score_all[(sm.jur == jurisdiccion_id)] if jurisdiccion_id else score_all
    pos = scope[scope > 0]
    p97, p90 = (np.percentile(pos, 97), np.percentile(pos, 90)) if len(pos) else (np.inf, np.inf)
    score = np.where(sm.jur == jurisdiccion_id, score_all, 0) if jurisdiccion_id else score_all
    cams = state["cameras"]
    cam_lat = np.array([c["lat"] for c in cams]); cam_lon = np.array([c["lon"] for c in cams])

    out = []
    for i in select_top(score, sm.lat, sm.lon, top):
        order = [int(k) for k in np.argsort(-L[i])]
        if grupo != "TODOS":                      # con filtro, el tipo filtrado va siempre primero
            kf = sm.grupos.index(grupo)
            order = [kf] + [k for k in order if k != kf]
        tot = L[i].sum()
        tipos = [{"grupo": sm.grupos[k], "etiqueta": GRUPO_LABEL[sm.grupos[k]], "esperadas": round(float(L[i, k]), 2),
                  "pct": round(float(100 * L[i, k] / tot), 1)} for k in order[:3] if L[i, k] > 0]
        cam = None
        if len(cams):
            d = haversine_m(sm.lat[i], sm.lon[i], cam_lat, cam_lon)
            j = int(np.argmin(d))
            cam = {"id": cams[j]["id"], "lat": cams[j]["lat"], "lon": cams[j]["lon"], "distancia_m": int(round(d[j]))}
        principal = tipos[0]["etiqueta"] if tipos else "incidencias"
        if cam and cam["distancia_m"] <= CAMERA_NEAR_M:
            accion = f"Verificar la cámara cercana (a {cam['distancia_m']} m) y reforzar la vigilancia de la zona."
        elif cam:
            accion = f"Verificar la zona con patrullaje o serenazgo (cámara conocida más cercana a {cam['distancia_m']} m)."
        else:
            accion = "Verificar la zona con patrullaje o serenazgo."
        tp = sm.turno_probable(i)
        sc = float(score_all[i])
        out.append({
            "id": f"z{int(sm.keys[i])}", "lat": round(float(sm.lat[i]), 6), "lon": round(float(sm.lon[i]), 6),
            "jurisdiccion_id": int(sm.jur[i]), "jurisdiccion": state["jur_name"].get(int(sm.jur[i])),
            "esperadas": round(sc, 2), "prob_alguna": round(float(1 - np.exp(-sc)), 3),
            "nivel": "alto" if sc >= p97 else "medio" if sc >= p90 else "bajo",
            "tipos": tipos, "turno": {"nombre": tp[0], "pct": round(100 * tp[1], 0)} if tp else None,
            "camara": cam, "titulo": "Posible incidencia",
            "mensaje": f"Tipo más probable: {principal}. {accion}",
        })
    sp = state["spatial"]
    return {
        "origen": origin.strftime("%Y-%m-%d"),
        "desde": (origin + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "hasta": (origin + pd.Timedelta(days=dias)).strftime("%Y-%m-%d"),
        "dias": dias, "grupo": grupo, "radio_m": 100, "rows": out, "pedidas": top, "version": state["live"].version,
        "calidad": {"top20_puntos_todos": sp["elegido"]["todos_top20_puntos"], "top5pct_celdas_todos": sp["elegido"]["todos_top5pct"],
                    "top10pct_celdas_patrimonio": sp["elegido"]["patrimonio_top10pct"]},
    }
