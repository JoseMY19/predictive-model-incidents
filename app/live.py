"""Estado "en vivo": sincroniza con PostgreSQL, limpia lo que llega y mantiene pronóstico y zonas al día.

Un hilo de fondo (cada ~2 s):  1) limpia filas nuevas de public.incidencias (mismas reglas que el batch),
2) si hay datos nuevos o cambió el día, reconstruye panel + modelo espacial, 3) si cambió el último día completo,
recalcula el pronóstico (el modelo NO se reentrena: eso es un proceso programado aparte).
También mide, con lo que va llegando, si las incidencias reales cayeron en las zonas que se habían marcado.
"""
from __future__ import annotations

import json
import threading
import time
import traceback

import joblib
import numpy as np
import pandas as pd

from . import config, db
from .clean import clean_new
from .features import build_wide
from .modeling import predict
from .remote import RemoteSync
from .spatial import SpatialModel, cell_keys, select_top

FREEZE_DAYS = 7
SYNC_SECONDS = 2.0
REMOTE_SECONDS = 10.0          # cada cuánto se consulta la API de producción


RELOAD_MIN_AGE_S = 2.0        # a file touched less than this long ago may still be mid-write: wait for the next cycle


def artifact_stamp(path) -> tuple[int, int] | None:
    """(mtime_ns, size) of a file, or None if it does not exist. Used to detect that the retraining job swapped the model."""
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return st.st_mtime_ns, st.st_size


def is_stable(path, min_age_s: float = RELOAD_MIN_AGE_S, now: float | None = None) -> bool:
    try:
        return (now if now is not None else time.time()) - path.stat().st_mtime >= min_age_s
    except FileNotFoundError:
        return False


class Live:
    def __init__(self) -> None:
        self.snap: dict = {}
        self.version = 0
        self.last_error: str | None = None
        self.last_sync: pd.Timestamp | None = None
        self.arrivals = pd.DataFrame()
        self.frozen: dict | None = None
        self.baseline_ts = None
        self._sig = None
        self._sim = False
        self.remote = RemoteSync()
        self._remote_t = 0.0
        self._stop = threading.Event()
        self.reload_error: str | None = None
        self.model_loaded_at: pd.Timestamp | None = None
        self._force_forecast = False
        self._load_artifacts()

    # ------------------------------------------------------------------ model artifacts (hot reload)
    def _load_artifacts(self) -> None:
        """Load model, metrics, spatial parameters and catalog. Everything is read first and assigned together, so a
        failure halfway leaves the previous (consistent) set in place."""
        art = joblib.load(config.MODEL_FILE)
        catalog = json.loads(config.CATALOG_JSON.read_text(encoding="utf-8"))
        spatial = json.loads(config.SPATIAL_JSON.read_text(encoding="utf-8"))
        metrics = json.loads(config.METRICS_JSON.read_text(encoding="utf-8"))
        self._art, self._base_catalog, self.spatial, self.metrics = art, catalog, spatial, metrics
        self.params = spatial["parametros"]
        self._stamp = artifact_stamp(config.METRICS_JSON)
        self.model_loaded_at = pd.Timestamp.now(tz=config.TZ)

    def check_reload(self) -> bool:
        """If the retraining job replaced the model (metrics.json is swapped last), load the new set and force a rebuild of
        the snapshot and the forecast. Returns True when a reload happened."""
        stamp = artifact_stamp(config.METRICS_JSON)
        if stamp is None or stamp == self._stamp or not is_stable(config.METRICS_JSON):
            return False
        try:
            self._load_artifacts()
        except Exception as e:                       # keep serving the old model; do not retry until the files change again
            self._stamp = stamp
            self.reload_error = f"{type(e).__name__}: {e}"[:200]
            return False
        self.reload_error = None
        self._sig = None
        self._force_forecast = True
        return True

    # ------------------------------------------------------------------ construcción del estado
    def _rebuild(self, conn, asof: pd.Timestamp) -> None:
        last_full = asof.normalize() - pd.Timedelta(days=1)
        today = asof.normalize()
        where = "incluida and periodo_confiable and fecha <= %(d)s"
        panel = db.query_df(
            f"select fecha, jurisdiccion_id, grupo, count(*)::int n from analytics.incidencias_limpias where {where} group by 1,2,3",
            conn, params={"d": last_full.date()}, parse_dates=["fecha"])
        geo = db.query_df(
            f"select fecha, hora, jurisdiccion_id, grupo, lat, lon from analytics.incidencias_limpias where {where} and lat is not null",
            conn, params={"d": today.date()}, parse_dates=["fecha"])
        cat = dict(self._base_catalog)
        cat["ultima_fecha"] = last_full.strftime("%Y-%m-%d")
        cat["rangos_validos"] = [[s, e or cat["ultima_fecha"]] for s, e in config.VALID_RANGES]
        wide = build_wide(panel, cat)
        # El spatial usa solo hasta el último día completo (así se validó); lo de hoy va a la capa "recientes".
        sm = SpatialModel.build(geo[geo["fecha"] <= last_full], cat["grupos"], last_full)

        prev = self.snap
        if prev and prev["origin"] == last_full and not self._force_forecast:
            fc = prev["forecast"]                         # mismo día: el pronóstico no cambia (salvo modelo nuevo)
        else:
            self._force_forecast = False
            fc = predict(self._art["kind"], self._art["model"], self._art["cols"], wide, last_full, config.HORIZON_MAX)
        snap = {"catalog": cat, "wide": wide, "forecast": fc, "origin": last_full, "sm": sm, "as_of": asof,
                "total": int(panel["n"].sum())}
        if self.frozen is None:                           # primera construcción = "estado al corte": se congelan las zonas marcadas
            self._freeze(snap)
        self.snap = snap

    def _freeze(self, snap: dict) -> None:
        sm: SpatialModel = snap["sm"]
        fc = snap["forecast"].iloc[:FREEZE_DAYS]
        lam = {c: float(fc[c].sum()) for c in fc.columns}
        score = sm.expected(snap["origin"], lam, self.params).sum(axis=1)
        top10 = np.argsort(-score)[: int(0.10 * len(score))]
        pins = select_top(score, sm.lat, sm.lon, 20)
        self.frozen = {"desde": snap["origin"] + pd.Timedelta(days=1), "hasta": snap["origin"] + pd.Timedelta(days=FREEZE_DAYS),
                       "keys10": set(sm.keys[top10].tolist()), "keys_pins": set(sm.keys[pins].tolist()), "celdas": len(score)}
        with db.connect() as c, c.cursor() as cur:
            cur.execute("select max(created_at) from analytics.incidencias_limpias where incluida")
            self.baseline_ts = cur.fetchone()[0]

    # ------------------------------------------------------------------ llegadas en vivo
    def _load_arrivals(self, conn) -> None:
        if self.baseline_ts is None:
            return
        df = db.query_df(
            "select id, ts, fecha, hora, lat, lon, grupo, jurisdiccion_id, created_at from analytics.incidencias_limpias "
            "where incluida and created_at > %(t)s order by created_at", conn, params={"t": self.baseline_ts})
        if len(df) and self.frozen:
            k = np.where(df["lat"].notna(), cell_keys(df["lat"].fillna(0), df["lon"].fillna(0)), -1)
            df["en_top10"] = [int(x) in self.frozen["keys10"] for x in k]
            df["en_pin"] = [int(x) in self.frozen["keys_pins"] for x in k]
            f = pd.to_datetime(df["fecha"])
            df["en_ventana"] = (f >= self.frozen["desde"]) & (f <= self.frozen["hasta"])
        self.arrivals = df

    # ------------------------------------------------------------------ ciclo
    def step(self, conn) -> None:
        self.check_reload()
        if self.remote.enabled and time.monotonic() - self._remote_t >= REMOTE_SECONDS:
            self._remote_t = time.monotonic()
            self.remote.sync()                              # trae lo nuevo de la API a la base local (errores quedan en status)
        n = clean_new(conn)
        with conn.cursor() as cur:
            cur.execute("select to_regclass('sim.estado')")
            self._sim = cur.fetchone()[0] is not None
        asof = db.as_of(conn)
        with conn.cursor() as cur:
            cur.execute("select count(*) from analytics.incidencias_limpias where incluida")
            total = cur.fetchone()[0]
        sig = (total, asof.normalize())
        if sig != self._sig or not self.snap:
            self._rebuild(conn, asof)
            self._sig = sig
            self.version += 1
        if n:
            self._load_arrivals(conn)
        self.snap["as_of"] = asof
        self.last_sync = pd.Timestamp.now(tz=config.TZ)
        self.last_error = None

    def start(self) -> None:
        conn = db.connect()
        conn.autocommit = True
        self.step(conn)                                    # construcción inicial síncrona
        self._load_arrivals(conn)

        def loop():
            c = conn
            while not self._stop.wait(SYNC_SECONDS):
                try:
                    self.step(c)
                except Exception:                           # no tumbar el hilo: reintentar con conexión nueva
                    self.last_error = traceback.format_exc(limit=3)
                    try:
                        c.close()
                    except Exception:
                        pass
                    try:
                        c = db.connect(); c.autocommit = True
                    except Exception:
                        pass
        threading.Thread(target=loop, daemon=True, name="live-sync").start()

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ estado para la API
    def status(self) -> dict:
        a = self.arrivals
        out = {"modo": "simulado" if self._sim else "real", "version": self.version, "ahora": self.snap["as_of"].strftime("%Y-%m-%d %H:%M"),
               "ultimo_dia_completo": self.snap["origin"].strftime("%Y-%m-%d"), "total_incidencias_modelo": self.snap["total"],
               "ultima_sincronizacion": self.last_sync.strftime("%H:%M:%S") if self.last_sync is not None else None,
               "error": self.last_error, "llegadas": int(len(a)), "api": self.remote.status(),
               "modelo": {"entrenado": self.metrics.get("entrenado"), "datos_hasta": self.metrics.get("ultima_fecha_datos"),
                          "cargado": self.model_loaded_at.strftime("%Y-%m-%d %H:%M:%S") if self.model_loaded_at is not None else None,
                          "error_recarga": self.reload_error}}
        if self.frozen:
            w = a[a["en_ventana"] & a["lat"].notna()] if len(a) else a
            n = int(len(w))
            out["aciertos"] = {
                "ventana": [self.frozen["desde"].strftime("%Y-%m-%d"), self.frozen["hasta"].strftime("%Y-%m-%d")],
                "con_coordenada": n,
                "en_10pct_celdas": int(w["en_top10"].sum()) if n else 0, "pct_10pct": round(100 * float(w["en_top10"].mean()), 1) if n else None,
                "en_20_zonas": int(w["en_pin"].sum()) if n else 0, "pct_20_zonas": round(100 * float(w["en_pin"].mean()), 1) if n else None,
                "azar_10pct": 10.0, "azar_20_zonas": round(100 * 20 / self.frozen["celdas"], 1),
            }
        return out

    def recientes(self, limit: int = 400) -> list[dict]:
        a = self.arrivals
        if not len(a):
            return []
        a = a[a["lat"].notna()].tail(limit)
        return [{"id": int(r.id), "lat": round(float(r.lat), 6), "lon": round(float(r.lon), 6), "grupo": r.grupo,
                 "ts": pd.Timestamp(r.ts).tz_convert(config.TZ).strftime("%d/%m %H:%M") if pd.notna(r.ts) else None,
                 "jurisdiccion_id": int(r.jurisdiccion_id) if pd.notna(r.jurisdiccion_id) else None, "en_zona": bool(r.en_top10)} for r in a.itertuples()]
