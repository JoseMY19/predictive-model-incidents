"""ETL: base limpia (analytics.incidencias_limpias) -> panel diario y puntos geográficos para los modelos.

Uso:   python -m app.etl
Flujo completo desde un backup:
    python -m app.restore --dump <backup>   # carga el origen en PostgreSQL (public.*)
    python -m app.clean                     # limpieza auditable -> analytics.*
    python -m app.etl                       # panel/puntos para entrenar
Solo se usan filas `incluida` y dentro de un período confiable; el último día (incompleto) se descarta.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config, db


def find_pg_restore() -> str:
    """Localiza pg_restore: variable PG_RESTORE, PATH o instalación típica de Windows."""
    env = os.environ.get("PG_RESTORE")
    if env and Path(env).exists():
        return env
    found = shutil.which("pg_restore")
    if found:
        return found
    candidates = sorted(glob.glob(r"C:\Program Files\PostgreSQL\*\bin\pg_restore.exe"), reverse=True)
    if candidates:
        return candidates[0]
    raise SystemExit("No se encontró pg_restore. Instala PostgreSQL client tools o define PG_RESTORE.")


def last_complete_day(conn) -> pd.Timestamp:
    """Último día completo = el día anterior a "ahora" (ese día aún está en curso)."""
    return db.as_of(conn).normalize() - pd.Timedelta(days=1)


def build(conn, last_full: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    where = "incluida and periodo_confiable and fecha <= %(last)s"
    panel = db.query_df(
        f"select fecha, jurisdiccion_id, grupo, count(*)::int n from analytics.incidencias_limpias where {where} "
        "group by 1,2,3 order by 1,2,3", conn, params={"last": last_full.date()}, parse_dates=["fecha"])
    geo = db.query_df(
        f"select fecha, hora, jurisdiccion_id, grupo, lat, lon from analytics.incidencias_limpias where {where} and lat is not null",
        conn, params={"last": last_full.date()}, parse_dates=["fecha"])
    jur = db.query_df("select id, codigo, nombre from public.jurisdicciones where habilitado order by codigo", conn)
    quality = db.query_df("select regla, filas from analytics.resumen_calidad order by filas desc", conn)
    return panel, geo, {"jur": jur, "calidad": quality}


def run() -> None:
    conn = db.connect()
    last_full = last_complete_day(conn)
    panel, geo, extra = build(conn, last_full)
    conn.close()
    config.DATA_DIR.mkdir(exist_ok=True)
    panel.to_csv(config.PANEL_CSV, index=False)
    geo.to_csv(config.GEO_CSV, index=False)
    catalog = {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fuente": "postgresql:analytics.incidencias_limpias",
        "unidad_id": config.UNIDAD_ID,
        "rangos_validos": [[s, e or last_full.strftime("%Y-%m-%d")] for s, e in config.VALID_RANGES],
        "ultima_fecha": last_full.strftime("%Y-%m-%d"),
        "jurisdicciones": [{"id": int(r.id), "codigo": r.codigo, "nombre": r.nombre} for r in extra["jur"].itertuples()],
        "grupos": config.GRUPO_NAMES,
        "total_incidencias": int(panel["n"].sum()),
        "calidad": {r.regla: int(r.filas) for r in extra["calidad"].itertuples()},
    }
    config.CATALOG_JSON.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Panel: {len(panel)} filas, {panel['n'].sum()} incidencias, {panel['fecha'].min().date()} -> {panel['fecha'].max().date()}")
    print(f"Con coordenadas: {len(geo)} ({len(geo)/panel['n'].sum():.1%}) | con hora real: {int(geo['hora'].notna().sum())}")


if __name__ == "__main__":
    run()
