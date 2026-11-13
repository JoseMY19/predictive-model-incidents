"""Limpieza de datos auditable: public.incidencias (origen, intacto) -> analytics.incidencias_limpias.

Cada fila conserva su id de origen, si fue incluida o excluida y por qué, y qué se corrigió.
Uso:   python -m app.clean          (reconstruye todo)
       desde código: clean_new(conn)   (incremental, para datos que llegan en tiempo real)
"""
from __future__ import annotations

import json
import unicodedata
from datetime import date

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values

from . import config, db
from .spatial import BBOX

PLACEHOLDER_MIN_REPEATS = 100        # misma coordenada exacta > N veces = punto por defecto
REMAP_BEFORE = date(2026, 1, 1)      # la permutación de jurisdicciones afecta solo a la carga histórica de 2025
REMAP_MIN_SHARE = 0.75               # una jurisdicción se remapea solo si >=75 % de sus puntos caen en un mismo polígono
DUP_WINDOW_S = 120                   # doble envío: misma incidencia registrada dos veces en <= 2 min

RULES = {
    "UNIDAD_NO_SERENAZGO": ("Unidad distinta de Serenazgo (Policía casi no se registra desde mar-2026)", "excluir"),
    "SIN_JURISDICCION_O_FECHA": ("Sin jurisdicción o sin fecha de ocurrencia", "excluir"),
    "FECHA_INVALIDA": ("Fecha absurda (año < 2024-12 o posterior al registro) sin fecha de registro confiable", "excluir"),
    "DUPLICADO_DOBLE_ENVIO": ("Misma incidencia registrada 2 veces en <=2 min (sistema nuevo)", "excluir copias"),
    "DUPLICADO_CARGA_HISTORICA": ("Duplicado exacto (incl. descripción) en la carga histórica", "excluir copias"),
    "FECHA_CORREGIDA": ("Fecha de ocurrencia futura/absurda: se usa la de registro (sistema nuevo)", "corregir"),
    "JURISDICCION_REMAPEADA": ("Jurisdicción de la carga 2025 corregida (permutación de IDs)", "corregir"),
    "TIPO_CORREGIDO": ("Subtipo no pertenece al tipo declarado: se usa el tipo del subtipo", "corregir"),
    "COORD_POR_DEFECTO": ("Coordenada exacta repetida >100 veces (punto por defecto)", "anular coordenada"),
    "COORD_FUERA_DE_AREA": ("Coordenada fuera de San Juan de Lurigancho o inválida", "anular coordenada"),
}


def norm(s: str) -> str:
    t = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().upper().strip()
    return t[3:] if t.startswith("LA ") else t


def load_polygons(conn) -> list[tuple[int, np.ndarray]]:
    jur = db.query_df("select id, nombre from public.jurisdicciones", conn)
    ids = {norm(r.nombre): int(r.id) for r in jur.itertuples()}
    gj = json.loads((config.GEO_DIR / "juridiccion.geojson").read_text(encoding="utf-8"))
    return [(ids[norm(f["properties"]["name"])], np.array(f["geometry"]["coordinates"][0])) for f in gj["features"]]


def _inside(x: np.ndarray, y: np.ndarray, ring: np.ndarray) -> np.ndarray:
    px, py = ring[:, 0], ring[:, 1]
    ins = np.zeros(len(x), bool)
    j = len(px) - 1
    for i in range(len(px)):
        ins ^= ((py[i] > y) != (py[j] > y)) & (x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i] + 1e-15) + px[i])
        j = i
    return ins


def polygon_of(lat: np.ndarray, lon: np.ndarray, polys) -> np.ndarray:
    out = np.zeros(len(lat), int)
    ok = ~(np.isnan(lat) | np.isnan(lon))
    for jid, ring in polys:
        m = np.zeros(len(lat), bool)
        m[ok] = _inside(lon[ok], lat[ok], ring)
        out[m] = jid
    return out


RAW_SQL = """
select i.id, i."ocurridoEn" ocu, i."registradoEn" reg, i."createdAt" cre, i."unidadId" unidad, i."tipoCasoId" tipo,
       i."subTipoCasoId" subtipo, s."tipoCasoId" tipo_del_subtipo, i."jurisdiccionId" jur, i.latitud lat, i.longitud lon,
       md5(coalesce(i.descripcion,'')) dh
from public.incidencias i left join public.sub_tipo_casos s on s.id = i."subTipoCasoId"
"""


def apply_rules(raw: pd.DataFrame, polys, remap: dict[int, int], placeholders: set[tuple[float, float]],
                known_keys: set | None = None) -> pd.DataFrame:
    """Aplica las reglas a un lote. `known_keys`: claves de duplicado ya existentes (modo incremental)."""
    df = raw.copy()
    for c in ("ocu", "reg", "cre"):
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    flags: list[list[str]] = [[] for _ in range(len(df))]
    motivo = np.array([None] * len(df), dtype=object)

    def mark(mask, code, exclude=False):
        for i in np.where(mask.to_numpy() if hasattr(mask, "to_numpy") else mask)[0]:
            flags[i].append(code)
            if exclude and motivo[i] is None:
                motivo[i] = code

    legacy = (df["reg"].isna() | (df["reg"] == df["ocu"])).to_numpy()
    cre_date = df["cre"].dt.tz_convert(config.TZ).dt.normalize().dt.tz_localize(None)

    mark(df["unidad"] != config.UNIDAD_ID, "UNIDAD_NO_SERENAZGO", exclude=True)
    mark(df["jur"].isna() | df["ocu"].isna(), "SIN_JURISDICCION_O_FECHA", exclude=True)

    # --- fecha efectiva ---
    ts = df["ocu"].copy()
    bad_future = (df["ocu"] > df["reg"] + pd.Timedelta(days=1)) | (df["ocu"].dt.year > 2100) | (df["ocu"] > df["cre"] + pd.Timedelta(days=1))
    bad_old = df["ocu"] < pd.Timestamp("2024-12-01", tz="UTC")
    fixable = (bad_future | bad_old) & ~pd.Series(legacy, index=df.index) & df["reg"].notna()
    ts[fixable] = df.loc[fixable, "reg"]
    mark(fixable, "FECHA_CORREGIDA")
    mark(((bad_future | bad_old) & ~fixable) & df["ocu"].notna(), "FECHA_INVALIDA", exclude=True)
    local = ts.dt.tz_convert(config.TZ).dt.tz_localize(None)
    df["fecha"] = local.dt.normalize()
    df["ts"] = ts
    df["hora"] = local.dt.hour.where(~pd.Series(legacy, index=df.index))      # hora real solo en el sistema nuevo

    # --- coordenadas ---
    lat = np.array(pd.to_numeric(df["lat"], errors="coerce"), dtype=float)
    lon = np.array(pd.to_numeric(df["lon"], errors="coerce"), dtype=float)
    is_ph = np.array([(round(a, 6), round(b, 6)) in placeholders for a, b in zip(lat, lon)])
    out_area = ~np.isnan(lat) & ~((lat >= BBOX[0]) & (lat <= BBOX[1]) & (lon >= BBOX[2]) & (lon <= BBOX[3]))
    mark(is_ph, "COORD_POR_DEFECTO")
    mark(out_area | (np.isnan(lat) != np.isnan(lon)), "COORD_FUERA_DE_AREA")
    bad_coord = is_ph | out_area | np.isnan(lat) | np.isnan(lon)
    lat[bad_coord] = np.nan
    lon[bad_coord] = np.nan
    df["lat"], df["lon"] = lat, lon

    # --- jurisdicción ---
    poly = polygon_of(lat, lon, polys)
    jur = df["jur"].astype("float").to_numpy()
    jur_orig = jur.copy()
    need_fix = legacy & (df["fecha"].dt.date < REMAP_BEFORE).to_numpy()
    mapped = np.array([remap.get(int(j), int(j)) if not np.isnan(j) else np.nan for j in jur])
    new = np.where(need_fix, np.where(poly > 0, poly, mapped), jur)
    changed = need_fix & ~np.isnan(jur) & (new != jur)
    mark(changed, "JURISDICCION_REMAPEADA")
    df["jurisdiccion_original"], df["jurisdiccion_id"] = jur_orig, new

    # --- tipo ---
    tdecl = df["tipo"].astype("float")
    tsub = df["tipo_del_subtipo"].astype("float")
    fix_t = tsub.notna() & (tsub != tdecl)
    mark(fix_t, "TIPO_CORREGIDO")
    df["tipo_caso_id"] = np.where(fix_t, tsub, tdecl)
    df["grupo"] = pd.Series(df["tipo_caso_id"]).map(lambda t: config.TIPO_TO_GRUPO.get(int(t), config.GRUPO_OTROS) if pd.notna(t) else config.GRUPO_OTROS).values

    # --- duplicados (solo sobre filas aún incluidas) ---
    alive = motivo == None  # noqa: E711
    d = df[alive].copy()
    d["legacy"] = legacy[alive]
    d["minuto"] = d["ts"].dt.floor("min")
    d["k_new"] = list(zip(d["minuto"], d["lat"].round(6), d["lon"].round(6), d["tipo_caso_id"], d["jurisdiccion_id"]))
    d["k_old"] = list(zip(d["ocu"], d["lat"].round(6), d["lon"].round(6), d["tipo_caso_id"], d["subtipo"], d["jurisdiccion_id"], d["dh"]))
    d = d.sort_values("id")
    pos = {idx: i for i, idx in enumerate(df.index)}
    live = d[~d["legacy"]]
    first = live.groupby("k_new")["cre"].transform("min")
    dup_live = live[(live["cre"] - first).dt.total_seconds().between(0, DUP_WINDOW_S) & live.duplicated("k_new", keep="first")]
    dup_old = d[d["legacy"] & d.duplicated("k_old", keep="first")]
    if known_keys:
        dup_live = pd.concat([dup_live, live[live["k_new"].isin(known_keys) & ~live.index.isin(dup_live.index)]])
    for idx in dup_live.index:
        flags[pos[idx]].append("DUPLICADO_DOBLE_ENVIO"); motivo[pos[idx]] = "DUPLICADO_DOBLE_ENVIO"
    for idx in dup_old.index:
        flags[pos[idx]].append("DUPLICADO_CARGA_HISTORICA"); motivo[pos[idx]] = "DUPLICADO_CARGA_HISTORICA"

    df["incluida"] = motivo == None  # noqa: E711
    df["motivo_exclusion"] = motivo
    df["flags"] = flags
    df["periodo_confiable"] = False
    for start, end in config.VALID_RANGES:
        m = df["fecha"] >= pd.Timestamp(start)
        if end:
            m &= df["fecha"] <= pd.Timestamp(end)
        df["periodo_confiable"] |= m
    return df


DDL = """
drop schema if exists analytics cascade;
create schema analytics;
create table analytics.incidencias_limpias (
  id integer primary key, fecha date, hora smallint, jurisdiccion_id integer, jurisdiccion_original integer,
  tipo_caso_id integer, grupo text, lat double precision, lon double precision, created_at timestamptz, ts timestamptz,
  incluida boolean not null, motivo_exclusion text, periodo_confiable boolean not null, flags text[] not null default '{}',
  dup_key_new text);
create index on analytics.incidencias_limpias (fecha) where incluida;
create table analytics.reglas_calidad (regla text primary key, descripcion text, accion text);
create table analytics.jur_remap (jurisdiccion_declarada integer primary key, jurisdiccion_real integer, participacion numeric, filas integer);
create table analytics.coord_por_defecto (lat double precision, lon double precision, repeticiones integer);
create table analytics.resumen_calidad (regla text, filas integer);
"""
COLS = ["id", "fecha", "hora", "jurisdiccion_id", "jurisdiccion_original", "tipo_caso_id", "grupo", "lat", "lon", "cre", "ts",
        "incluida", "motivo_exclusion", "periodo_confiable", "flags"]


def _py(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    return v


def insert(conn, df: pd.DataFrame) -> None:
    rows = []
    for r in df[COLS].itertuples(index=False):
        v = [_py(x) for x in r]
        v[1] = v[1].date() if v[1] is not None else None            # fecha
        v[2] = int(v[2]) if v[2] is not None else None              # hora
        for k in (3, 4, 5):
            v[k] = int(v[k]) if v[k] is not None else None
        rows.append(tuple(v))
    with conn.cursor() as cur:
        execute_values(cur, f"insert into analytics.incidencias_limpias (id, fecha, hora, jurisdiccion_id, jurisdiccion_original, tipo_caso_id, "
                            f"grupo, lat, lon, created_at, ts, incluida, motivo_exclusion, periodo_confiable, flags) values %s on conflict (id) do nothing",
                       rows, page_size=5000)
    conn.commit()


def derive_remap(raw: pd.DataFrame, polys) -> tuple[dict[int, int], pd.DataFrame]:
    """Aprende la permutación de jurisdicciones comparando etiqueta declarada vs polígono, en la carga histórica de 2025."""
    r = raw[raw["reg"].isna() | (pd.to_datetime(raw["reg"], utc=True) == pd.to_datetime(raw["ocu"], utc=True))].copy()
    r["f"] = pd.to_datetime(r["ocu"], utc=True, errors="coerce").dt.tz_convert(config.TZ).dt.tz_localize(None)
    r = r[(r["f"] >= "2025-01-01") & (r["f"] < "2026-01-01") & (r["unidad"] == config.UNIDAD_ID)]
    rep = r.groupby(["lat", "lon"])["id"].transform("size")
    r = r[(rep <= PLACEHOLDER_MIN_REPEATS) & r["lat"].notna()]
    r["poly"] = polygon_of(r["lat"].astype(float).to_numpy(), r["lon"].astype(float).to_numpy(), polys)
    r = r[r["poly"] > 0]
    out, rows = {}, []
    for jd, g in r.groupby("jur"):
        vc = g["poly"].value_counts()
        share = vc.iloc[0] / len(g)
        rows.append((int(jd), int(vc.index[0]), round(float(share), 3), int(len(g))))
        if share >= REMAP_MIN_SHARE:
            out[int(jd)] = int(vc.index[0])
    return out, pd.DataFrame(rows, columns=["declarada", "real", "participacion", "filas"])


def build_all() -> None:
    conn = db.connect()
    polys = load_polygons(conn)
    raw = db.query_df(RAW_SQL, conn)
    print(f"origen: {len(raw)} filas")
    remap, tbl = derive_remap(raw, polys)
    print("permutación de jurisdicciones aprendida:\n", tbl.to_string(index=False))
    la = pd.to_numeric(raw["lat"], errors="coerce"); lo = pd.to_numeric(raw["lon"], errors="coerce")
    rep = raw.assign(lat_=la.round(6), lon_=lo.round(6)).groupby(["lat_", "lon_"])["id"].size()
    ph = rep[rep > PLACEHOLDER_MIN_REPEATS]
    placeholders = set(ph.index)
    print("coordenadas por defecto:", dict(ph))
    df = apply_rules(raw, polys, remap, placeholders)
    with conn.cursor() as cur:
        cur.execute(DDL)
        execute_values(cur, "insert into analytics.reglas_calidad values %s", [(k, v[0], v[1]) for k, v in RULES.items()])
        execute_values(cur, "insert into analytics.jur_remap values %s",
                       [(int(r.declarada), int(r.real), float(r.participacion), int(r.filas)) for r in tbl.itertuples()])
        execute_values(cur, "insert into analytics.coord_por_defecto values %s", [(float(a), float(b), int(n)) for (a, b), n in ph.items()])
    conn.commit()
    insert(conn, df)
    summ = {k: sum(k in f for f in df["flags"]) for k in RULES}
    with conn.cursor() as cur:
        execute_values(cur, "insert into analytics.resumen_calidad values %s", list(summ.items()))
    conn.commit()
    print("\nreglas (filas afectadas):")
    for k, n in summ.items():
        print(f"  {k:<28}{n:>8}   {RULES[k][1]}")
    print(f"\nincluidas: {int(df['incluida'].sum())}  excluidas: {int((~df['incluida']).sum())} de {len(df)}")
    conn.close()


def clean_new(conn) -> int:
    """Limpieza incremental: procesa las filas de public.incidencias que aún no están en analytics (llegadas en vivo)."""
    with conn.cursor() as cur:                       # consulta barata: ¿hay filas sin procesar?
        cur.execute("select (select count(*) from public.incidencias), (select count(*) from analytics.incidencias_limpias)")
        n_src, n_clean = cur.fetchone()
    if n_src == n_clean:
        return 0
    raw = db.query_df(RAW_SQL + " left join analytics.incidencias_limpias a on a.id = i.id where a.id is null", conn)
    if raw.empty:
        return 0
    polys = load_polygons(conn)
    rm = db.query_df("select jurisdiccion_declarada d, jurisdiccion_real r, participacion p from analytics.jur_remap", conn)
    remap = {int(r.d): int(r.r) for r in rm.itertuples() if r.p >= REMAP_MIN_SHARE}
    ph = db.query_df("select lat, lon from analytics.coord_por_defecto", conn)
    placeholders = {(round(float(a), 6), round(float(b), 6)) for a, b in zip(ph["lat"], ph["lon"])}
    tmin = pd.to_datetime(raw["ocu"], utc=True, errors="coerce").min()
    known = db.query_df(
        "select ts, lat, lon, tipo_caso_id, jurisdiccion_id from analytics.incidencias_limpias where incluida and lat is not null and ts >= %(t)s",
        conn, params={"t": (tmin - pd.Timedelta(days=3)) if pd.notna(tmin) else pd.Timestamp("2026-01-01", tz="UTC")})
    known["ts"] = pd.to_datetime(known["ts"], utc=True)
    keys = set(zip(known["ts"].dt.floor("min"), known["lat"].round(6), known["lon"].round(6), known["tipo_caso_id"].astype(float), known["jurisdiccion_id"].astype(float)))
    df = apply_rules(raw, polys, remap, placeholders, keys)
    insert(conn, df)
    return len(df)


if __name__ == "__main__":
    build_all()
