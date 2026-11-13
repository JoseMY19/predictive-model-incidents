"""Conexión a PostgreSQL. Usa DATABASE_URL (variable de entorno o backend/.env)."""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import psycopg2

from . import config

ENV_FILE = config.ROOT / ".env"


def env(name: str, default: str | None = None) -> str | None:
    """Variable de entorno o, si no existe, la línea NOMBRE=valor de backend/.env."""
    if name in os.environ:
        return os.environ[name]
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip()
    return default


def database_url() -> str:
    url = env("DATABASE_URL")
    if not url:
        raise RuntimeError("Define DATABASE_URL (variable de entorno o backend/.env)")
    return url


def connect(dbname: str | None = None):
    u = urlparse(database_url())
    return psycopg2.connect(host=u.hostname, port=u.port or 5432, user=u.username, password=u.password,
                            dbname=dbname or u.path.lstrip("/"))


def ensure_database() -> None:
    """Crea la base indicada en DATABASE_URL si no existe."""
    name = urlparse(database_url()).path.lstrip("/")
    conn = connect("postgres")
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("select 1 from pg_database where datname=%s", (name,))
        if not cur.fetchone():
            cur.execute(f'create database "{name}" encoding \'UTF8\' template template0 lc_collate \'C\' lc_ctype \'C\'')
            print(f"Base creada: {name}")
        else:
            print(f"Base ya existe: {name}")
    conn.close()


def query_df(sql: str, conn, params=None, parse_dates=None) -> pd.DataFrame:
    """Consulta -> DataFrame usando el cursor de psycopg2 (sin SQLAlchemy ni avisos de pandas)."""
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        df = pd.DataFrame.from_records(cur.fetchall(), columns=cols, coerce_float=True)
    for c in parse_dates or []:
        df[c] = pd.to_datetime(df[c])
    return df


def as_of(conn) -> pd.Timestamp:
    """"Ahora" del sistema, hora de Lima sin zona: el reloj simulado si hay simulación; si no, el último registro
    insertado (en producción coincide con el momento actual)."""
    with conn.cursor() as cur:
        cur.execute("select to_regclass('sim.estado')")
        if cur.fetchone()[0]:
            cur.execute("select sim_now from sim.estado")
        else:
            cur.execute('select max("createdAt") from public.incidencias')
        v = cur.fetchone()[0]
    return pd.Timestamp(v).tz_convert(config.TZ).tz_localize(None)


if __name__ == "__main__":
    ensure_database()
    with connect() as c, c.cursor() as cur:
        cur.execute("select current_database(), split_part(version(),' ',2)")
        print(cur.fetchone())

