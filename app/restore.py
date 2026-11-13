"""Restaura un backup pg_dump (formato custom) en la base definida por DATABASE_URL.

Uso:  python -m app.restore --dump <archivo>
El esquema original queda en `public` (datos de origen, intactos). La limpieza vive en el esquema `analytics`.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from . import db
from .etl import find_pg_restore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, type=Path)
    a = ap.parse_args()
    db.ensure_database()
    u = urlparse(db.database_url())
    env = {**os.environ, "PGPASSWORD": u.password or ""}
    cmd = [find_pg_restore(), "-h", u.hostname, "-p", str(u.port or 5432), "-U", u.username, "-d", u.path.lstrip("/"),
           "--no-owner", "--no-acl", "--exit-on-error", str(a.dump)]
    print("Restaurando…")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit(f"pg_restore falló:\n{r.stderr[-2000:]}")
    with db.connect() as c, c.cursor() as cur:
        cur.execute("select count(*) from public.incidencias")
        print("incidencias restauradas:", cur.fetchone()[0])


if __name__ == "__main__":
    main()
