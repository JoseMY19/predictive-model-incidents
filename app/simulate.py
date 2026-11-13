"""Simulador de llegada de incidencias en tiempo (acelerado) sobre la base local.

    python -m app.simulate prepare --cutoff 2026-10-23   # retiene lo llegado desde esa fecha (las "futuras")
    python -m app.etl && python -m app.train             # reentrenar SOLO con datos anteriores al corte (prueba honesta)
    python -m uvicorn app.api:app --port 8000            # la API toma el estado "al corte"
    python -m app.simulate run --speed 7200              # 2 h simuladas por segundo real (8 días ≈ 1.5 min)
    python -m app.simulate reset                         # devuelve todo a su sitio (y borra el reloj simulado)

Las filas se re-insertan en public.incidencias en el orden original de llegada (createdAt), con sus evidencias y
serenos. La API detecta lo nuevo, lo limpia con las mismas reglas y recalcula zonas y pronóstico.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from . import config, db

CHILDREN = {"evidencias": '"incidenciaId"', "incidencia_serenos": '"incidenciaId"'}


def prepare(cutoff: str) -> None:
    ts = pd.Timestamp(cutoff, tz=config.TZ)
    with db.connect() as c, c.cursor() as cur:
        cur.execute("create schema if not exists sim")
        cur.execute("select to_regclass('sim.pendientes')")
        if cur.fetchone()[0]:
            raise SystemExit("Ya hay una simulación preparada. Ejecuta `reset` primero.")
        cur.execute('create table sim.pendientes as select * from public.incidencias where "createdAt" >= %s', (ts.to_pydatetime(),))
        for t, fk in CHILDREN.items():
            cur.execute(f"create table sim.{t}_pend as select * from public.{t} where {fk} in (select id from sim.pendientes)")
        cur.execute("delete from public.incidencias where id in (select id from sim.pendientes)")   # cascada a hijos
        cur.execute("delete from analytics.incidencias_limpias where id in (select id from sim.pendientes)")
        cur.execute("create table sim.estado (id int primary key default 1, sim_now timestamptz, corte timestamptz, velocidad numeric, "
                    "total int, corriendo boolean default false)")
        cur.execute("select count(*) from sim.pendientes")
        total = cur.fetchone()[0]
        cur.execute("insert into sim.estado (sim_now, corte, velocidad, total) values (%s, %s, 0, %s)", (ts.to_pydatetime(), ts.to_pydatetime(), total))
        cur.execute('select count(*) from public.incidencias'); left = cur.fetchone()[0]
    print(f"Retenidas {total} incidencias llegadas desde {ts}. Quedan {left} en la base. Reloj simulado = {ts}.")


def _release(cur, upto) -> int:
    cur.execute('select id from sim.pendientes where "createdAt" <= %s order by "createdAt", id', (upto,))
    ids = [r[0] for r in cur.fetchall()]
    if not ids:
        return 0
    cur.execute("insert into public.incidencias select * from sim.pendientes where id = any(%s)", (ids,))
    for t, fk in CHILDREN.items():
        cur.execute(f"insert into public.{t} select * from sim.{t}_pend where {fk} = any(%s)", (ids,))
        cur.execute(f"delete from sim.{t}_pend where {fk} = any(%s)", (ids,))
    cur.execute("delete from sim.pendientes where id = any(%s)", (ids,))
    return len(ids)


def run(speed: float, tick: float = 0.5) -> None:
    conn = db.connect()
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("select sim_now, total from sim.estado where id=1")
    sim_now, total = cur.fetchone()
    sim_now = pd.Timestamp(sim_now)
    cur.execute("select count(*) from sim.pendientes")
    pend = cur.fetchone()[0]
    print(f"Inicio {sim_now} | pendientes {pend}/{total} | velocidad x{speed:.0f} (1 s real = {speed/3600:.1f} h simuladas)")
    cur.execute("update sim.estado set corriendo=true, velocidad=%s where id=1", (speed,))
    try:
        while pend > 0:
            time.sleep(tick)
            sim_now = sim_now + pd.Timedelta(seconds=speed * tick)
            n = _release(cur, sim_now)
            pend -= n
            cur.execute("update sim.estado set sim_now=%s where id=1", (sim_now.to_pydatetime(),))
            if n:
                print(f"  {sim_now.tz_convert(config.TZ):%Y-%m-%d %H:%M}  +{n:>3} incidencias  (faltan {pend})", flush=True)
    except KeyboardInterrupt:
        print("Detenido (el reloj queda donde estaba; puedes reanudar con `run`).")
    cur.execute("update sim.estado set corriendo=false where id=1")
    print("Simulación terminada." if pend == 0 else f"Pendientes: {pend}")


def reset() -> None:
    with db.connect() as c, c.cursor() as cur:
        cur.execute("select to_regclass('sim.pendientes')")
        if cur.fetchone()[0]:
            _release(cur, pd.Timestamp("2100-01-01", tz="UTC").to_pydatetime())
        cur.execute("drop schema if exists sim cascade")
    print("Restaurado. Ejecuta `python -m app.etl && python -m app.train` para reentrenar con todos los datos "
          "(la API incorporará lo restaurado en su siguiente sincronización).")


def status() -> None:
    with db.connect() as c, c.cursor() as cur:
        cur.execute("select to_regclass('sim.estado')")
        if not cur.fetchone()[0]:
            print("Sin simulación (modo real).")
            return
        cur.execute("select sim_now, corte, velocidad, total, corriendo from sim.estado")
        print(cur.fetchone())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prepare"); p.add_argument("--cutoff", required=True, help="YYYY-MM-DD (hora de Lima)")
    r = sp.add_parser("run"); r.add_argument("--speed", type=float, default=7200)
    sp.add_parser("reset"); sp.add_parser("status")
    a = ap.parse_args()
    {"prepare": lambda: prepare(a.cutoff), "run": lambda: run(a.speed), "reset": reset, "status": status}[a.cmd]()
