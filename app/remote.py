"""Sincronización con la API de incidencias en producción (solo lectura: únicamente peticiones GET/login).

Trae a `public.incidencias` (la copia local) las incidencias nuevas de Serenazgo. El resto del sistema (limpieza,
modelo, mapa) no cambia: sigue leyendo la base local.

Configuración (variable de entorno o backend/.env; nunca se imprime ni se guarda en el código):
  INCIDENCIAS_API_URL       p. ej. http://<host>:<puerto>/api/v1       (si falta, la sincronización queda apagada)
  INCIDENCIAS_API_USER      usuario  \\ se hace login y se renueva el token solo cuando caduca (recomendado:
  INCIDENCIAS_API_PASSWORD  clave    /  usuario dedicado de solo lectura)
  INCIDENCIAS_API_TOKEN     alternativa: token fijo (caduca; sirve para pruebas)

Minimización de datos: NO se copian nombres, teléfonos ni direcciones; solo los campos que usa el modelo.
Prueba manual:  python -m app.remote
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from psycopg2.extras import execute_values

from . import config, db

PAGE = 100                      # la API no admite más de 100 por página
TIMEOUT = 20


def _find_token(o) -> str | None:
    if isinstance(o, dict):
        for k in ("accessToken", "access_token", "token", "jwt"):
            if isinstance(o.get(k), str):
                return o[k]
        for v in o.values():
            t = _find_token(v)
            if t:
                return t
    return None


def _num(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


class RemoteSync:
    def __init__(self) -> None:
        self.url = (db.env("INCIDENCIAS_API_URL") or "").rstrip("/")
        self.user = db.env("INCIDENCIAS_API_USER")
        self.password = db.env("INCIDENCIAS_API_PASSWORD")
        self.token = db.env("INCIDENCIAS_API_TOKEN")
        self.total_importadas = 0
        self.ultima_importadas = 0
        self.ultima_ok: float | None = None
        self.error: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.url and (self.token or (self.user and self.password)))

    # ------------------------------------------------------------------ HTTP
    def _login(self) -> None:
        if not (self.user and self.password):
            raise RuntimeError("Token vencido y no hay INCIDENCIAS_API_USER/PASSWORD para renovarlo")
        req = urllib.request.Request(self.url + "/auth/login", method="POST", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"username": self.user, "password": self.password}).encode())
        with urllib.request.urlopen(req, timeout=TIMEOUT) as f:
            tok = _find_token(json.load(f))
        if not tok:
            raise RuntimeError("El login no devolvió un token reconocible")
        self.token = tok

    def _get(self, path: str):
        for intento in (1, 2):
            if not self.token:
                self._login()
            req = urllib.request.Request(self.url + path, headers={"Authorization": f"Bearer {self.token}"})
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT) as f:
                    return json.load(f)
            except urllib.error.HTTPError as e:
                if e.code == 401 and intento == 1:
                    self.token = None                       # vencido: renovar y reintentar una vez
                    continue
                raise RuntimeError(f"La API respondió HTTP {e.code} en {path.split('?')[0]}") from None
        raise RuntimeError("No se pudo autenticar en la API")

    # ------------------------------------------------------------------ sincronización
    def fetch_new(self, max_id: int) -> list[dict]:
        """Páginas de más nuevo a más antiguo hasta tocar lo que ya se tiene (la API ordena por id descendente)."""
        out: list[dict] = []
        page = 1
        while True:
            js = self._get(f"/incidencias?unidadId={config.UNIDAD_ID}&limit={PAGE}&page={page}")["data"]
            rows = js["data"]
            out += rows
            if not rows or min(r["id"] for r in rows) <= max_id or page >= js["meta"]["totalPages"]:
                return out
            page += 1

    def sync(self) -> int:
        """Inserta en la base local lo que falta, todo en UNA transacción (si algo falla no queda un hueco a medias).
        Devuelve cuántas filas nuevas entraron."""
        if not self.enabled:
            return 0
        try:
            with db.connect() as conn:                        # conexión propia, sin autocommit
                with conn.cursor() as cur:
                    cur.execute('select coalesce(max(id), 0) from public.incidencias')
                    max_id = cur.fetchone()[0]
                n = self._insert(conn, self.fetch_new(max_id))
            conn.close()
            self.ultima_importadas = n
            self.total_importadas += n
            self.ultima_ok = time.time()
            self.error = None
            return n
        except Exception as e:                                  # la sincronización falla sin tumbar el resto
            self.error = f"{type(e).__name__}: {e}"[:200]
            return 0

    def _insert(self, conn, items: list[dict]) -> int:
        if not items:
            return 0
        tipos = {i["tipoCaso"]["id"]: i["tipoCaso"] for i in items if i.get("tipoCaso")}
        subs = {i["subTipoCaso"]["id"]: i["subTipoCaso"] for i in items if i.get("subTipoCaso")}
        jurs = {i["jurisdiccion"]["id"]: i["jurisdiccion"] for i in items if i.get("jurisdiccion")}
        rows = [(i["id"], i.get("codigoIncidencia"), i.get("unidadId"), i.get("tipoCasoId"), i.get("subTipoCasoId"),
                 i.get("jurisdiccionId"), _num(i.get("latitud")), _num(i.get("longitud")), i.get("descripcion"),
                 i.get("registradoEn"), i.get("ocurridoEn"), i["createdAt"], i["updatedAt"]) for i in items]
        with conn.cursor() as cur:
            # catálogos que aún no existan localmente (para no violar las llaves foráneas)
            execute_values(cur, 'insert into public.tipo_casos (id, descripcion, habilitado, "unidadId", "createdAt", "updatedAt", codigo) '
                                'values %s on conflict (id) do nothing',
                           [(t["id"], t.get("descripcion"), t.get("habilitado", True), t.get("unidadId"), t["createdAt"], t["updatedAt"], t.get("codigo"))
                            for t in tipos.values()])
            execute_values(cur, 'insert into public.sub_tipo_casos (id, descripcion, habilitado, "tipoCasoId", "createdAt", "updatedAt", codigo, urgencia) '
                                'values %s on conflict (id) do nothing',
                           [(s["id"], s.get("descripcion"), s.get("habilitado", True), s.get("tipoCasoId"), s["createdAt"], s["updatedAt"],
                             s.get("codigo"), s.get("urgencia")) for s in subs.values()])
            execute_values(cur, 'insert into public.jurisdicciones (id, codigo, nombre, habilitado, "createdAt", "updatedAt") '
                                'values %s on conflict (id) do nothing',
                           [(j["id"], j.get("codigo"), j.get("nombre"), j.get("habilitado", True), j["createdAt"], j["updatedAt"]) for j in jurs.values()])
            cur.execute("select count(*) from public.incidencias")
            antes = cur.fetchone()[0]
            execute_values(cur, 'insert into public.incidencias (id, "codigoIncidencia", "unidadId", "tipoCasoId", "subTipoCasoId", "jurisdiccionId", '
                                'latitud, longitud, descripcion, "registradoEn", "ocurridoEn", "createdAt", "updatedAt") values %s '
                                'on conflict (id) do nothing', rows)
            cur.execute("select count(*) from public.incidencias")
            despues = cur.fetchone()[0]
        return despues - antes

    def status(self) -> dict:
        return {"activa": self.enabled, "origen": self.url or None, "total_importadas": self.total_importadas,
                "ultima_importadas": self.ultima_importadas,
                "ultima_ok": time.strftime("%H:%M:%S", time.localtime(self.ultima_ok)) if self.ultima_ok else None,
                "error": self.error}


if __name__ == "__main__":
    r = RemoteSync()
    if not r.enabled:
        raise SystemExit("Configura INCIDENCIAS_API_URL y (INCIDENCIAS_API_USER + INCIDENCIAS_API_PASSWORD) o INCIDENCIAS_API_TOKEN")
    t = time.time()
    n = r.sync()
    print(f"nuevas: {n}  ({time.time() - t:.1f}s)  estado: {r.status()}")
