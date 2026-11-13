"""Importa las cámaras municipales REALES (backend Sivecam del mapa) a geo/camaras_municipales.json.

El mapa MAP610NEW no tiene un JSON de las 610 cámaras: las pide al backend Sivecam (GET {API}/municipal?page=0, con
token). Hay dos formas de traerlas:

  A) Con login (solo lectura, solo GET).  En backend/.env:
        SIVECAM_API_URL=http://<host>:<puerto>/api        (la VITE_API_URL del mapa)
        SIVECAM_API_USER=...   SIVECAM_API_PASSWORD=...
     y ejecutar:   python -m app.camaras_remote

  B) Sin credenciales aquí: en el mapa ya logueado, abrir DevTools > Network, recargar, buscar la petición
     «municipal?page=0», clic derecho > Copy > Copy response, guardarla como un .json y ejecutar:
        python -m app.camaras_remote --archivo ruta\\a\\respuesta.json

Se guardan solo los campos útiles (nombre, tipo, coordenadas, ángulo, estado); no se guarda nada de usuarios.
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

from . import config, db
from .remote import _find_token

SALIDA = config.GEO_DIR / "camaras_municipales.json"
SALIDA_VECINAL = config.GEO_DIR / "camaras_vecinales.json"
TIPOS = {"C180": "TIPO I", "C360": "TIPO II", "LPR": "TIPO III"}


def extraer(resp) -> list[dict]:
    """La API responde {message, data: {data: [...], totalCount, ...}}; también acepto una lista suelta."""
    if isinstance(resp, list):
        return resp
    d = resp.get("data", resp)
    if isinstance(d, dict):
        d = d.get("data", [])
    return d if isinstance(d, list) else []


def es_vecinal(items: list[dict]) -> bool:
    """Las cámaras vecinales (endpoint /communal) traen marca/vecino/serial; las municipales (/municipal) traen name/camera."""
    return bool(items) and any(("neighbor" in r or "brand" in r) and "camera" not in r for r in items[:20])


def _limites() -> tuple[float, float, float, float]:
    """Caja de San Juan de Lurigancho (+3 km) a partir del geojson de jurisdicciones."""
    g = json.loads((config.GEO_DIR / "juridiccion.geojson").read_text(encoding="utf-8"))
    xs: list[float] = []; ys: list[float] = []

    def walk(c):
        if isinstance(c[0], (int, float)):
            xs.append(c[0]); ys.append(c[1])
        else:
            for i in c:
                walk(i)
    for f in g["features"]:
        walk(f["geometry"]["coordinates"])
    m = 0.03
    return min(ys) - m, max(ys) + m, min(xs) - m, max(xs) + m


def _coord(r: dict, lim) -> tuple[float, float, bool] | None:
    """Coordenada válida dentro de SJL. Si la latitud viene con signo invertido (error de captura típico: +11.97 en vez
    de -11.97) y al invertirla cae en SJL, se corrige y se marca. Fuera de SJL => None (se descarta)."""
    la, lo = r.get("latitude"), r.get("longitude")
    if la is None or lo is None:
        return None
    la, lo = float(la), float(lo)
    dentro = lambda a, b: lim[0] <= a <= lim[1] and lim[2] <= b <= lim[3]
    if dentro(la, lo):
        return la, lo, False
    if dentro(-la, lo):
        return -la, lo, True
    return None


def normalizar(items: list[dict]) -> tuple[list[dict], list[str]]:
    """Cámaras municipales -> (lista, avisos). Descarta borradas y las que quedan fuera de SJL."""
    lim, out, avisos = _limites(), [], []
    for r in items:
        if r.get("deleted_at"):
            continue
        c = _coord(r, lim)
        if c is None:
            avisos.append(f"descartada (coordenada fuera de SJL): {r.get('name') or r.get('address')} ({r.get('latitude')}, {r.get('longitude')})")
            continue
        out.append({"nombre": r.get("name"), "tipo": TIPOS.get(r.get("camera"), r.get("camera")), "modelo": r.get("camera"),
                    "direccion": r.get("address"), "latitude": c[0], "longitude": c[1], "corregida": c[2],
                    "angulo": r.get("angle"), "oculta": bool(r.get("map_hidden")),
                    "megafono": bool(r.get("megaphone")), "boton": bool(r.get("buttom"))})
    return out, avisos


def normalizar_vecinales(items: list[dict]) -> tuple[list[dict], list[str]]:
    """Cámaras vecinales: SOLO posición, marca y modo. Se descartan vecino, teléfono, serial y usuario/clave."""
    lim, out, avisos = _limites(), [], []
    for r in items:
        if r.get("deleted_at"):
            continue
        c = _coord(r, lim)
        if c is None:
            avisos.append(f"descartada (coordenada fuera de SJL): ({r.get('latitude')}, {r.get('longitude')})")
            continue
        out.append({"tipo": "VECINAL", "marca": r.get("brand"), "modo": r.get("mode"), "latitude": c[0], "longitude": c[1], "corregida": c[2]})
    return out, avisos


def _http(url: str, token: str | None = None, body: dict | None = None):
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=h, data=json.dumps(body).encode() if body is not None else None,
                                 method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=60) as f:
            return json.load(f)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"La API respondió HTTP {e.code} en {url.split('?')[0]}") from None


def desde_api() -> list[dict]:
    base = (db.env("SIVECAM_API_URL") or "").rstrip("/")
    user, pwd = db.env("SIVECAM_API_USER"), db.env("SIVECAM_API_PASSWORD")
    if not (base and user and pwd):
        raise SystemExit("Faltan SIVECAM_API_URL, SIVECAM_API_USER y SIVECAM_API_PASSWORD en backend/.env (o usa --archivo)")
    tok = _find_token(_http(base + "/auth/login", body={"username": user, "password": pwd}))
    if not tok:
        raise SystemExit("El login no devolvió un token reconocible")
    resp = _http(base + "/municipal?page=0", token=tok)
    total = resp.get("data", {}).get("totalCount") if isinstance(resp.get("data"), dict) else None
    items = extraer(resp)
    if total and total != len(items):
        print(f"AVISO: la API dice {total} cámaras pero devolvió {len(items)} (¿paginación?)")
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archivo", help="JSON con la respuesta de /municipal guardada desde el navegador")
    a = ap.parse_args()
    if a.archivo:
        with open(a.archivo, encoding="utf-8-sig") as f:
            items = extraer(json.load(f))
    else:
        items = desde_api()
    from collections import Counter
    if es_vecinal(items):
        cams, avisos = normalizar_vecinales(items)
        destino, tipo = SALIDA_VECINAL, "vecinales"
        print("OJO: esta respuesta es de cámaras VECINALES (endpoint communal), no de las municipales.\n"
              "     Se guarda aparte (solo posición/marca/modo; sin datos de vecinos ni claves). Las municipales son la petición «municipal?page=0».")
    else:
        cams, avisos = normalizar(items)
        destino, tipo = SALIDA, "municipales"
    if not cams:
        raise SystemExit("No se obtuvo ninguna cámara con coordenadas válidas; no se escribe nada.")
    destino.write_text(json.dumps({"camaras": cams}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(cams)} cámaras {tipo} guardadas en {destino}  (corregidas por signo de latitud: {sum(c['corregida'] for c in cams)})")
    if tipo == "municipales":
        print("  por tipo:", dict(Counter(c["tipo"] for c in cams)), " fuera de servicio:", sum(c["oculta"] for c in cams))
    for a in avisos:
        print("  AVISO:", a)


if __name__ == "__main__":
    main()
