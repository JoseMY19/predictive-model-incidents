"""Configuración central del proyecto de predicción de incidencias."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
GEO_DIR = ROOT / "geo"                         # polígonos de jurisdicción y cámaras (copiados del mapa)

PANEL_CSV = DATA_DIR / "daily_panel.csv"      # conteos diarios agregados (sin datos personales)
GEO_CSV = DATA_DIR / "incidents_geo.csv"       # incidencias válidas con coordenadas (para el modelo espacial)
CATALOG_JSON = DATA_DIR / "catalog.json"       # jurisdicciones, grupos, rango de fechas
MODEL_FILE = MODEL_DIR / "model.joblib"
METRICS_JSON = MODEL_DIR / "metrics.json"
SPATIAL_JSON = MODEL_DIR / "spatial_metrics.json"

TZ = "America/Lima"

# Solo Serenazgo (unidadId=1): es la única unidad con carga consistente en todo el histórico.
# A partir de marzo 2026 la unidad Policía casi deja de registrarse (~4 000/mes -> ~30/mes).
UNIDAD_ID = 1

# Períodos con registro fiable (por fecha de ocurrencia). `None` = hasta el último día completo.
# Excluidos a propósito:
#   - 2024-12-30/31 : carga histórica con fecha por defecto.
#   - 2025-10-29 .. 2026-03-04 : hueco (nov/dic 2025 sin datos, ene-feb 2026 carga parcial ~40/día).
#   - 2026-03-05 .. 2026-03-31 : arranque del sistema (0 -> 350/día en 3 semanas).
VALID_RANGES = [("2025-01-01", "2025-10-28"), ("2026-04-01", None)]

# Agrupación de `tipoCasoId` (catálogo tipo_casos, unidad Serenazgo) en grupos de interés.
GRUPOS = {
    "PATRIMONIO": [3, 11],            # robos / hurtos / daños (foco de la tesis)
    "PERSONA": [1, 2, 10],            # contra la vida, el cuerpo, la libertad, la persona
    "ORDENANZAS": [18],               # infracciones a ordenanzas y licencias municipales
    "AUXILIO": [19],                  # ayuda, auxilio y rescate
    "ALCOHOL_COSTUMBRES": [13],       # consumo de alcohol/drogas en vía pública, etc.
    "TRANQUILIDAD": [17, 7, 14],      # alteración de tranquilidad y orden
    "TRANSITO": [15],                 # accidentes e infracciones de tránsito
}
GRUPO_OTROS = "OTROS"
GRUPO_NAMES = list(GRUPOS) + [GRUPO_OTROS]
TIPO_TO_GRUPO = {t: g for g, ts in GRUPOS.items() for t in ts}

HORIZON_MAX = 28          # días máximos a pronosticar
BACKTEST_DAYS = 56        # tamaño de cada ventana de evaluación (8 semanas)
BACKTEST_FOLDS = 3        # nº de ventanas móviles (la última termina en el último día con datos)
BACKTEST_HORIZON = 7      # horizonte (días) evaluado en cada origen del backtest
RANDOM_STATE = 42
