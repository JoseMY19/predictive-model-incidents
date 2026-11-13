"""Entrena, evalúa (backtest temporal en varias ventanas) y guarda el mejor modelo.

Uso:  python -m app.train
Requiere haber ejecutado antes:  python -m app.etl --dump <backup>
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from . import config
from .features import build_wide, load_catalog
from .modeling import MODEL_LABELS, backtest, fit

CANDIDATES = ["baseline_media_7d", "baseline_semana_anterior", "random_forest", "extra_trees", "hist_gradient_boosting", "ensemble_rf_es"]
METRIC_KEYS = ["mae_serie", "rmse_serie", "mae_jurisdiccion", "rmse_jurisdiccion", "wape_jurisdiccion_pct",
               "mae_patrimonio", "mae_semana_jurisdiccion", "wape_semana_jurisdiccion_pct"]


def main() -> None:
    catalog = load_catalog()
    panel = pd.read_csv(config.PANEL_CSV, parse_dates=["fecha"])
    wide = build_wide(panel, catalog)
    last = pd.Timestamp(catalog["ultima_fecha"])
    # Ventanas móviles de BACKTEST_DAYS días, la última termina en el último día con datos.
    windows = [(last - pd.Timedelta(days=config.BACKTEST_DAYS * (i + 1) - 1) , last - pd.Timedelta(days=config.BACKTEST_DAYS * i))
               for i in range(config.BACKTEST_FOLDS)][::-1]
    print(f"Datos válidos: {int(wide.notna().all(axis=1).sum())} días | ventanas de backtest: "
          + ", ".join(f"{a.date()}→{b.date()}" for a, b in windows))

    results = {}
    for kind in CANDIDATES:
        folds = [backtest(kind, wide, a, b)[0] for a, b in windows]
        mean = {k: float(np.mean([f[k] for f in folds])) for k in METRIC_KEYS}
        results[kind] = {"nombre": MODEL_LABELS[kind], **mean,
                         "mae_jurisdiccion_por_ventana": [round(f["mae_jurisdiccion"], 3) for f in folds]}
        print(f"{MODEL_LABELS[kind]:<40} MAE/jur-día={mean['mae_jurisdiccion']:.2f} (ventanas {results[kind]['mae_jurisdiccion_por_ventana']})  "
              f"RMSE={mean['rmse_jurisdiccion']:.2f}  WAPE día={mean['wape_jurisdiccion_pct']:.1f}%  "
              f"WAPE semana={mean['wape_semana_jurisdiccion_pct']:.1f}%")

    # Selección entre modelos de ML (los baselines son solo referencia) por MAE jurisdicción-día promedio.
    ml = {k: v for k, v in results.items() if not k.startswith("baseline")}
    best = min(ml, key=lambda k: ml[k]["mae_jurisdiccion"])
    base = results["baseline_media_7d"]["mae_jurisdiccion"]
    mejora = 100 * (base - ml[best]["mae_jurisdiccion"]) / base
    print(f"\nMejor modelo: {MODEL_LABELS[best]} (mejora {mejora:.1f}% de MAE vs baseline) -> reentrenando con todos los datos hasta {last.date()}")

    model, cols, n_rows = fit(best, wide)
    importancias = None
    if hasattr(model, "feature_importances_"):
        importancias = dict(sorted(zip(cols, map(float, model.feature_importances_)), key=lambda kv: -kv[1])[:12])

    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "cols": cols, "kind": best, "last_date": last}, config.MODEL_FILE)
    metrics = {
        "entrenado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modelo": best,
        "modelo_nombre": MODEL_LABELS[best],
        "filas_entrenamiento": n_rows,
        "ultima_fecha_datos": last.strftime("%Y-%m-%d"),
        "mejora_vs_baseline_pct": round(mejora, 2),
        "backtest": {
            "desde": windows[0][0].strftime("%Y-%m-%d"), "hasta": last.strftime("%Y-%m-%d"),
            "ventanas": [[a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")] for a, b in windows],
            "horizonte_dias": config.BACKTEST_HORIZON,
            "descripcion": f"{config.BACKTEST_FOLDS} ventanas móviles de {config.BACKTEST_DAYS} días. En cada una se entrena con "
                           f"datos anteriores y se pronostican {config.BACKTEST_HORIZON} días recursivamente, cada {config.BACKTEST_HORIZON} días. "
                           "Los resultados son el promedio de las ventanas.",
            "resultados": results,
        },
        "importancias": importancias,
    }
    config.METRICS_JSON.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Guardado: {config.MODEL_FILE}\nGuardado: {config.METRICS_JSON}")


if __name__ == "__main__":
    main()
