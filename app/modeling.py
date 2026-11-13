"""Modelos, pronóstico recursivo y backtest temporal."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from . import config
from .features import feature_columns, make_features, training_rows
from .outliers import winsorize_target


def _base_model(kind: str):
    if kind == "random_forest":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=15, max_features=0.5,
                                     n_jobs=-1, random_state=config.RANDOM_STATE)
    if kind == "extra_trees":
        return ExtraTreesRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.5,
                                   n_jobs=-1, random_state=config.RANDOM_STATE)
    if kind == "hist_gradient_boosting":
        # Pérdida Poisson: apropiada para conteos (no negativos, muchos ceros).
        return HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=300,
                                             min_samples_leaf=20, random_state=config.RANDOM_STATE)
    if kind == "ridge":
        # Linear baseline. RobustScaler (median/IQR) keeps heavy-tailed count features from dominating the fit.
        return make_pipeline(RobustScaler(), Ridge(alpha=10.0))
    if kind == "knn":
        # Distance-based model: scaling is mandatory, otherwise lag_28 would outweigh the one-hot columns.
        return make_pipeline(RobustScaler(), KNeighborsRegressor(n_neighbors=25, weights="distance", n_jobs=-1))
    if kind == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as e:                                    # optional dependency
            raise ImportError("xgboost is not installed: pip install xgboost") from e
        return XGBRegressor(objective="count:poisson", n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=10, n_jobs=-1, random_state=config.RANDOM_STATE)
    raise ValueError(kind)


def make_model(kind: str, **params):
    """Model with the project defaults; `params` override any hyper-parameter (sklearn `set_params` names, so
    pipeline steps are addressed as `ridge__alpha`, `kneighborsregressor__n_neighbors`)."""
    model = _base_model(kind)
    if params:
        model.set_params(**params)
    return model


MODEL_LABELS = {
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "ensemble_rf_es": "Ensamble (Random Forest + suavizado exponencial)",
    "hist_gradient_boosting": "Gradient Boosting (Poisson)",
    "ridge": "Ridge regression",
    "knn": "K-Nearest Neighbors",
    "xgboost": "XGBoost (Poisson)",
    "baseline_media_7d": "Baseline: media últimos 7 días",
    "baseline_semana_anterior": "Baseline: mismo día semana anterior",
}


def fit(kind: str, wide: pd.DataFrame, train_end: pd.Timestamp | None = None, winsorize_pct: float | None = None,
        params: dict | None = None):
    """Entrena con todas las filas completas hasta `train_end` (inclusive).
    `winsorize_pct` (p. ej. 98): limita el objetivo de ENTRENAMIENTO al percentil por serie (ver app/outliers.py)."""
    df = training_rows(make_features(wide))
    if train_end is not None:
        df = df[df["fecha"] <= train_end]
    if winsorize_pct is not None:
        df = winsorize_target(df, winsorize_pct)
    cols = feature_columns(df)
    model = make_model("random_forest" if kind == ENSEMBLE else kind, **(params or {}))
    model.fit(df[cols], df["y"])
    return model, cols, len(df)


def recursive_forecast(model, cols: list[str], wide: pd.DataFrame, origin: pd.Timestamp,
                       horizon: int) -> pd.DataFrame:
    """Pronostica `horizon` días después de `origin` (último día observado).
    Cada día predicho alimenta los lags del siguiente. Devuelve fecha x serie."""
    w = wide.loc[:origin].copy()
    preds = []
    for h in range(1, horizon + 1):
        day = origin + pd.Timedelta(days=h)
        w.loc[day] = np.nan
        feats = make_features(w)
        today = feats[feats["fecha"] == day]
        yhat = np.clip(model.predict(today[cols]), 0, None)
        w.loc[day] = yhat
        preds.append(pd.Series(yhat, index=w.columns, name=day))
    return pd.DataFrame(preds)


ENSEMBLE = "ensemble_rf_es"
ES_ALPHA, ES_WEIGHT_RF = 0.4, 0.5      # elegidos por backtest (ver README): nivel exponencial 0.4, 50 % RF / 50 % suavizado


def es_dow_forecast(hist: pd.Series, alpha: float, h: int, dow_weeks: int = 8) -> np.ndarray:
    """Suavizado exponencial del total (desestacionalizado por día de la semana) sobre el último tramo CONTINUO de datos."""
    y = hist.dropna()
    gaps = np.where(y.index.to_series().diff().dt.days.to_numpy() > 1)[0]
    if len(gaps):
        y = y.iloc[gaps[-1]:]
    tail = y.iloc[-7 * dow_weeks:]
    fac = tail.groupby(tail.index.dayofweek).mean() / tail.mean()
    fac = fac.reindex(range(7)).fillna(1.0)
    ds = y / fac.reindex(y.index.dayofweek).to_numpy()
    level = float(ds.iloc[0])
    for v in ds.iloc[1:]:
        level = alpha * v + (1 - alpha) * level
    return np.array([level * fac[(hist.index[-1] + pd.Timedelta(days=k)).dayofweek] for k in range(1, h + 1)])


def predict(kind: str, model, cols: list[str], wide: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """Pronóstico fecha x serie para cualquier tipo de modelo. En el ensamble, el total de cada jurisdicción es
    la mezcla RF/suavizado exponencial y la composición por tipo viene del RF (reconciliación proporcional)."""
    fc = recursive_forecast(model, cols, wide, origin, horizon)
    if kind != ENSEMBLE:
        return fc
    hist_tot = wide.loc[:origin].T.groupby(level=0).sum(min_count=1).T
    out = fc.copy()
    for j in hist_tot.columns:
        rf_tot = fc.loc[:, j].sum(axis=1).to_numpy()
        target = ES_WEIGHT_RF * rf_tot + (1 - ES_WEIGHT_RF) * es_dow_forecast(hist_tot[j], ES_ALPHA, horizon)
        scale = np.where(rf_tot > 1e-9, target / np.maximum(rf_tot, 1e-9), 1.0)
        out.loc[:, j] = fc.loc[:, j].to_numpy() * scale[:, None]
    return out


def baseline_forecast(kind: str, wide: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
    hist = wide.loc[:origin]
    days = [origin + pd.Timedelta(days=h) for h in range(1, horizon + 1)]
    if kind == "baseline_media_7d":
        mean7 = hist.iloc[-7:].mean()
        return pd.DataFrame([mean7.rename(d) for d in days])
    if kind == "baseline_semana_anterior":
        # día t -> valor observado en t-7*ceil(h/7) (siempre <= origen)
        rows = []
        for h, d in enumerate(days, start=1):
            back = 7 * int(np.ceil(h / 7))
            rows.append(hist.loc[d - pd.Timedelta(days=back)].rename(d))
        return pd.DataFrame(rows)
    raise ValueError(kind)


def _metrics(pred: pd.DataFrame, actual: pd.DataFrame) -> dict:
    """MAE/RMSE a nivel serie-día y jurisdicción-día (suma de grupos), + MAE de PATRIMONIO."""
    e = (pred - actual).to_numpy().ravel()
    out = {"mae_serie": float(np.mean(np.abs(e))), "rmse_serie": float(np.sqrt(np.mean(e ** 2)))}
    pj, aj = pred.T.groupby(level=0).sum().T, actual.T.groupby(level=0).sum().T
    ej = (pj - aj).to_numpy().ravel()
    out["mae_jurisdiccion"] = float(np.mean(np.abs(ej)))
    out["rmse_jurisdiccion"] = float(np.sqrt(np.mean(ej ** 2)))
    out["wape_jurisdiccion_pct"] = float(100 * np.abs(ej).sum() / max(aj.to_numpy().sum(), 1e-9))
    a_flat = aj.to_numpy().ravel()
    out["r2_jurisdiction"] = float(1 - (ej ** 2).sum() / max(((a_flat - a_flat.mean()) ** 2).sum(), 1e-9))
    pat = [c for c in pred.columns if c[1] == "PATRIMONIO"]
    out["mae_patrimonio"] = float(np.mean(np.abs((pred[pat] - actual[pat]).to_numpy())))
    return out


def backtest(kind: str, wide: pd.DataFrame, test_start: pd.Timestamp, test_end: pd.Timestamp,
             horizon: int = config.BACKTEST_HORIZON, winsorize_pct: float | None = None,
             params: dict | None = None) -> tuple[dict, dict]:
    """Origen móvil: entrena con datos < test_start y pronostica `horizon` días recursivamente
    desde cada origen (cada `horizon` días) dentro de [test_start, test_end].
    Devuelve (metricas globales, mae jurisdicción-día por día de horizonte)."""
    is_baseline = kind.startswith("baseline")
    model = cols = None
    if not is_baseline:
        model, cols, _ = fit(kind, wide, train_end=test_start - pd.Timedelta(days=1), winsorize_pct=winsorize_pct, params=params)
    preds, actuals = [], []
    origin = test_start - pd.Timedelta(days=1)
    while origin + pd.Timedelta(days=horizon) <= test_end:
        p = (baseline_forecast(kind, wide, origin, horizon) if is_baseline
             else predict(kind, model, cols, wide, origin, horizon))
        a = wide.loc[p.index]
        preds.append(p); actuals.append(a)
        origin += pd.Timedelta(days=horizon)
    P, A = pd.concat(preds), pd.concat(actuals)
    # Error del total semanal (horizonte completo) por jurisdicción.
    wk_p = np.concatenate([p.T.groupby(level=0).sum().sum(axis=1).to_numpy() for p in preds])
    wk_a = np.concatenate([a.T.groupby(level=0).sum().sum(axis=1).to_numpy() for a in actuals])
    by_h = {}
    for h in range(1, horizon + 1):
        ph = pd.concat([p.iloc[[h - 1]] for p in preds]); ah = pd.concat([a.iloc[[h - 1]] for a in actuals])
        by_h[h] = _metrics(ph, ah)["mae_jurisdiccion"]
    metrics = _metrics(P, A)
    metrics["mae_semana_jurisdiccion"] = float(np.mean(np.abs(wk_p - wk_a)))
    metrics["wape_semana_jurisdiccion_pct"] = float(100 * np.abs(wk_p - wk_a).sum() / wk_a.sum())
    return metrics, by_h


def backtest_windows(last: pd.Timestamp, days: int = config.BACKTEST_DAYS,
                     folds: int = config.BACKTEST_FOLDS) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Rolling evaluation windows of `days` days, oldest first; the newest one ends on `last`."""
    return [(last - pd.Timedelta(days=days * (i + 1) - 1), last - pd.Timedelta(days=days * i)) for i in range(folds)][::-1]
