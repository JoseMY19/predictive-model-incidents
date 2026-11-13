import numpy as np
import pandas as pd
import pytest

from app.compare import add_relative, to_markdown
from app.modeling import _metrics, backtest_windows, fit, make_model, predict


def test_backtest_windows_are_contiguous_and_end_on_last():
    last = pd.Timestamp("2026-10-29")
    w = backtest_windows(last, days=56, folds=3)
    assert len(w) == 3 and w[-1][1] == last
    assert all((b - a).days == 55 for a, b in w)
    assert all(w[i + 1][0] - w[i][1] == pd.Timedelta(days=1) for i in range(2))     # no gaps, no overlap


def test_r2_is_one_for_perfect_and_near_zero_for_the_mean():
    idx = pd.date_range("2026-01-01", periods=30)
    cols = pd.MultiIndex.from_tuples([(1, "PATRIMONIO"), (2, "PATRIMONIO")])
    actual = pd.DataFrame(np.random.default_rng(1).poisson(5, (30, 2)).astype(float), index=idx, columns=cols)
    assert _metrics(actual, actual)["r2_jurisdiction"] == pytest.approx(1.0)
    mean_pred = actual * 0 + actual.to_numpy().mean()
    assert abs(_metrics(mean_pred, actual)["r2_jurisdiction"]) < 0.05


def test_relative_improvement_is_negative_when_better_than_baseline():
    rows = add_relative([{"model": "baseline_media_7d", "mae_jurisdiccion": 10.0}, {"model": "x", "mae_jurisdiccion": 9.0}])
    assert rows[1]["vs_baseline_pct"] == pytest.approx(-10.0)


def test_markdown_table_is_sorted_best_first():
    rows = add_relative([
        {"model": "baseline_media_7d", "label": "Baseline", "mae_jurisdiccion": 10.0, "rmse_jurisdiccion": 12.0, "r2_jurisdiction": 0.2,
         "wape_jurisdiccion_pct": 40.0, "wape_semana_jurisdiccion_pct": 25.0, "mae_by_window": [10.0]},
        {"model": "m", "label": "Better", "mae_jurisdiccion": 9.0, "rmse_jurisdiccion": 11.0, "r2_jurisdiction": 0.3,
         "wape_jurisdiccion_pct": 36.0, "wape_semana_jurisdiccion_pct": 22.0, "mae_by_window": [9.0]}])
    md = to_markdown(rows, [(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-25"))])
    assert md.index("Better") < md.index("Baseline") and "-10.0 %" in md


@pytest.mark.parametrize("kind", ["ridge", "knn", "random_forest", "hist_gradient_boosting"])
def test_models_fit_and_forecast_non_negative(kind, wide_panel):
    model, cols, _ = fit(kind, wide_panel)
    fc = predict(kind, model, cols, wide_panel, wide_panel.index[-1], 5)
    assert fc.shape == (5, wide_panel.shape[1]) and (fc.to_numpy() >= 0).all() and np.isfinite(fc.to_numpy()).all()


def test_xgboost_if_installed(wide_panel):
    pytest.importorskip("xgboost")
    model, cols, _ = fit("xgboost", wide_panel)
    assert (model.predict(pd.DataFrame(np.zeros((1, len(cols))), columns=cols)) >= 0).all()


def test_unknown_model_kind_raises():
    with pytest.raises(ValueError):
        make_model("does_not_exist")
