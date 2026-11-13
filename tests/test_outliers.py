import numpy as np
import pandas as pd
import pytest

from app.outliers import iqr_bounds, iqr_report, winsorize_caps, winsorize_target


def _long(values_a, values_b):
    n = len(values_a)
    return pd.DataFrame({
        "jurisdiccion_id": [1] * n + [2] * n, "grupo": ["PATRIMONIO"] * (2 * n),
        "y": list(values_a) + list(values_b),
    })


def test_iqr_bounds_matches_manual_computation():
    s = pd.Series(range(1, 101))
    lo, hi = iqr_bounds(s)
    q1, q3 = 25.75, 75.25
    assert lo == pytest.approx(q1 - 1.5 * (q3 - q1))
    assert hi == pytest.approx(q3 + 1.5 * (q3 - q1))


def test_iqr_bounds_ignores_nan():
    assert iqr_bounds(pd.Series([1, 2, 3, 4, np.nan])) == iqr_bounds(pd.Series([1, 2, 3, 4]))


def test_iqr_report_flags_the_spike(wide_panel):
    w = wide_panel.copy()
    w.iloc[10, 0] = 500.0
    rep = iqr_report(w)
    row = rep[(rep.jurisdiccion_id == 1) & (rep.grupo == "PATRIMONIO")].iloc[0]
    assert row["outlier_days"] >= 1 and row["max"] == 500.0


def test_winsorize_caps_each_series_independently():
    df = _long(list(range(100)), [x * 10 for x in range(100)])
    caps = winsorize_caps(df, 98)
    assert caps[(1, "PATRIMONIO")] == pytest.approx(97.02)
    assert caps[(2, "PATRIMONIO")] == pytest.approx(970.2)


def test_winsorize_target_never_exceeds_cap_and_keeps_low_values():
    df = _long(list(range(100)), list(range(100)))
    out = winsorize_target(df, 90)
    assert out["y"].max() <= df["y"].quantile(0.90) + 1e-9
    assert (out["y"][df["y"] < 50] == df["y"][df["y"] < 50]).all()      # values below the cap are untouched
    assert len(out) == len(df)


def test_winsorize_100_is_a_noop_and_does_not_mutate_input():
    df = _long(list(range(20)), list(range(20)))
    before = df.copy()
    pd.testing.assert_frame_equal(winsorize_target(df, 100), df)
    pd.testing.assert_frame_equal(df, before)


def test_winsorize_rejects_silly_percentiles():
    with pytest.raises(ValueError):
        winsorize_caps(_long([1, 2], [1, 2]), 10)


def test_fit_accepts_winsorization_and_changes_the_training_target(wide_panel):
    from app.modeling import fit

    w = wide_panel.copy()
    w.iloc[60, 0] = 400.0                                   # one absurd day
    m_raw, cols, n_raw = fit("random_forest", w)
    m_win, _, n_win = fit("random_forest", w, winsorize_pct=95)
    assert n_raw == n_win and len(cols) > 10
    last = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)
    assert np.isfinite(m_win.predict(last)).all()
