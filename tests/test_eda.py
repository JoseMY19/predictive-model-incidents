import numpy as np
import pandas as pd
import pytest

from app import eda


def _panel_from(wide: pd.DataFrame) -> pd.DataFrame:
    s = wide.stack([0, 1]).reset_index()
    s.columns = ["fecha", "jurisdiccion_id", "grupo", "n"]
    return s[s["n"] > 0]


@pytest.fixture
def geo() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    hot = pd.DataFrame({"lat": -11.95 + rng.normal(0, 0.0003, 400), "lon": -76.99 + rng.normal(0, 0.0003, 400)})
    cold = pd.DataFrame({"lat": rng.uniform(-12.03, -11.92, 200), "lon": rng.uniform(-77.02, -76.96, 200)})
    g = pd.concat([hot, cold], ignore_index=True)
    g["fecha"] = "2026-06-01"
    g["hora"] = rng.choice([8, 9, 20, 21, 22], len(g), p=[0.1, 0.1, 0.3, 0.3, 0.2]).astype(float)
    return g


@pytest.fixture
def catalog() -> dict:
    return {"jurisdicciones": [{"id": 1, "nombre": "ALPHA"}, {"id": 2, "nombre": "BETA"}]}


def test_overview_counts_days_series_and_incidents(wide_panel):
    ov = eda.overview(wide_panel)
    assert ov["days"] == 120 and ov["series"] == 4
    assert ov["incidents"] == int(wide_panel.to_numpy().sum())
    assert 0 <= ov["zero_share_pct"] <= 100


def test_weekday_profile_averages_to_one(wide_panel):
    prof = eda.weekday_profile(eda.daily_total(wide_panel))
    assert len(prof) == 7 and prof.mean() == pytest.approx(1.0, abs=0.05)


def test_share_by_sums_to_100(wide_panel):
    assert eda.share_by(_panel_from(wide_panel), "grupo").sum() == pytest.approx(100.0)


def test_group_correlation_is_symmetric_with_unit_diagonal(wide_panel):
    c = eda.group_correlation(wide_panel)
    assert np.allclose(np.diag(c), 1.0) and np.allclose(c, c.T)


def test_autocorrelation_is_perfect_for_a_pure_weekly_pattern():
    idx = pd.date_range("2026-01-05", periods=70)
    total = pd.Series(np.tile([10, 12, 11, 13, 20, 25, 9], 10), index=idx, dtype=float)
    assert eda.autocorrelation(total, lags=(7,))[7] == pytest.approx(1.0)


def test_hourly_profile_ignores_untrusted_dates_and_sums_to_100(geo):
    old = geo.copy()
    old["fecha"] = "2025-03-01"
    old["hora"] = 19.0
    p = eda.hourly_profile(pd.concat([geo, old]))
    assert p.sum() == pytest.approx(100.0) and p[19] == 0


def test_spatial_concentration_finds_the_hot_cells(geo):
    sp = eda.spatial_concentration(geo, top_share=0.05)
    assert sp["top_share_pct"] > 30 and sp["top_cells"] >= 1


def test_recent_trend_detects_a_doubling():
    idx = pd.date_range("2026-01-01", periods=56)
    total = pd.Series([10.0] * 28 + [20.0] * 28, index=idx)
    assert eda.recent_trend(total) == pytest.approx(100.0)


def test_recent_trend_is_nan_when_history_is_too_short():
    assert np.isnan(eda.recent_trend(pd.Series(1.0, index=pd.date_range("2026-01-01", periods=10))))


def test_holiday_effect_is_negative_when_holidays_are_quiet():
    idx = pd.date_range("2026-04-27", periods=14)                 # includes 1 May 2026 (national holiday)
    total = pd.Series(100.0, index=idx)
    total[pd.Timestamp("2026-05-01")] = 50.0
    assert eda.holiday_effect(total) == pytest.approx(-50.0)


def test_build_insights_returns_text_for_every_section(wide_panel, geo, catalog):
    wide = wide_panel.copy()
    wide.columns = pd.MultiIndex.from_tuples([(j, g) for j, g in wide.columns])
    ins = eda.build_insights(wide, _panel_from(wide), geo, catalog)
    assert len(ins) >= 5
    assert all(i["title"] and len(i["text"]) > 40 and "evidence" in i for i in ins)
    assert any("ALPHA" in i["text"] or "BETA" in i["text"] for i in ins)
