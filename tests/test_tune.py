import pandas as pd
import pytest

from app.modeling import backtest_windows, make_model
from app.tune import GRIDS, expand_grid, pick_best, split_windows, tune_model


def test_expand_grid_is_the_cartesian_product_in_stable_order():
    g = expand_grid({"a": [1, 2], "b": ["x", "y", "z"]})
    assert len(g) == 6 and g[0] == {"a": 1, "b": "x"} and g[-1] == {"a": 2, "b": "z"}


def test_split_windows_holds_out_the_newest_and_never_overlaps_it():
    w = backtest_windows(pd.Timestamp("2026-10-29"), days=56, folds=3)
    tune, hold = split_windows(w)
    assert hold == w[-1] and len(tune) == 2 and all(b < hold[0] for _, b in tune)


def test_split_windows_needs_two_windows():
    with pytest.raises(ValueError):
        split_windows([(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01"))])


def test_pick_best_takes_lowest_mae_and_first_on_ties():
    assert pick_best([({"a": 1}, 5.0), ({"a": 2}, 4.0), ({"a": 3}, 4.0)]) == ({"a": 2}, 4.0)


def test_make_model_overrides_hyperparameters():
    assert make_model("random_forest", min_samples_leaf=7).min_samples_leaf == 7
    assert make_model("ridge", ridge__alpha=3.0).named_steps["ridge"].alpha == 3.0
    with pytest.raises(ValueError):
        make_model("random_forest", not_a_param=1)


def test_every_grid_parameter_exists_on_its_model():
    pytest.importorskip("xgboost")
    for kind, grid in GRIDS.items():
        valid = make_model(kind).get_params()
        assert set(grid) <= set(valid), (kind, set(grid) - set(valid))


def test_tune_model_end_to_end_on_a_tiny_panel(wide_panel):
    windows = backtest_windows(wide_panel.index[-1], days=14, folds=2)
    r = tune_model("ridge", wide_panel, windows, {"ridge__alpha": [1.0, 100.0]})
    assert r["combinations"] == 2 and r["best_params"]["ridge__alpha"] in (1.0, 100.0)
    assert r["best_mae_tuning"] <= max(x["mae_tuning"] for x in r["all"])
    assert r["holdout_window"][1] == wide_panel.index[-1].strftime("%Y-%m-%d")


def test_max_combos_limits_the_search_reproducibly(wide_panel):
    windows = backtest_windows(wide_panel.index[-1], days=14, folds=2)
    grid = {"ridge__alpha": [0.1, 1.0, 10.0, 100.0]}
    a = tune_model("ridge", wide_panel, windows, grid, max_combos=2)
    b = tune_model("ridge", wide_panel, windows, grid, max_combos=2)
    assert a["combinations"] == 2 and [x["params"] for x in a["all"]] == [x["params"] for x in b["all"]]
