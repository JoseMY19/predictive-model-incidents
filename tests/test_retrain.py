import json
import os
import time

import pandas as pd
import pytest

from app import retrain
from app.retrain import RetrainLocked, lock, promote, quality_gate, run, should_retrain

D = pd.Timestamp


# ------------------------------------------------------------------ decision rules
def test_should_retrain_when_forced_or_no_model():
    assert should_retrain(D("2026-10-01"), D("2026-10-02"), 7, force=True) == (True, "forced")
    assert should_retrain(None, D("2026-10-02"), 7)[0] is True


def test_should_retrain_only_with_enough_new_days():
    assert should_retrain(D("2026-10-01"), D("2026-10-08"), 7)[0] is True       # exactly 7 days
    go, why = should_retrain(D("2026-10-01"), D("2026-10-05"), 7)
    assert go is False and "4 new complete day" in why


def _spatial(top5):
    return {"elegido": {"todos_top5pct": top5}}


def test_gate_accepts_a_good_model():
    ok, reasons = quality_gate({"mejora_vs_baseline_pct": 4.9}, _spatial(0.42), _spatial(0.40))
    assert ok and reasons == []


@pytest.mark.parametrize("metrics, spatial, old, fragment", [
    ({"mejora_vs_baseline_pct": -2.0}, _spatial(0.42), None, "worse than the naive baseline"),
    ({"mejora_vs_baseline_pct": None}, _spatial(0.42), None, "no valid gain"),
    ({"mejora_vs_baseline_pct": 3.0}, _spatial(0.10), None, "below the floor"),
    ({"mejora_vs_baseline_pct": 3.0}, _spatial(0.30), _spatial(0.42), "dropped"),
    ({"mejora_vs_baseline_pct": 3.0}, {}, None, "no valid top-5"),
    ({"mejora_vs_baseline_pct": float("nan")}, _spatial(0.42), None, "no valid gain"),
])
def test_gate_rejects_bad_models(metrics, spatial, old, fragment):
    ok, reasons = quality_gate(metrics, spatial, old)
    assert not ok and any(fragment in r for r in reasons)


def test_gate_allows_small_spatial_fluctuation():
    assert quality_gate({"mejora_vs_baseline_pct": 1.0}, _spatial(0.38), _spatial(0.42))[0]     # -4 points < 5 allowed


# ------------------------------------------------------------------ lock
def test_lock_is_exclusive_and_released(tmp_path):
    p = tmp_path / ".lock"
    with lock(p):
        with pytest.raises(RetrainLocked):
            with lock(p):
                pass
    assert not p.exists()
    with lock(p):                                                                   # free again
        pass


def test_stale_lock_is_taken_over(tmp_path):
    p = tmp_path / ".lock"
    p.write_text("crashed")
    old = time.time() - 10_000
    os.utime(p, (old, old))
    with lock(p, stale_seconds=3600):
        assert p.exists()


# ------------------------------------------------------------------ swap
def _write_set(folder, tag, gain=4.0, top5=0.42, until="2026-10-01"):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.joblib").write_bytes(tag.encode())
    (folder / "metrics.json").write_text(json.dumps({"mejora_vs_baseline_pct": gain, "ultima_fecha_datos": until, "modelo": tag}))
    (folder / "spatial_metrics.json").write_text(json.dumps({"elegido": {"todos_top5pct": top5}, "tag": tag}))


def test_promote_swaps_files_and_keeps_a_backup(tmp_path):
    live, stage = tmp_path / "models", tmp_path / "models" / ".staging"
    _write_set(live, "old")
    _write_set(stage, "new")
    promote(stage, live)
    assert (live / "model.joblib").read_bytes() == b"new"
    assert (live / "previous" / "model.joblib").read_bytes() == b"old"
    assert json.loads((live / "metrics.json").read_text())["modelo"] == "new"
    assert not (stage / "model.joblib").exists()                                    # moved, not copied


# ------------------------------------------------------------------ full run() with a fake pipeline
class Fake:
    """Pipeline double: writes a staged artifact set and records that it was called."""
    def __init__(self, gain=5.0, top5=0.45, until="2026-10-20", boom=False):
        self.kw, self.calls, self.boom = dict(gain=gain, top5=top5, until=until), 0, boom

    def __call__(self, staging):
        self.calls += 1
        if self.boom:
            raise ValueError("training exploded")
        _write_set(staging, "new", **self.kw)


def _run(tmp_path, pipeline, **kw):
    return run(model_dir=tmp_path / "models", refresh=lambda: None, pipeline=pipeline,
               latest_day_fn=lambda: D("2026-10-20"), **kw)


def _live_tag(tmp_path):
    return (tmp_path / "models" / "model.joblib").read_bytes().decode()


def _history(tmp_path):
    return json.loads((tmp_path / "models" / "retrain_history.json").read_text())["runs"]


def test_run_skips_when_there_are_few_new_days(tmp_path):
    _write_set(tmp_path / "models", "old", until="2026-10-18")
    fake = Fake()
    assert _run(tmp_path, fake) == 0 and fake.calls == 0
    assert _live_tag(tmp_path) == "old" and _history(tmp_path)[-1]["outcome"] == "skipped"


def test_run_promotes_a_good_new_model(tmp_path):
    _write_set(tmp_path / "models", "old", until="2026-10-01")
    assert _run(tmp_path, Fake()) == 0
    assert _live_tag(tmp_path) == "new"
    assert not (tmp_path / "models" / ".staging").exists() and not (tmp_path / "models" / ".retrain.lock").exists()
    assert _history(tmp_path)[-1]["outcome"] == "promoted"


def test_run_rejects_a_worse_model_and_keeps_the_current_one(tmp_path):
    _write_set(tmp_path / "models", "old", until="2026-10-01")
    assert _run(tmp_path, Fake(gain=-3.0)) == 1
    assert _live_tag(tmp_path) == "old"
    h = _history(tmp_path)[-1]
    assert h["outcome"] == "rejected" and "worse than the naive baseline" in h["reason"][0]


def test_run_error_leaves_the_live_model_untouched_and_releases_the_lock(tmp_path):
    _write_set(tmp_path / "models", "old", until="2026-10-01")
    assert _run(tmp_path, Fake(boom=True)) == 2
    assert _live_tag(tmp_path) == "old" and not (tmp_path / "models" / ".retrain.lock").exists()
    assert "training exploded" in _history(tmp_path)[-1]["reason"]


def test_dry_run_evaluates_but_does_not_swap(tmp_path):
    _write_set(tmp_path / "models", "old", until="2026-10-01")
    assert _run(tmp_path, Fake(), dry_run=True) == 0
    assert _live_tag(tmp_path) == "old" and _history(tmp_path)[-1]["outcome"] == "dry-run-ok"


def test_force_retrains_even_without_new_days(tmp_path):
    _write_set(tmp_path / "models", "old", until="2026-10-20")
    fake = Fake()
    assert _run(tmp_path, fake, force=True) == 0 and fake.calls == 1 and _live_tag(tmp_path) == "new"


def test_run_does_nothing_when_another_run_holds_the_lock(tmp_path):
    _write_set(tmp_path / "models", "old", until="2026-10-01")
    (tmp_path / "models" / ".retrain.lock").write_text("busy")
    fake = Fake()
    assert _run(tmp_path, fake) == 0 and fake.calls == 0 and _live_tag(tmp_path) == "old"


def test_first_run_without_any_model_trains(tmp_path):
    fake = Fake()
    assert _run(tmp_path, fake) == 0 and fake.calls == 1 and _live_tag(tmp_path) == "new"


def test_staging_paths_are_restored_after_the_pipeline():
    from app import config
    before = (config.MODEL_FILE, config.METRICS_JSON, config.SPATIAL_JSON)
    with pytest.raises(RuntimeError):
        with retrain._staged_paths(config.MODEL_DIR / ".x"):
            assert config.MODEL_FILE.parent.name == ".x"
            raise RuntimeError
    assert (config.MODEL_FILE, config.METRICS_JSON, config.SPATIAL_JSON) == before
