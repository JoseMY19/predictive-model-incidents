import json
import os
import time

import joblib
import pandas as pd
import pytest

from app import config
from app.live import Live, artifact_stamp, is_stable


@pytest.fixture
def live(tmp_path, monkeypatch):
    """A Live object whose artifacts live in a temp folder (no database is touched: only reload logic runs)."""
    paths = {n: tmp_path / n for n in ("model.joblib", "metrics.json", "spatial_metrics.json", "catalog.json")}
    monkeypatch.setattr(config, "MODEL_FILE", paths["model.joblib"])
    monkeypatch.setattr(config, "METRICS_JSON", paths["metrics.json"])
    monkeypatch.setattr(config, "SPATIAL_JSON", paths["spatial_metrics.json"])
    monkeypatch.setattr(config, "CATALOG_JSON", paths["catalog.json"])

    def write(tag):
        joblib.dump({"kind": tag, "model": None, "cols": []}, paths["model.joblib"])
        paths["spatial_metrics.json"].write_text(json.dumps({"parametros": {"tag": tag}}))
        paths["catalog.json"].write_text(json.dumps({"grupos": [tag]}))
        paths["metrics.json"].write_text(json.dumps({"entrenado": tag}))                     # swapped last, like the job
        old = time.time() - 30                                                              # make it "stable" (not mid-write)
        os.utime(paths["metrics.json"], (old, old))

    write("v1")
    obj = Live.__new__(Live)
    obj.reload_error, obj.model_loaded_at, obj._force_forecast, obj._sig = None, None, False, ("x",)
    obj._load_artifacts()
    obj.write = write
    obj.paths = paths
    return obj


def test_stamp_and_stability_helpers(tmp_path):
    f = tmp_path / "a.json"
    assert artifact_stamp(f) is None and is_stable(f) is False
    f.write_text("{}")
    assert artifact_stamp(f) is not None
    assert is_stable(f, min_age_s=60) is False                       # just written
    assert is_stable(f, min_age_s=60, now=time.time() + 120) is True


def test_no_reload_when_nothing_changed(live):
    assert live.check_reload() is False and live.metrics["entrenado"] == "v1"


def test_reload_picks_up_a_swapped_model_and_forces_a_rebuild(live):
    live.write("v2")
    assert live.check_reload() is True
    assert live.metrics["entrenado"] == "v2" and live.params == {"tag": "v2"} and live._art["kind"] == "v2"
    assert live._base_catalog == {"grupos": ["v2"]}
    assert live._sig is None and live._force_forecast is True and live.reload_error is None
    assert isinstance(live.model_loaded_at, pd.Timestamp)
    assert live.check_reload() is False                                # same files: no second reload


def test_reload_waits_while_the_file_is_still_being_written(live):
    live.write("v2")
    os.utime(live.paths["metrics.json"], None)                         # touched just now -> maybe mid-write
    assert live.check_reload() is False and live.metrics["entrenado"] == "v1"


def test_corrupt_artifacts_keep_the_old_model_and_do_not_retry_forever(live):
    live.paths["metrics.json"].write_text("{ not json")
    old = time.time() - 30
    os.utime(live.paths["metrics.json"], (old, old))
    assert live.check_reload() is False
    assert live.metrics["entrenado"] == "v1" and live.reload_error and "JSONDecodeError" in live.reload_error
    assert live._art["kind"] == "v1"                                   # consistent old set still served
    assert live.check_reload() is False and live._stamp == artifact_stamp(live.paths["metrics.json"])
