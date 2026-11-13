"""Scheduled retraining job: refresh data -> train in a staging folder -> quality gate -> atomic swap.

Why a job and not "every incident": the daily forecast only changes once a day and a model needs weeks of new data to learn
anything. Retraining a few times a week is enough; running it per incident would only burn CPU.

Safety design
* The new model is trained into ``models/.staging``. The live files are replaced only if the quality gate passes, so a failed
  or worse run never touches the model that is serving (that *is* the rollback).
* The previous artifacts are kept in ``models/previous/``.
* ``metrics.json`` is swapped LAST: the live API watches it as the "commit marker" and hot-reloads everything when it changes.
* A lock file prevents two runs at once; a run that finds no/few new days exits early (``--force`` overrides).

Usage:
    python -m app.retrain                 # retrain only if >= 7 new complete days since the last training
    python -m app.retrain --force         # retrain now
    python -m app.retrain --dry-run       # train + evaluate the gate, but do not swap

Exit code: 0 = promoted or skipped, 1 = rejected by the quality gate, 2 = error.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import logging.handlers
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from . import config

LOG_DIR = config.ROOT / "logs"
ARTIFACTS = ("model.joblib", "spatial_metrics.json", "metrics.json")      # metrics.json LAST = commit marker
MIN_NEW_DAYS = 7
LOCK_STALE_SECONDS = 3 * 3600
SPATIAL_FLOOR = 0.25          # top-5 % cells must capture at least this share of next incidents (chance = 5 %)
SPATIAL_MAX_DROP = 0.05       # ...and may not fall more than 5 points below the model being replaced
HISTORY_KEEP = 30

log = logging.getLogger("retrain")


# ------------------------------------------------------------------ decisions (pure, unit-tested)
def should_retrain(trained_until: pd.Timestamp | None, latest_day: pd.Timestamp, min_new_days: int, force: bool = False) -> tuple[bool, str]:
    """Retrain when there is no model, when forced, or when at least `min_new_days` new complete days exist."""
    if force:
        return True, "forced"
    if trained_until is None:
        return True, "no previous model"
    new_days = int((latest_day.normalize() - trained_until.normalize()).days)
    if new_days >= min_new_days:
        return True, f"{new_days} new complete days since the last training"
    return False, f"only {new_days} new complete day(s) (< {min_new_days})"


def quality_gate(new_metrics: dict, new_spatial: dict, old_spatial: dict | None = None) -> tuple[bool, list[str]]:
    """Returns (accepted, reasons for rejection). Temporal MAE of different windows is not comparable, so the temporal
    check is relative to the baseline measured on the SAME windows; the spatial check is absolute and relative to the old model."""
    reasons: list[str] = []
    gain = new_metrics.get("mejora_vs_baseline_pct")
    if gain is None or gain != gain:
        reasons.append("temporal model has no valid gain vs. baseline")
    elif gain < 0:
        reasons.append(f"temporal model is worse than the naive baseline ({gain:+.1f} %)")
    top5 = new_spatial.get("elegido", {}).get("todos_top5pct")
    if top5 is None or top5 != top5:
        reasons.append("spatial model has no valid top-5 % capture")
    else:
        if top5 < SPATIAL_FLOOR:
            reasons.append(f"spatial top-5 % capture {top5:.1%} is below the floor {SPATIAL_FLOOR:.0%}")
        old = (old_spatial or {}).get("elegido", {}).get("todos_top5pct")
        if old is not None and top5 < old - SPATIAL_MAX_DROP:
            reasons.append(f"spatial top-5 % capture dropped from {old:.1%} to {top5:.1%}")
    return (not reasons), reasons


# ------------------------------------------------------------------ lock and file swap
class RetrainLocked(RuntimeError):
    pass


@contextlib.contextmanager
def lock(path: Path, stale_seconds: float = LOCK_STALE_SECONDS):
    """Exclusive lock file. A lock older than `stale_seconds` is considered a crashed run and is taken over."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and time.time() - path.stat().st_mtime > stale_seconds:
        log.warning("removing stale lock %s", path)
        path.unlink()
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RetrainLocked(f"another retraining is running ({path})") from None
    try:
        os.write(fd, f"pid={os.getpid()} started={datetime.now(timezone.utc).isoformat()}".encode())
        os.close(fd)
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


def promote(staging: Path, target: Path) -> None:
    """Back up the live artifacts to target/previous and replace them with the staged ones (atomic per file;
    metrics.json last so a watcher sees a complete set)."""
    prev = target / "previous"
    prev.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS:
        if (target / name).exists():
            shutil.copy2(target / name, prev / name)
    for name in ARTIFACTS:
        os.replace(staging / name, target / name)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _append_history(model_dir: Path, entry: dict) -> None:
    path = model_dir / "retrain_history.json"
    hist = (_read_json(path) or {}).get("runs", [])
    hist.append(entry)
    path.write_text(json.dumps({"runs": hist[-HISTORY_KEEP:]}, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ default pipeline (touches DB and the real modules)
def default_refresh() -> None:
    """Pull the newest incidents from the production API (if configured) and clean them, so training sees today's data."""
    from . import db
    from .clean import clean_new
    from .remote import RemoteSync

    rs = RemoteSync()
    if rs.enabled:
        n = rs.sync()
        log.info("production API sync: %d new rows%s", n, f" (error: {rs.error})" if rs.error else "")
    with db.connect() as conn:
        try:
            log.info("cleaned %d new rows", clean_new(conn))
        except Exception as e:                       # the live service may be cleaning the same rows right now
            log.warning("clean_new skipped: %s", e)


def default_latest_day() -> pd.Timestamp:
    from . import db, etl

    with db.connect() as conn:
        return etl.last_complete_day(conn)


@contextlib.contextmanager
def _staged_paths(staging: Path):
    """Point the training modules at the staging folder (they read config.* at call time)."""
    saved = (config.MODEL_FILE, config.METRICS_JSON, config.SPATIAL_JSON)
    config.MODEL_FILE, config.METRICS_JSON, config.SPATIAL_JSON = (staging / "model.joblib", staging / "metrics.json", staging / "spatial_metrics.json")
    try:
        yield
    finally:
        config.MODEL_FILE, config.METRICS_JSON, config.SPATIAL_JSON = saved


def default_pipeline(staging: Path) -> None:
    from . import etl, spatial_eval, train

    staging.mkdir(parents=True, exist_ok=True)
    with _staged_paths(staging):
        etl.run()                    # data/*.csv + catalog (not served by the API, which reads the database)
        train.main()                 # temporal model + backtest
        spatial_eval.main()          # spatial model parameters + backtest


# ------------------------------------------------------------------ orchestration
def run(force: bool = False, min_new_days: int = MIN_NEW_DAYS, dry_run: bool = False, model_dir: Path | None = None,
        refresh: Callable[[], None] = default_refresh, pipeline: Callable[[Path], None] = default_pipeline,
        latest_day_fn: Callable[[], pd.Timestamp] = default_latest_day) -> int:
    model_dir = Path(model_dir or config.MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry: dict = {"started": started, "dry_run": dry_run}
    try:
        with lock(model_dir / ".retrain.lock"):
            refresh()
            old_m, old_s = _read_json(model_dir / "metrics.json"), _read_json(model_dir / "spatial_metrics.json")
            trained_until = pd.Timestamp(old_m["ultima_fecha_datos"]) if old_m else None
            latest = latest_day_fn()
            go, why = should_retrain(trained_until, latest, min_new_days, force)
            log.info("decision: %s (%s)", "retrain" if go else "skip", why)
            if not go:
                _append_history(model_dir, {**entry, "outcome": "skipped", "reason": why})
                return 0

            staging = model_dir / ".staging"
            shutil.rmtree(staging, ignore_errors=True)
            pipeline(staging)
            new_m, new_s = _read_json(staging / "metrics.json"), _read_json(staging / "spatial_metrics.json")
            if new_m is None or new_s is None or not (staging / "model.joblib").exists():
                raise RuntimeError("the pipeline did not produce all artifacts")
            ok, reasons = quality_gate(new_m, new_s, old_s)
            summary = {"data_until": new_m.get("ultima_fecha_datos"), "model": new_m.get("modelo"), "gain_vs_baseline_pct": new_m.get("mejora_vs_baseline_pct"),
                       "spatial_top5pct": new_s.get("elegido", {}).get("todos_top5pct")}
            if not ok:
                log.error("REJECTED, keeping the current model: %s", "; ".join(reasons))
                _append_history(model_dir, {**entry, "outcome": "rejected", "reason": reasons, **summary})
                return 1
            if dry_run:
                log.info("dry run: gate passed, nothing swapped. %s", summary)
                _append_history(model_dir, {**entry, "outcome": "dry-run-ok", **summary})
                return 0
            promote(staging, model_dir)
            shutil.rmtree(staging, ignore_errors=True)
            log.info("PROMOTED new model. %s", summary)
            _append_history(model_dir, {**entry, "outcome": "promoted", "reason": why, **summary})
            return 0
    except RetrainLocked as e:
        log.warning("%s", e)
        return 0
    except Exception as e:
        log.exception("retraining failed; the current model was not touched")
        with contextlib.suppress(Exception):
            _append_history(model_dir, {**entry, "outcome": "error", "reason": f"{type(e).__name__}: {e}"[:300]})
        return 2


def _setup_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.handlers.RotatingFileHandler(LOG_DIR / "retrain.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    for h in (fh, sh):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="retrain even if there are few new days")
    ap.add_argument("--min-new-days", type=int, default=MIN_NEW_DAYS)
    ap.add_argument("--dry-run", action="store_true", help="train and evaluate the gate but do not swap the live model")
    a = ap.parse_args()
    _setup_logging()
    sys.exit(run(force=a.force, min_new_days=a.min_new_days, dry_run=a.dry_run))


if __name__ == "__main__":
    main()
