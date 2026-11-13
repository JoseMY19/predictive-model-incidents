# Predictive Incidence – backend

Short-term **incident forecasting and hotspot detection** for the Serenazgo (municipal security) service of
San Juan de Lurigancho (SJL), Lima. It cleans the historical incident records, trains a temporal model
(incidents per jurisdiction / type / day) and a spatial model (200 m cells), and serves both through a
FastAPI service that can follow a live incident feed. A React front-end (separate folder) draws the map and the forecast.

> The spatial model answers *"where should we verify next?"*: it ranks zones and, for each one, says which kind of
> incident is most likely, at what time of day, and which camera is closest to check.

```
dump / production API ─▶ PostgreSQL public.*  (source, kept untouched)
                              │ clean   (auditable rules, every decision has a reason)
                              ▼
                        analytics.incidencias_limpias ◀── live rows (incremental cleaning)
                              │ etl                                │
                              ▼                                    ▼
          temporal model + spatial model ───────────────▶ FastAPI (/api/*) ──▶ front-end (map + forecast)
```

## Quick start

Requirements: Python 3.11+, PostgreSQL 15+ (client tools: `pg_restore`), a backup of the incident database.

```powershell
cd backend
python -m pip install -r requirements.txt
copy .env.example .env                                  # then edit it (never commit .env)

.\scripts\db_start.ps1                                  # optional: dedicated local PostgreSQL on :5433 (data in pgdata/)

python -m app.restore --dump "C:\path\to\backup"        # 1) load the dump into public.*
python -m app.clean                                     # 2) cleaning rules -> analytics.*
python -m app.etl                                       # 3) daily panel + geo points for the models
python -m app.train                                     # 4) temporal model (rolling-window backtest)
python -m app.spatial_eval                              # 5) spatial model (backtest + tuned parameters)
python -m uvicorn app.api:app --port 8000               # 6) API (Swagger UI at /docs)
```

To use your own PostgreSQL instead of the dedicated one, set `DATABASE_URL` (environment variable or `.env`).
Front-end: `cd ../frontend && npm install && npm run dev` (the Vite dev server proxies `/api` to `:8000`;
override with `VITE_API_TARGET`).

## What the data looked like (audit summary)

The raw table is **not** model-ready. `app/clean.py` applies explicit, countable rules; nothing is deleted from `public.*`.

| Finding | Handling |
|---|---|
| Only the Serenazgo unit has a consistent history | model restricted to it |
| Gaps / ramp-up periods (see `VALID_RANGES` in `config.py`) | excluded from training and evaluation |
| Jurisdiction labels of 2025 are a **cyclic permutation** of the real ones (label vs. polygon match 15 % in 2025, 99.3 % in 2026) | learned from coordinates and re-mapped (validated against 2026 data) |
| Placeholder coordinate used by operators | nulled |
| Double submissions, historical duplicates, out-of-area points, wrong dates, subtype ≠ type | flagged / excluded with the reason stored |
| History loaded with a default hour | hour of day only trusted for the new system |

Daily counts are over-dispersed (≈ 4.7× Poisson), so there is a practical ceiling: an *oracle* that knows the
exact weekly pattern still misses ≈ 30 % per jurisdiction-day. What is missing is operational information
(shift roster, planned operations) rather than more modelling.

## Models and results

**Temporal** (jurisdiction × incident group × day, 7-day horizon, 3 rolling 8-week windows):

| Model | MAE | Note |
|---|---|---|
| Baseline: 7-day mean | 11.94 | |
| Random Forest / HistGradientBoosting (Poisson) | 11.71 / 11.97 | |
| **Ensemble RF + exponential smoothing** | **11.35** | −4.9 % vs. baseline; WAPE 35.2 % daily, 20.1 % weekly |

Hyper-parameters of the production models were tuned by hand on the same windows, so that gain is slightly optimistic; see the
holdout-based search in *Analysis toolkit* below.

**Spatial** (200 m cells, 21-day half-life, 16-week backtest): the top 5 % of cells capture ≈ 42 % of the next incidents
(static ranking: 38 %), the top 10 % ≈ 57 %; the 20 pinned zones capture ≈ 21 % (chance ≈ 1.5 %).

**Live replay** (model trained only on data before the cut-off, zones frozen, ≈ 1 500 real incidents afterwards):
59–66 % fell in the top 10 % of cells (chance 10 %) and ≈ 20–24 % in the 20 pinned zones.
`app/simulate.py` replays that scenario; `/api/live/status` keeps measuring it with real arrivals.

## Analysis toolkit

Four modules cover the usual data-science workflow. They read the aggregated panel only (no personal data) and write to the
git-ignored `reports/` folder.

| Command | What it does |
|---|---|
| `python -m app.eda` | Univariate / bivariate / multivariate analysis, six charts and data-generated insights (`reports/eda/EDA.md`) |
| `python -m app.outliers` | IQR outlier report per series (jurisdiction × group) |
| `python -m app.compare [--winsorize 98]` | Ridge, KNN, Random Forest, Extra Trees, Gradient Boosting, XGBoost, the ensemble and two baselines on the same rolling windows: MAE, RMSE, R², WAPE |
| `python -m app.tune [--models ...] [--max-combos N]` | Time-aware hyper-parameter search; the newest window is a holdout the search never sees |

Two rules differ from a classic tabular workflow: there is **no random train/test split** (it would train on days that come
after the days it is scored on), and winsorization is applied to the **training target only**, with caps computed from training
rows, while evaluation always uses raw counts.

**Model comparison** (3 rolling windows of 56 days, 7-day horizon, jurisdiction-day level):

| Model | MAE | RMSE | R² | vs. baseline |
|---|---|---|---|---|
| **Ensemble RF + exponential smoothing** | **11.35** | 15.02 | 0.413 | −4.9 % |
| Ridge regression | 11.55 | 15.44 | 0.395 | −3.2 % |
| XGBoost (Poisson) | 11.64 | 15.43 | 0.385 | −2.5 % |
| Random Forest | 11.71 | 15.55 | 0.378 | −1.9 % |
| Extra Trees | 11.83 | 15.72 | 0.369 | −0.9 % |
| Baseline: 7-day mean | 11.94 | 15.66 | 0.358 | 0 % |
| Gradient Boosting (Poisson) | 11.97 | 15.75 | 0.368 | +0.3 % |
| K-Nearest Neighbors | 12.24 | 16.43 | 0.320 | +2.6 % |

What this says:

- The ensemble is still the best model, but the spread between sensible models is small (≈ 5 % of MAE). That matches the
  practical ceiling measured in the audit: better models will not remove the unexplained day-to-day variance.
- A plain **linear model (Ridge) beats Random Forest and Extra Trees**: the signal is mostly level + weekly pattern, which
  the lag/rolling features already expose.
- **Winsorizing the training target (P98) did not help** – it was neutral for the ensemble and worse for every other model
  (e.g. Random Forest 11.71 → 11.87), because the extreme days are real demand, not errors. It stays off by default.
- KNN is the weakest (distance in 40+ mixed features is not informative here).

**Hyper-parameter search** (`app.tune`, 6 random combinations per model, tuned on the two older windows, scored on the newest
window the search never saw):

| Model | MAE defaults (holdout) | MAE best combination (holdout) |
|---|---|---|
| Random Forest | 10.203 | 10.203 |
| XGBoost (Poisson) | 10.162 | 10.191 |
| Ridge regression | 10.042 | 10.042 |
| Gradient Boosting (Poisson) | 10.211 | 10.247 |

The "best" combinations looked 0.2–0.5 % better on the tuning windows and were equal or slightly worse on the holdout: the
current defaults are already at the optimum, and what looked like a gain was selection noise. **Tuning is not the lever** – the
remaining error is variance the features cannot explain (see *What would raise accuracy*).

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health`, `/api/meta` | status, catalogue, model quality |
| `GET /api/history?days=` | real daily counts |
| `GET /api/forecast?horizon=` | expected incidents per jurisdiction/group/day |
| `GET /api/hotspots?dias=&grupo=&top=&jurisdiccion_id=` | ranked zones with likely type, time slot, risk level and nearest camera |
| `GET /api/capas` | jurisdiction polygons and known cameras for drawing the map |
| `GET /api/live/status`, `/api/live/recientes` | sync state and incidents that arrived since start-up |

The service has **no authentication yet** – keep it on a trusted network or put it behind a gateway.

## Live data

`app/remote.py` polls `GET {INCIDENCIAS_API_URL}/incidencias?unidadId=1` every 10 s (read-only, newest-first until it meets
rows it already has) and copies the new rows into the local `public.incidencias` in one transaction. **Personal data
(names, phones, addresses) is never copied.** Cleaning, models and map are unchanged. Credentials go in `.env`
(`INCIDENCIAS_API_USER` / `INCIDENCIAS_API_PASSWORD`; the token is renewed automatically). Limitation: it detects *new*
incidents only, not later edits to rows already copied.

## Cameras

Camera positions are sensitive and are **not in this repository** (see `geo/README.md`). To import the municipal cameras
from the Sivecam API, set `SIVECAM_API_URL/USER/PASSWORD` and run `python -m app.camaras_remote`, or save the response of
`GET /municipal?page=0` from your browser and run `python -m app.camaras_remote --archivo response.json`.
Neighbourhood cameras are detected and stored separately, keeping only position, brand and mode.

## Contributing / Git workflow

- **Branches:** `main` is always releasable; `develop` integrates finished work; every change lives in a short-lived branch cut
  from `develop` and named `feature/<topic>`, `fix/<topic>`, `chore/<topic>` or `docs/<topic>`.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) in English, small and focused
  (`feat: ...`, `fix: ...`, `refactor: ...`, `test: ...`, `docs: ...`, `chore: ...`).
- **Pull requests:** feature branch -> `develop` (with the PR template), then `develop` -> `main` for a release.
  Nothing is pushed directly to `main`.
- **Tests:** `pip install -r requirements-dev.txt && python -m pytest`. Tests use synthetic data only.
- **Never commit** `.env`, real incident data, models, reports derived from real data, or camera locations (see `.gitignore`).

## Privacy and security

- Secrets live only in `.env` (git-ignored). `.env.example` lists every variable.
- `data/`, `models/`, `pgdata/`, dumps and camera files are git-ignored: they are derived from real incident records.
- The pipeline reads only the columns it needs; temporary files with personal data are not kept.

## Project layout

```
app/
  config.py        paths, valid periods, incident groups, backtest settings
  db.py            PostgreSQL access (+ .env reader)
  restore.py       pg_restore wrapper                    clean.py     auditable cleaning rules (batch + incremental)
  etl.py           daily panel / geo points              features.py  calendar + lag features
  modeling.py      RF, ExtraTrees, HGB, smoothing, ensemble, backtest      train.py   model selection
  spatial.py       200 m cells, hotspot scoring, cameras                   spatial_eval.py   spatial backtest/tuning
  live.py          background sync + live state          remote.py    production API sync
  camaras_remote.py  camera importer                     simulate.py  replay a past cut-off in "real time"
  outliers.py      IQR report + winsorization             compare.py   model comparison report
  tune.py          time-aware hyper-parameter search       eda.py       exploratory analysis + insights
  api.py           FastAPI service
tests/             pytest suite on synthetic data (python -m pytest)
geo/               jurisdiction polygons
scripts/db_start.ps1
```

Code comments and docstrings are in Spanish.

## What would raise accuracy

1. Shift roster and operations calendar (the only variable that explains the unexplained day-to-day variance).
2. More history (Nov 2025 – Feb 2026 are missing; Police and Fire units).
3. Scheduled re-training and drift monitoring (the live service does **not** retrain by itself).
4. Operator feedback on each flagged zone (was it verified? did something happen?).
5. Authentication on the API.
