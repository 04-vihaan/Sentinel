# SentinelML

A runnable predictive-maintenance platform demonstrating the path from sensor history to monitored, versioned predictions. FastAPI + scikit-learn + SQLite with a responsive browser dashboard. No Node.js, cloud account, or database server required.

**This is a synthetic-data, local demonstration, not an industrial safety system.** It predicts failure within the next six simulated hours for twelve machines. Model scores demonstrate the workflow; they are not evidence of performance on real equipment.

## Run locally

Python 3.13 is the tested version.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
python -m sentinelml
```

Open **http://127.0.0.1:8000**. Interactive API: **http://127.0.0.1:8000/docs**.

First startup generates 8,640 hourly readings, trains two models, writes dataset/model artifacts, initializes SQLite, and replays six hours. Allow around a minute on first startup; subsequent starts reuse persisted artifacts. Replay is paused initially. All sensor times are explicitly simulated UTC event times, separate from request/audit timestamps.

### Docker

```bash
docker compose up --build
```

A named volume preserves artifacts and prediction history. The container runs as a non-root user and binds the host port to localhost. Run exactly one API worker: the MVP serializes state mutations with an in-process lock and uses one SQLite connection. Multiple workers or replicas are unsupported.

## What to try

1. Start replay or advance one hour at a time. Inspect equipment cards, sensor trends, alert history, and the sensor sensitivity explanation.
2. Select **Sensor outage**, advance one hour: three machines become stale and their predictions are withheld. Select **Schema change**: two machines fail input validation.
3. Select **Operating drift**, advance about ten hours: temperature and vibration shifts appear in PSI monitoring. This does not deploy or automatically train a model.
4. Select **Delayed labels (+24h)**: pending labels accumulate; observed-performance metrics only use labels whose arrival time has passed. Returning to normal releases outcomes due under the normal six-hour horizon.
5. Open **Model registry**. Review validation gates and test metrics. **Approve & deploy** switches models only when the candidate passes. **Roll back** restores the previous deployment and adds an audit entry.
6. **Reproduce candidate** reruns training on the immutable original training partition and saves a job artifact. It leaves the deployment unchanged. It does not incorporate replay/test observations or create a newly approvable registry version.

Replay stops at hour 713 so every scored window has a complete future target. To create a separate clean demo, run with an unused directory:

```bash
SENTINEL_HOME=/tmp/sentinel-fresh python -m sentinelml
```

## Included

- Seeded sensor simulator with recurring wear/failure/reset cycles, four sensors and twelve machines.
- Strict schema, finite numeric ranges, timezone-aware timestamps and contiguous hourly windows.
- Shared, versioned training/serving features: last, mean, standard deviation and slope per sensor.
- Logistic regression baseline and random forest, chronologically split with target-horizon purging.
- Validation-selected alert thresholds; precision, recall, F1, average precision, Brier score, event recall, warning lead time, and false alarm windows per machine.
- SHA-256 dataset identity, checksummed model artifacts, JSON experiment reports and SQLite model registry.
- Predictions persisted with raw input window, feature vector/version, model version, threshold, timestamps, explanation and delayed outcome.
- Feature/prediction PSI, withheld/stale data events, `/health`, and basic Prometheus-format service counters at `/metrics`.
- Explicit deployment action, validation gate and persisted rollback history.

## Tests

```bash
python -m pytest -q
```

Tests exercise target semantics, feature timestamp validation, strict prediction input validation, audit fields, outage and schema scenarios, delayed outcomes, approval and rejection gates, rollback, drift without deployment, and a reproducible training job. They train isolated artifacts in a temporary directory. CI configuration is included.

## API overview

| Endpoint | Purpose |
|---|---|
| `GET /health` | Database/model readiness |
| `GET /metrics` | Process request/error/duration counters |
| `GET /api/summary` | Fleet, alerts, models, outcomes, monitoring |
| `POST /api/predict` | Score six hourly readings for a machine |
| `GET /api/machines/{id}/history` | Latest 36 replay observations |
| `POST /api/replay` | Advance `{"steps":1}` (1–24) |
| `POST /api/scenario` | Choose `normal`, `outage`, `schema_change`, `drift`, `delayed_labels` |
| `POST /api/models/{version}/approve` | Approve a candidate that passes gates |
| `POST /api/models/rollback` | Restore the previous active version |
| `POST /api/training` | Synchronous, immutable-data candidate reproduction |

`/api/predict` accepts `machine_id` and `readings`: six objects with `timestamp` (ISO 8601 with timezone), `temperature` (°C), `vibration` (mm/s), `pressure` (psi) and `load` (%). Unknown sensor keys, missing values, invalid ranges and non-hourly windows are rejected. External API predictions are audited but excluded from replay health and outcome metrics; a production label-ingestion API is not implemented.

## Scope and limitations

This MVP implements local equivalents of dataset versioning, experiment tracking and a model registry. It does **not** include MLflow, DVC, Kafka, a feature store, scheduled retraining, authentication, RBAC, external labels, notifications, or distributed deployment. Approval is an explicit local UI/API action, not an authenticated reviewer workflow. Keep it bound to localhost until security and operational controls are added. Training holds the platform lock and temporarily blocks requests that need it.

Probabilities are uncalibrated model estimates; Brier scores are reported. False alarms count individual positive hourly windows, not deduplicated maintenance incidents. Explanations are grouped median-replacement sensitivity, not additive/causal SHAP values. PSI warnings are heuristic and can be noisy on small, correlated samples. Drifted scenarios retain the simulator's original outcomes to represent sensor/operating distribution changes; they do not establish a realistic causal failure model.

SQLite audit records are useful for debugging but are not tamper-proof. Loading pickle artifacts requires a trusted local artifact directory; checksums detect accidental modification, not an attacker who can rewrite both artifact and report. All prediction versions should be retained alongside their pinned runtime. Font loading is optional and falls back to system fonts offline.

See [architecture](docs/ARCHITECTURE.md) and the [implementation roadmap](docs/ROADMAP.md) for system contracts, evaluation definitions, production design and acceptance criteria.
