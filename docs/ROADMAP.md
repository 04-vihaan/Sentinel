# Implementation roadmap

The MVP is a runnable foundation. The phases below are proposed work, not capabilities already delivered. Timing assumes one full-time engineer with part-time domain and operations support; real data access and label quality may dominate the schedule.

## Phase 0 — Delivered demonstration

**Scope:** Synthetic data; shared feature contract; logistic regression and forest; purged chronological evaluation; local artifacts and experiment reports; API and dashboard; fault scenarios; delayed outcomes; explicit deployment gate and rollback; unit/integration tests; Docker and CI configuration.

**Acceptance:** Start from an empty data directory, generate artifacts, serve a valid prediction, reject malformed windows, replay faults, inspect versioned evidence, approve a passing candidate and restore the previous version. The included test suite exercises these behaviors. Container and hosted CI execution still need verification in those environments.

## Phase 1 — Real data and failure definitions (1–2 weeks)

- Select a licensed dataset or partner export with equipment identity, event time, sensor units, maintenance events and repair history. Document legal use and redistribution constraints.
- Define failure taxonomy, usable warning horizon, observation cadence, censoring and maintenance interventions with domain experts.
- Build an import adapter and deterministic unit conversion. Quarantine invalid readings; report missingness, duplicates, late arrivals and sensor coverage by asset.
- Separate source snapshots, cleaned observations, feature datasets and labels. Record hashes, schema versions, source lineage and transformation code revision in a manifest.
- Add an equipment catalog and avoid treating simulator display labels as real asset metadata.

**Deliverables:** Data dictionary, ingestion command, quality report, versioned dataset manifest and failure-label specification.

**Acceptance:** A repeated import is idempotent; invalid records are inspectable; two independent reviewers can reproduce sampled labels and point-in-time windows without consulting model outputs.

## Phase 2 — Reliable temporal modeling (1–2 weeks)

- Add rolling-origin evaluation and held-out equipment splits, with horizon purging and explicit repair/downtime rules.
- Compare constant/prevalence and simple engineering-rule baselines, logistic regression, random forest and one boosted-tree model.
- Restrict hyperparameter search to training/validation folds. Add grouped bootstrap uncertainty estimates by equipment/event.
- Calibrate probabilities on a dedicated temporal calibration partition; evaluate calibration plots and Brier score.
- Choose thresholds from maintenance capacity, false alarm cost and lead-time utility. Evaluate event-level precision/recall, alerts per machine-day and actionable warning time.
- Add time-window leakage regression checks and feature parity tests with known fixtures.

**Deliverables:** Experiment table, model card, calibration report and frozen final holdout protocol.

**Acceptance:** A proposed model beats agreed operational baselines on predeclared metrics and uncertainty bounds; no test outcome influences training, threshold selection or deployment decisions.

## Phase 3 — Durable ingestion and serving (2 weeks)

- Replace browser-driven replay with a durable worker and broker/queue; support event IDs, watermarks, late arrivals and backpressure.
- Move transactional state to PostgreSQL. Add unique `(machine, window_end, feature_version, model_version, source_event)` keys and atomic checkpointing.
- Put immutable data/artifacts in object storage and track runs/models in MLflow or an equivalent registry. Use DVC or manifest-based versioning for reproducible datasets.
- Separate training from API processes. Queue jobs with retries, cancellation, resource limits and visible failure records.
- Cache online features with a documented consistency model. Enforce training/serving parity and retain raw provenance.
- Add external label ingestion with source IDs, revisions, availability time and explicit maturation rules.

**Deliverables:** Worker service, migration scripts, idempotent event contracts, lineage links and load-test report.

**Acceptance:** Replaying the same input twice creates no duplicate prediction or alert; killing a worker mid-batch loses no event; serving meets an agreed latency/throughput SLO while training runs.

## Phase 4 — Monitoring and controlled updates (1–2 weeks)

- Instrument latency histograms, endpoint error rates, queue lag, data freshness and artifact load failures. Add Prometheus/Grafana dashboards and tested alert routes.
- Monitor missingness and feature/prediction distributions by equipment type and operating regime. Require minimum effective sample sizes and sustained windows to reduce false drift alerts.
- Report outcome metrics only for mature cohorts and expose sample size, censoring and label coverage. Treat label delay as a separate health signal.
- Allow sustained drift to queue candidate training with cooldowns and idempotent trigger IDs. Never let drift bypass validation or approval.
- Compare candidates on a frozen reference cohort and recent mature holdout; record gate evidence, reviewer identity and reason.
- Add shadow/canary deployment, health-based rollback and feature/model compatibility checks.

**Deliverables:** Monitoring configuration, retraining workflow, approval record schema and operational runbooks.

**Acceptance:** Sensor outage, distribution shift, delayed labels and API failure each fire the intended signal; a bad candidate is blocked; approval and rollback are auditable and tested under concurrent requests.

## Phase 5 — Operator workflow and security (1–2 weeks)

- Add SSO, role-based access, scoped API credentials, request limits, CSRF protections appropriate to session design, TLS and secret management.
- Deduplicate window alerts into incidents; define cooldown, escalation, acknowledgement and maintenance feedback flows.
- Add asset filtering, historical version comparisons, explanation uncertainty and exportable investigation records.
- Sign trusted model artifacts; restrict deserialization and isolate training identities. Add dependency scanning and supply-chain provenance.
- Exercise backup/restore and retention policies; establish privacy and access policies for maintenance records.

**Deliverables:** Authenticated operator UI, incident lifecycle, threat model, deployment guide and restore drill results.

**Acceptance:** Unauthorized users cannot promote models or change data; one deterioration episode produces the intended incident count; every operator action records an identity; backup restoration meets the agreed recovery objective.

## Phase 6 — Prospective pilot (2–4 weeks minimum)

Run in shadow mode against new, naturally occurring observations. Predeclare success criteria and do not modify them after seeing results. Review false alarms, missed events, lead time and maintenance impact with operators. Expand beyond a small asset cohort only after enough failure events have accumulated to support a meaningful estimate.

**Deliverables:** Prospective evaluation, failure analysis, cost/benefit estimate and a go/no-go decision.

**Acceptance:** Evidence comes from a genuinely subsequent time period; both model quality and operational reliability meet the agreed requirements. Synthetic/offline metrics alone cannot satisfy this milestone.

## Key design decisions to resolve

| Decision | Why it matters | Owner |
|---|---|---|
| Failure horizon and downtime policy | Defines target and usable warnings | Maintenance + data science |
| Incident cost and alert capacity | Determines threshold and gates | Operations |
| Outcome availability and corrections | Prevents optimistic monitoring | Data engineering |
| Cadence and late-event tolerance | Determines event-time processing | Platform engineering |
| Model promotion/rollback authority | Defines access and audit workflow | Platform + operations |
| Storage/availability SLOs | Determines infrastructure and cost | Platform engineering |

A strong portfolio demonstration should include a scripted normal→outage→drift→delayed-label→promotion→rollback sequence, saved model reports, architecture rationale, test evidence and an honest discussion of the difference between synthetic evaluation and field validation.
