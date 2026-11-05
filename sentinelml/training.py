"""Deterministic synthetic dataset, purged temporal evaluation, local artifacts."""
import hashlib
import json
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .features import Reading, build_features, FEATURE_NAMES, FEATURE_VERSION, WINDOW, HORIZON

SEED = 42
HOURS = 720
MACHINES = 12
TRAIN_END = 432
VALID_END = 576

def generate_dataset():
    rng = np.random.default_rng(SEED)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    for machine in range(MACHINES):
        wear = rng.uniform(.05, .8)
        for hour in range(HOURS):
            load = float(np.clip(62 + 13 * np.sin(hour / 19 + machine) + rng.normal(0, 5), 25, 95))
            wear += .009 + load / 14000 + rng.uniform(-.003, .003)
            failed = wear >= 1
            rows.append({'machine_id': f'EQ-{machine+1:03d}', 'hour': hour,
                'timestamp': (start + timedelta(hours=hour)).isoformat(),
                'temperature': round(48 + 31 * wear + .12 * load + rng.normal(0, 2), 3),
                'vibration': round(max(.1, 1.1 + 6 * wear**3 + rng.normal(0, .25)), 3),
                'pressure': round(112 - 23 * wear + rng.normal(0, 2), 3),
                'load': round(load, 3), 'failure': int(failed)})
            if failed:
                wear = rng.uniform(0, .1)
    return rows

def as_reading(row):
    return Reading(**{k: row[k] for k in ('timestamp', 'temperature', 'vibration', 'pressure', 'load')})

def make_examples(rows):
    x, y, meta = [], [], []
    for machine in sorted({r['machine_id'] for r in rows}):
        series = [r for r in rows if r['machine_id'] == machine]
        for i in range(WINDOW - 1, len(series) - HORIZON):
            future = series[i+1:i+1+HORIZON]
            x.append(build_features([as_reading(r) for r in series[i-WINDOW+1:i+1]]))
            y.append(int(any(r['failure'] for r in future)))
            meta.append({'machine_id': machine, 'hour': i, 'event_hour': next((r['hour'] for r in future if r['failure']), None)})
    return np.array(x), np.array(y), meta

def metrics(y, probabilities, threshold, meta):
    alerts = probabilities >= threshold
    covered = {}
    events = { (m['machine_id'], m['event_hour']) for m in meta if m['event_hour'] is not None }
    for alert, m in zip(alerts, meta):
        if alert and m['event_hour'] is not None:
            key = (m['machine_id'], m['event_hour'])
            covered[key] = max(covered.get(key, 0), m['event_hour'] - m['hour'])
    machines = len({m['machine_id'] for m in meta}) or 1
    return {'precision': float(precision_score(y, alerts, zero_division=0)),
        'recall': float(recall_score(y, alerts, zero_division=0)),
        'f1': float(f1_score(y, alerts, zero_division=0)),
        'average_precision': float(average_precision_score(y, probabilities)),
        'brier_score': float(brier_score_loss(y, probabilities)),
        'event_recall': len(covered) / max(1, len(events)),
        'mean_warning_lead_hours': float(np.mean(list(covered.values()))) if covered else 0.,
        'false_alarm_windows_per_machine': float(np.sum(alerts & (y == 0)) / machines),
        'evaluated_windows': len(y), 'failure_events': len(events)}

def train_models(root: Path):
    rows = generate_dataset()
    raw = json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()
    dataset_id = hashlib.sha256(raw).hexdigest()
    (root / 'data').mkdir(parents=True, exist_ok=True)
    (root / 'artifacts').mkdir(parents=True, exist_ok=True)
    (root / 'data' / 'synthetic-v1.json').write_bytes(raw)
    x, y, meta = make_examples(rows)
    hours = np.array([m['hour'] for m in meta])
    # Purge horizon at right boundary: labels cannot cross into next partition.
    train = hours < TRAIN_END - HORIZON
    valid = (hours >= TRAIN_END) & (hours < VALID_END - HORIZON)
    test = hours >= VALID_END
    models = {'baseline-v1': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, random_state=SEED)),
              'forest-v1': RandomForestClassifier(n_estimators=120, max_depth=7, min_samples_leaf=8, random_state=SEED, n_jobs=1)}
    results = []
    for version, model in models.items():
        model.fit(x[train], y[train])
        vp = model.predict_proba(x[valid])[:, 1]
        thresholds = np.linspace(.1, .9, 33)
        threshold = float(max(thresholds, key=lambda t: f1_score(y[valid], vp >= t, zero_division=0)))
        report = {'version': version, 'algorithm': 'Logistic regression' if version.startswith('baseline') else 'Random forest',
            'dataset_id': dataset_id, 'feature_version': FEATURE_VERSION, 'threshold': threshold,
            'validation': metrics(y[valid], vp, threshold, [m for m, keep in zip(meta, valid) if keep]),
            'test': metrics(y[test], model.predict_proba(x[test])[:, 1], threshold, [m for m, keep in zip(meta, test) if keep]),
            'split': {'train': [5, 425], 'validation': [432, 569], 'test': [576, 713], 'purge_hours': HORIZON},
            'training_windows': int(train.sum()), 'seed': SEED}
        artifact = {'model': model, 'reference': np.median(x[train], axis=0),
            'reference_std': x[train].std(axis=0), 'reference_samples': x[train][::10], 'report': report}
        path = root / 'artifacts' / f'{version}.pkl'
        path.write_bytes(pickle.dumps(artifact))
        report['artifact_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        (root / 'artifacts' / f'{version}.json').write_text(json.dumps(report, indent=2))
        results.append(report)
    return results
