import hashlib
import json
import pickle
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .features import FEATURE_NAMES, FEATURE_VERSION, HORIZON, build_features
from .training import VALID_END, HOURS, as_reading, metrics, train_models


def now():
    return datetime.now(timezone.utc).isoformat()

class Platform:
    def __init__(self, root: Path):
        self.root = root
        self.lock = threading.RLock()
        self.cache = {}
        root.mkdir(parents=True, exist_ok=True)
        if not (root / 'artifacts' / 'forest-v1.json').exists():
            train_models(root)
        dataset_bytes = (root / 'data' / 'synthetic-v1.json').read_bytes()
        manifest = json.loads((root / 'artifacts' / 'forest-v1.json').read_text())
        if hashlib.sha256(dataset_bytes).hexdigest() != manifest['dataset_id']:
            raise ValueError('Dataset checksum mismatch; restore the versioned dataset')
        self.rows = json.loads(dataset_bytes)
        self.series = {m: [r for r in self.rows if r['machine_id'] == m] for m in sorted({r['machine_id'] for r in self.rows})}
        self.db = sqlite3.connect(root / 'data' / 'sentinel.db', check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS models(version TEXT PRIMARY KEY, report TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS deployments(id INTEGER PRIMARY KEY, version TEXT, previous TEXT, action TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY, machine_id TEXT, timestamp TEXT, created_at TEXT,
            hour INTEGER, model_version TEXT, feature_version TEXT, probability REAL, threshold REAL, alert INTEGER,
            features TEXT, readings TEXT, attribution TEXT, outcome INTEGER, source TEXT);
        CREATE TABLE IF NOT EXISTS checks(id INTEGER PRIMARY KEY, created_at TEXT, kind TEXT, severity TEXT, message TEXT);
        CREATE TABLE IF NOT EXISTS jobs(id INTEGER PRIMARY KEY, created_at TEXT, status TEXT, reason TEXT, result TEXT);
        ''')
        for path in sorted((root / 'artifacts').glob('*.json')):
            report = json.loads(path.read_text())
            self.db.execute('INSERT OR IGNORE INTO models VALUES (?,?)', (report['version'], json.dumps(report)))
        if not self.state('active'):
            self.set_state('active', 'baseline-v1')
            self.set_state('cursor', str(VALID_END - 1))
            self.set_state('scenario', 'normal')
            self.db.execute('INSERT INTO deployments(version,previous,action,created_at) VALUES (?,?,?,?)', ('baseline-v1', None, 'initial', now()))
        self.db.commit()

    def state(self, key):
        row = self.db.execute('SELECT value FROM state WHERE key=?', (key,)).fetchone()
        return row[0] if row else None

    def set_state(self, key, value):
        self.db.execute('INSERT OR REPLACE INTO state VALUES (?,?)', (key, value))

    def artifact(self, version):
        if version not in self.cache:
            report = json.loads(self.db.execute('SELECT report FROM models WHERE version=?', (version,)).fetchone()[0])
            if report['feature_version'] != FEATURE_VERSION:
                raise ValueError('Model feature contract is incompatible with serving')
            data = (self.root / 'artifacts' / f'{version}.pkl').read_bytes()
            if hashlib.sha256(data).hexdigest() != report['artifact_sha256']:
                raise ValueError('Model artifact checksum mismatch')
            # Only application-generated, locally trusted artifacts may be loaded.
            self.cache[version] = pickle.loads(data)
        return self.cache[version]

    def check(self, kind, severity, message):
        self.db.execute('INSERT INTO checks(created_at,kind,severity,message) VALUES (?,?,?,?)', (now(), kind, severity, message))

    def predict(self, machine_id, readings, hour=None, source='api'):
        with self.lock, self.db:
            x = build_features(readings)
            version = self.state('active')
            artifact = self.artifact(version)
            probability = float(artifact['model'].predict_proba(x.reshape(1,-1))[0,1])
            # Grouped median-replacement sensitivity, not SHAP or causal attribution.
            variants = np.tile(x, (4,1))
            for i in range(4):
                variants[i, i*4:(i+1)*4] = artifact['reference'][i*4:(i+1)*4]
            perturbed = artifact['model'].predict_proba(variants)[:,1]
            attribution = sorted([{'sensor': sensor, 'contribution': float(probability-p)} for sensor,p in zip(('temperature','vibration','pressure','load'),perturbed)], key=lambda a: abs(a['contribution']), reverse=True)
            result = {'machine_id': machine_id, 'timestamp': readings[-1].timestamp.isoformat(), 'created_at': now(),
                'model_version': version, 'feature_version': FEATURE_VERSION, 'probability': probability,
                'threshold': artifact['report']['threshold'], 'attribution': attribution, 'horizon_hours': HORIZON}
            result['alert'] = probability >= result['threshold']
            cursor = self.db.execute('''INSERT INTO predictions(machine_id,timestamp,created_at,hour,model_version,feature_version,
                probability,threshold,alert,features,readings,attribution,source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (machine_id, result['timestamp'], result['created_at'], hour, version, FEATURE_VERSION, probability, result['threshold'],
                 int(result['alert']), json.dumps(x.tolist()), json.dumps([r.model_dump(mode='json') for r in readings]), json.dumps(attribution), source))
            result['id'] = cursor.lastrowid
            return result

    def reveal_labels(self, hour, delay):
        pending = self.db.execute("SELECT id,machine_id,hour FROM predictions WHERE source='replay' AND outcome IS NULL AND hour+?<=?", (HORIZON+delay, hour)).fetchall()
        for row in pending:
            future = self.series[row['machine_id']][row['hour']+1:row['hour']+1+HORIZON]
            self.db.execute('UPDATE predictions SET outcome=? WHERE id=?', (int(any(r['failure'] for r in future)), row['id']))

    def replay(self, steps=1):
        with self.lock:
            results = []
            scenario = self.state('scenario')
            for _ in range(steps):
                hour = int(self.state('cursor')) + 1
                if hour >= HOURS - HORIZON:
                    break
                for index, (machine, series) in enumerate(self.series.items()):
                    window = [dict(row) for row in series[hour-5:hour+1]]
                    if scenario == 'outage' and index < 3:
                        with self.db:
                            self.check('missing_data', 'critical', f'{machine}: missing vibration sample at replay hour {hour}; prediction withheld.')
                        continue
                    if scenario == 'schema_change' and index < 2:
                        window[-1]['temperature_celsius'] = window[-1].pop('temperature')
                    if scenario == 'drift':
                        for row in window:
                            row['temperature'] += 18
                            row['vibration'] += 2.5
                    try:
                        results.append(self.predict(machine, [as_reading(r) for r in window], hour, 'replay'))
                    except (ValueError, KeyError) as error:
                        with self.db:
                            self.check('schema', 'critical', f'{machine}: invalid sensor contract at hour {hour}; prediction withheld ({type(error).__name__}).')
                with self.db:
                    self.set_state('cursor', str(hour))
                    self.reveal_labels(hour, 24 if scenario == 'delayed_labels' else 0)
            with self.db:
                monitor = self.monitor()
                if monitor['drift_detected']:
                    self.check('feature_drift', 'warning', f"Feature PSI {monitor['max_psi']:.2f} exceeds 0.25. Candidate review recommended; deployment unchanged.")
            return {'processed': len(results), 'cursor': int(self.state('cursor')), 'exhausted': int(self.state('cursor')) >= HOURS-HORIZON-1}

    def monitor(self):
        rows = self.db.execute("SELECT features,probability FROM predictions WHERE source='replay' ORDER BY id DESC LIMIT 120").fetchall()
        if len(rows) < 24:
            return {'status': 'warming_up', 'sample_count': len(rows), 'max_psi': 0, 'drift_detected': False, 'feature_psi': [], 'prediction_psi': None}
        artifact = self.artifact(self.state('active'))
        reference = artifact['reference_samples']
        current = np.array([json.loads(r['features']) for r in rows])
        def psi(a,b):
            edges = np.unique(np.quantile(a, np.linspace(0,1,11)))
            edges[0], edges[-1] = -np.inf, np.inf
            p = np.maximum(np.histogram(a, edges)[0] / len(a), .0001)
            q = np.maximum(np.histogram(b, edges)[0] / len(b), .0001)
            return float(np.sum((q-p)*np.log(q/p)))
        values = [{'feature': name, 'psi': psi(reference[:,i], current[:,i])} for i,name in enumerate(FEATURE_NAMES)]
        # Rescore the same windows with current model to avoid mixing model versions.
        prediction_psi = psi(artifact['model'].predict_proba(reference)[:,1], artifact['model'].predict_proba(current)[:,1])
        maximum = max(v['psi'] for v in values)
        return {'status': 'warning' if maximum > .25 else 'healthy', 'sample_count': len(rows), 'max_psi': maximum,
                'drift_detected': maximum > .25, 'feature_psi': sorted(values,key=lambda v:v['psi'],reverse=True), 'prediction_psi': prediction_psi}

    def registry(self):
        active = self.state('active')
        reports = [json.loads(row[0]) for row in self.db.execute('SELECT report FROM models')]
        incumbent = next(r for r in reports if r['version'] == active)
        for report in reports:
            report['active'] = report['version'] == active
            v, old = report['validation'], incumbent['validation']
            report['gate_passed'] = v['recall'] >= .70 and v['precision'] >= .60 and v['f1'] >= old['f1'] - .02
            report['gate_rule'] = 'Validation recall ≥ 0.70, precision ≥ 0.60, F1 ≥ incumbent F1 − 0.02'
        return reports

    def approve(self, version):
        with self.lock, self.db:
            report = next((r for r in self.registry() if r['version'] == version), None)
            if not report:
                raise ValueError('Unknown model version')
            if report['active']:
                raise ValueError('Model is already active')
            if not report['gate_passed']:
                raise ValueError('Validation gate failed; deployment blocked')
            self.artifact(version)
            previous = self.state('active')
            self.set_state('active', version)
            self.db.execute('INSERT INTO deployments(version,previous,action,created_at) VALUES (?,?,?,?)', (version, previous, 'approved', now()))
            return {'active': version, 'previous': previous}

    def rollback(self):
        with self.lock, self.db:
            last = self.db.execute('SELECT * FROM deployments ORDER BY id DESC LIMIT 1').fetchone()
            if not last or not last['previous']:
                raise ValueError('No previous deployment to restore')
            previous = self.state('active')
            self.artifact(last['previous'])
            self.set_state('active', last['previous'])
            self.db.execute('INSERT INTO deployments(version,previous,action,created_at) VALUES (?,?,?,?)', (last['previous'], previous, 'rollback', now()))
            return {'active': last['previous']}

    def train_candidate(self):
        # A reproducibility job: no drifted or test rows enter training.
        with self.lock:
            start = time.perf_counter()
            job_dir = self.root / 'artifacts' / 'jobs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
            reports = train_models(job_dir)
            result = {'version': 'forest-v1', 'artifact_directory': str(job_dir), 'elapsed_seconds': round(time.perf_counter()-start,2),
                      'validation': reports[1]['validation'], 'deployment_changed': False,
                      'note': 'Reproduced candidate on immutable training partition. New-data retraining is a roadmap item.'}
            with self.db:
                self.db.execute('INSERT INTO jobs(created_at,status,reason,result) VALUES (?,?,?,?)', (now(), 'completed', 'manual_candidate_training', json.dumps(result)))
            return result

    def summary(self):
        with self.lock:
            machines = []
            cursor = int(self.state('cursor'))
            for machine in self.series:
                row = self.db.execute('SELECT * FROM predictions WHERE machine_id=? AND source=\'replay\' ORDER BY id DESC LIMIT 1', (machine,)).fetchone()
                if row:
                    item = dict(row)
                    item['sensors'] = json.loads(item.pop('readings'))[-1]
                    item['attribution'] = json.loads(item['attribution'])
                    item.pop('features')
                    item['stale'] = item['hour'] < cursor
                    machines.append(item)
            alerts = [dict(r) for r in self.db.execute('SELECT id,machine_id,timestamp,probability,model_version,outcome FROM predictions WHERE alert=1 ORDER BY id DESC LIMIT 30')]
            labeled = self.db.execute("SELECT * FROM predictions WHERE outcome IS NOT NULL AND source='replay'").fetchall()
            performance = []
            for version in sorted({r['model_version'] for r in labeled}):
                group = [r for r in labeled if r['model_version']==version]
                meta = [{'machine_id': r['machine_id'], 'hour': r['hour'], 'event_hour': next((s['hour'] for s in self.series[r['machine_id']][r['hour']+1:r['hour']+1+HORIZON] if s['failure']), None)} for r in group]
                if group:
                    values = metrics(np.array([r['outcome'] for r in group]), np.array([r['probability'] for r in group]), group[0]['threshold'], meta)
                    performance.append({'version': version, **values})
            return {'active_model': self.state('active'), 'scenario': self.state('scenario'), 'cursor': cursor,
                'simulated_time': self.series['EQ-001'][cursor]['timestamp'], 'machines': machines, 'alerts': alerts,
                'total_predictions': self.db.execute('SELECT COUNT(*) FROM predictions').fetchone()[0],
                'pending_labels': self.db.execute("SELECT COUNT(*) FROM predictions WHERE outcome IS NULL AND source='replay'").fetchone()[0],
                'monitoring': self.monitor(), 'performance': performance, 'models': self.registry(),
                'checks': [dict(r) for r in self.db.execute('SELECT * FROM checks ORDER BY id DESC LIMIT 12')],
                'deployments': [dict(r) for r in self.db.execute('SELECT * FROM deployments ORDER BY id DESC LIMIT 10')],
                'jobs': [dict(r) for r in self.db.execute('SELECT id,created_at,status,reason FROM jobs ORDER BY id DESC LIMIT 5')]}
