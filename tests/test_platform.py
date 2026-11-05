from datetime import timedelta
from pathlib import Path
import json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sentinelml import app as app_module
from sentinelml.features import build_features, FEATURE_VERSION
from sentinelml.training import generate_dataset, as_reading, make_examples, TRAIN_END, HORIZON
from sentinelml.service import Platform

@pytest.fixture(scope='module')
def platform(tmp_path_factory):
    p = Platform(tmp_path_factory.mktemp('platform'))
    yield p
    p.db.close()

@pytest.fixture
def client(platform):
    # Use existing trained artifacts; TestClient lifecycle creates its own connection.
    app_module.ROOT = platform.root
    with TestClient(app_module.app) as c:
        yield c


def test_features_are_past_only_and_reject_gaps():
    rows = generate_dataset()[:20]
    readings = [as_reading(r) for r in rows[:6]]
    before = build_features(readings)
    rows[7]['temperature'] = 190
    assert np.array_equal(before, build_features(readings))
    readings[3] = readings[3].model_copy(update={'timestamp': readings[3].timestamp + timedelta(minutes=1)})
    with pytest.raises(ValueError, match='contiguous'):
        build_features(readings)


def test_temporal_label_purge_and_future_target():
    rows = generate_dataset()[:100]
    x,y,meta = make_examples(rows)
    for label,m in zip(y,meta):
        assert label == int(any(r['failure'] for r in rows[m['hour']+1:m['hour']+1+HORIZON]))
    assert 425 + HORIZON < TRAIN_END
    assert x.shape[1] == 16


def test_prediction_contract_and_audit(client, platform):
    readings = [{k:v for k,v in r.items() if k in ('timestamp','temperature','vibration','pressure','load')} for r in platform.rows[:6]]
    response = client.post('/api/predict', json={'machine_id':'EXTERNAL-01','readings':readings})
    assert response.status_code == 200
    result=response.json()
    assert 0<=result['probability']<=1
    assert result['feature_version']==FEATURE_VERSION
    assert len(result['attribution'])==4
    readings[0]['unexpected_sensor']=42
    assert client.post('/api/predict',json={'machine_id':'EQ-001','readings':readings}).status_code==422
    del readings[0]['unexpected_sensor']
    readings[0]['temperature']=None
    assert client.post('/api/predict',json={'machine_id':'EQ-001','readings':readings}).status_code==422
    assert client.get('/health').status_code==200
    assert 'sentinel_requests_total' in client.get('/metrics').text


def test_outage_withholds_predictions_and_labels_mature(client):
    client.post('/api/scenario',json={'scenario':'outage'})
    response=client.post('/api/replay',json={'steps':1})
    assert response.json()['processed']==9
    summary=client.get('/api/summary').json()
    assert sum(m['stale'] for m in summary['machines'])==3
    assert any(c['kind']=='missing_data' for c in summary['checks'])
    client.post('/api/scenario',json={'scenario':'normal'})
    client.post('/api/replay',json={'steps':7})
    summary=client.get('/api/summary').json()
    assert summary['performance']
    assert all(m['hour']+HORIZON <= summary['cursor'] for m in [dict(r) for r in client.app.state.platform.db.execute('SELECT hour FROM predictions WHERE outcome IS NOT NULL')])


def test_schema_and_delayed_labels(client):
    client.post('/api/scenario',json={'scenario':'schema_change'})
    assert client.post('/api/replay',json={'steps':1}).json()['processed']==10
    client.post('/api/scenario',json={'scenario':'delayed_labels'})
    p=client.app.state.platform
    before=p.db.execute('SELECT COUNT(*) FROM predictions WHERE outcome IS NOT NULL').fetchone()[0]
    client.post('/api/replay',json={'steps':2})
    after=p.db.execute('SELECT COUNT(*) FROM predictions WHERE outcome IS NOT NULL').fetchone()[0]
    assert after==before
    client.post('/api/scenario',json={'scenario':'normal'})


def test_approval_gate_and_rollback(client):
    summary=client.get('/api/summary').json()
    active=summary['active_model']
    candidate=next(m for m in summary['models'] if not m['active'])
    response=client.post(f"/api/models/{candidate['version']}/approve")
    if candidate['gate_passed']:
        assert response.status_code==200
        assert client.get('/health').json()['model_version']==candidate['version']
        assert client.post('/api/models/rollback').json()['active']==active
    else:
        assert response.status_code==409
        assert client.get('/health').json()['model_version']==active
    assert client.post('/api/models/nonexistent/approve').status_code==409


def test_gate_blocks_bad_candidate(client):
    p=client.app.state.platform
    row=p.db.execute("SELECT report FROM models WHERE version='forest-v1'").fetchone()[0]
    bad=json.loads(row)
    bad['validation']['recall']=0
    with p.db: p.db.execute("UPDATE models SET report=? WHERE version='forest-v1'",(json.dumps(bad),))
    assert client.post('/api/models/forest-v1/approve').status_code==409
    with p.db: p.db.execute("UPDATE models SET report=? WHERE version='forest-v1'",(row,))


def test_drift_does_not_deploy(client):
    active=client.get('/health').json()['model_version']
    client.post('/api/scenario',json={'scenario':'drift'})
    client.post('/api/replay',json={'steps':10})
    result=client.get('/api/summary').json()
    assert result['monitoring']['drift_detected']
    assert result['active_model']==active
    client.post('/api/scenario',json={'scenario':'normal'})


def test_reproducible_candidate_job(client):
    active=client.get('/health').json()['model_version']
    response=client.post('/api/training')
    assert response.status_code==200
    assert not response.json()['deployment_changed']
    assert client.get('/health').json()['model_version']==active
    assert Path(response.json()['artifact_directory']).exists()


def test_restart_restores_state_and_rejects_corrupt_dataset(platform):
    cursor = platform.state('cursor')
    active = platform.state('active')
    restored = Platform(platform.root)
    assert restored.state('cursor') == cursor
    assert restored.state('active') == active
    restored.db.close()
    path = platform.root / 'data' / 'synthetic-v1.json'
    original = path.read_bytes()
    try:
        path.write_bytes(original + b' ')
        with pytest.raises(ValueError, match='Dataset checksum mismatch'):
            Platform(platform.root)
    finally:
        path.write_bytes(original)
