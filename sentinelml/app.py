import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from .features import Reading, FEATURE_VERSION
from .service import Platform

ROOT = Path(os.getenv('SENTINEL_HOME', Path(__file__).resolve().parents[1]))
STATIC = Path(__file__).parent / 'static'

@asynccontextmanager
async def lifespan(app):
    app.state.platform = Platform(ROOT)
    app.state.requests = 0
    app.state.errors = 0
    app.state.latency = 0.
    if int(app.state.platform.state('cursor')) == 575:
        app.state.platform.replay(6)
    yield
    app.state.platform.db.close()

app = FastAPI(title='SentinelML', version='0.1.0', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=STATIC), name='static')

class PredictRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    machine_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,40}$')
    readings: list[Reading] = Field(min_length=6, max_length=6)

class ReplayRequest(BaseModel):
    steps: int = Field(default=1, ge=1, le=24)

class ScenarioRequest(BaseModel):
    scenario: Literal['normal', 'outage', 'drift', 'delayed_labels', 'schema_change']

@app.middleware('http')
async def observe(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        request.app.state.errors += 1
        raise
    finally:
        request.app.state.requests += 1
        request.app.state.latency += time.perf_counter() - start
    if response.status_code >= 500:
        request.app.state.errors += 1
    return response

@app.get('/', include_in_schema=False)
def index():
    return FileResponse(STATIC / 'index.html')

@app.get('/health')
def health(request: Request):
    p = request.app.state.platform
    with p.lock:
        p.db.execute('SELECT 1').fetchone()
        p.artifact(p.state('active'))
        return {'status': 'ok', 'model_version': p.state('active'), 'feature_version': FEATURE_VERSION}

@app.get('/metrics', response_class=PlainTextResponse)
def prometheus(request: Request):
    s = request.app.state
    return f'# TYPE sentinel_requests_total counter\nsentinel_requests_total {s.requests}\n# TYPE sentinel_errors_total counter\nsentinel_errors_total {s.errors}\n# TYPE sentinel_request_duration_seconds_sum counter\nsentinel_request_duration_seconds_sum {s.latency}\n'

@app.get('/api/summary')
def summary(request: Request):
    return request.app.state.platform.summary()

@app.post('/api/predict')
def predict(body: PredictRequest, request: Request):
    try:
        return request.app.state.platform.predict(body.machine_id, body.readings)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error

@app.post('/api/replay')
def replay(body: ReplayRequest, request: Request):
    return request.app.state.platform.replay(body.steps)

@app.post('/api/scenario')
def scenario(body: ScenarioRequest, request: Request):
    p = request.app.state.platform
    with p.lock, p.db:
        p.set_state('scenario', body.scenario)
        p.check('scenario', 'info', f'Scenario changed to {body.scenario}. Applies to subsequent replay hours.')
    return {'scenario': body.scenario}

@app.post('/api/models/{version}/approve')
def approve(version: str, request: Request):
    try:
        return request.app.state.platform.approve(version)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error

@app.post('/api/models/rollback')
def rollback(request: Request):
    try:
        return request.app.state.platform.rollback()
    except ValueError as error:
        raise HTTPException(409, str(error)) from error

@app.post('/api/training')
def training(request: Request):
    return request.app.state.platform.train_candidate()

@app.get('/api/machines/{machine_id}/history')
def history(machine_id: str, request: Request):
    p = request.app.state.platform
    with p.lock:
        if machine_id not in p.series:
            raise HTTPException(404, 'Unknown replay machine')
        rows = p.db.execute("SELECT timestamp,probability,readings,model_version FROM predictions WHERE machine_id=? AND source='replay' ORDER BY id DESC LIMIT 36", (machine_id,)).fetchall()
        import json
        return [{'timestamp':r['timestamp'],'probability':r['probability'],'model_version':r['model_version'],**json.loads(r['readings'])[-1]} for r in reversed(rows)]
