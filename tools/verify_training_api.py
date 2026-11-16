"""Run real subprocess jobs on the explicitly synthetic dataset."""
import _bootstrap
import json,time
from fastapi.testclient import TestClient
from retailvision.api.app import app,jobs
with TestClient(app) as client:
    body=dict(task='recognition',model='dinov2-vitb14',config='configs/recognition/fixture_smoke.py',dataset='datasets/fixture_v001/recognition')
    start=time.perf_counter();response=client.post('/api/training/start',json=body);elapsed=time.perf_counter()-start
    assert response.status_code==202,response.text
    eid=response.json()['experiment_id']
    assert client.get('/api/training/'+eid).status_code==200
    for i in range(60):
        state=client.get('/api/training/'+eid).json()
        if state['status'] in ('completed','failed'):break
        time.sleep(.5)
    assert state['status']=='completed',state
    jobs.processes[eid].wait(timeout=15)
    response=client.post('/api/training/start',json=body);assert response.status_code==202,response.text
    eid2=response.json()['experiment_id'];stopped=client.post(f'/api/training/{eid2}/stop')
    assert stopped.status_code==200 and client.get('/api/training/'+eid2).json()['status']=='stopped'
    result=dict(fixture=True,start_http=202,start_request_ms=elapsed*1000,experiment_id=eid,status=state['status'],stop_status='stopped')
    from pathlib import Path
    Path('results/training_api_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
