"""
Tests for RESTful Telemetry, Detection, and Analytics APIs.
"""
import pytest
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c

def test_api_status_and_stats(client):
    res = client.get("/api/stats")
    assert res.status_code == 200
    data = res.get_json()
    assert "worker_count" in data
    assert "compliance_rate" in data
    assert "camera_online" in data

def test_api_active_alerts(client):
    res = client.get("/api/alerts/active")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert isinstance(data["alerts"], list)

def test_api_detections_frame(client):
    res = client.get("/api/detections/frame")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "workers" in data

def test_api_analytics_endpoints(client):
    for endpoint in [
        "/api/analytics/violations-week",
        "/api/analytics/ppe-breakdown",
        "/api/analytics/compliance-trend",
        "/api/analytics?range=7d"
    ]:
        res = client.get(endpoint)
        assert res.status_code == 200
        json_data = res.get_json()
        assert json_data is not None

def test_api_system_health_run(client):
    res = client.post("/api/system-health/run")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "health" in data
    assert data["health"]["db_ok"] is True

def test_data_exports_csv(client):
    res_inc = client.get("/api/incidents/export")
    assert res_inc.status_code == 200
    assert "text/csv" in res_inc.content_type

    res_viol = client.get("/api/violations/export")
    assert res_viol.status_code == 200
    assert "text/csv" in res_viol.content_type
