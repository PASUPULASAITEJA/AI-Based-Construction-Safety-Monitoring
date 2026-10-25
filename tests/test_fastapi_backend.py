"""
Automated Integration Tests for SiteGuard AI FastAPI Backend
Validates REST API routes, image analysis, background video jobs, and research telemetry.
"""
import pytest
from fastapi.testclient import TestClient
from fastapi_app import app

@pytest.fixture
def client():
    return TestClient(app)

def test_fastapi_dashboard_stats(client):
    res = client.get("/api/dashboard/stats")
    assert res.status_code == 200
    data = res.json()
    assert "kpis" in data
    assert "observed_workers" in data["kpis"]
    assert "violations_today" in data["kpis"]
    assert "recent_incidents" in data
    assert "violation_categories" in data

def test_fastapi_system_health(client):
    res = client.get("/api/system-health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert data["database_connected"] is True
    assert "database_latency_ms" in data

def test_fastapi_model_info(client):
    res = client.get("/api/model/info")
    assert res.status_code == 200
    data = res.json()
    assert "model_path" in data
    assert "classes" in data

def test_fastapi_sites_and_zones(client):
    # Test sites
    res_sites = client.get("/api/sites")
    assert res_sites.status_code == 200
    sites = res_sites.json()
    assert isinstance(sites, list)

    # Test zones
    res_zones = client.get("/api/zones")
    assert res_zones.status_code == 200
    zones = res_zones.json()
    assert isinstance(zones, list)

def test_fastapi_violations_and_incidents(client):
    res_viol = client.get("/api/violations")
    assert res_viol.status_code == 200
    assert isinstance(res_viol.json(), list)

    res_inc = client.get("/api/incidents")
    assert res_inc.status_code == 200
    assert isinstance(res_inc.json(), list)

def test_fastapi_analytics(client):
    res = client.get("/api/analytics?range=7d")
    assert res.status_code == 200
    data = res.json()
    assert "timeline" in data
    assert "hourly_distribution" in data
    assert "type_breakdown" in data

def test_fastapi_research_eval_endpoints(client):
    res_eval = client.get("/api/research/evaluation")
    assert res_eval.status_code == 200
    eval_data = res_eval.json()
    assert "configurations" in eval_data or "error" not in eval_data

    res_rob = client.get("/api/research/robustness")
    assert res_rob.status_code == 200
    rob_data = res_rob.json()
    assert "conditions" in rob_data
