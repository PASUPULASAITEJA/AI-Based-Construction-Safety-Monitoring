"""
Tests for Authentication, Session RBAC, and Route Protection.
"""
import pytest
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c

def test_public_and_operator_access(client):
    # Public / dashboard view
    res = client.get("/")
    assert res.status_code == 200

    # Monitor view
    res = client.get("/monitor")
    assert res.status_code == 200

    # System health diagnostics view
    res = client.get("/system-health")
    assert res.status_code == 200

def test_session_role_assignment(client):
    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["username"] = "OfficerLead"
        sess["role"] = "ADMIN"

    res = client.get("/audit-logs")
    assert res.status_code == 200

def test_viewer_role_forbidden_on_admin_routes(client):
    with client.session_transaction() as sess:
        sess["user_id"] = 2
        sess["username"] = "GuestViewer"
        sess["role"] = "VIEWER"

    # POST to create site should be forbidden (403)
    res = client.post("/api/sites", json={"name": "Unauthorized Site"})
    assert res.status_code == 403
