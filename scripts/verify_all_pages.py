"""
Verify all 18 template routes render with HTTP 200 and no 500 or Jinja2 syntax errors.
"""
import sys
import os
sys.path.insert(0, os.path.abspath("."))
from app import app

def test_routes():
    routes = [
        "/",
        "/monitor",
        "/video-analysis",
        "/image-analysis",
        "/violations",
        "/alerts",
        "/incidents",
        "/sites",
        "/cameras",
        "/workers",
        "/analytics",
        "/reports",
        "/model",
        "/insights",
        "/system-health",
        "/notifications",
        "/audit-logs",
        "/settings",
        "/upload",
        "/ppe",
        "/login"
    ]
    
    api_routes = [
        "/api/stats",
        "/api/alerts/active",
        "/api/detections/frame",
        "/api/analytics/violations-week",
        "/api/analytics/ppe-breakdown",
        "/api/analytics/compliance-trend",
        "/api/analytics?range=7d"
    ]

    client = app.test_client()
    passed = 0
    failed = 0

    print("--- TESTING PAGE ROUTES ---")
    for r in routes:
        try:
            res = client.get(r, follow_redirects=True)
            if res.status_code == 200:
                print(f"[OK] {r} -> HTTP 200")
                passed += 1
            else:
                print(f"[FAIL] {r} -> HTTP {res.status_code}")
                failed += 1
        except Exception as e:
            print(f"[ERROR] {r} -> EXCEPTION: {e}")
            failed += 1

    print("\n--- TESTING API ROUTES ---")
    for api in api_routes:
        try:
            res = client.get(api)
            if res.status_code == 200:
                print(f"[OK] {api} -> HTTP 200 (JSON)")
                passed += 1
            else:
                print(f"[FAIL] {api} -> HTTP {res.status_code}")
                failed += 1
        except Exception as e:
            print(f"[ERROR] {api} -> EXCEPTION: {e}")
            failed += 1

    print(f"\nVerification Results: {passed} PASSED, {failed} FAILED")
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    test_routes()
