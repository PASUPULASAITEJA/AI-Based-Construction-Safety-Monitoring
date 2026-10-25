"""
SiteGuard AI — System Health Check Runner
Wraps and executes full system diagnostics across dependencies, AI model,
database schema, CV inference pipeline, and HTTP web endpoints.
"""

import os
import sys
from pathlib import Path

# Ensure repo root is in python path regardless of invocation directory
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import verify_full_system

if __name__ == "__main__":
    verify_full_system.main()
