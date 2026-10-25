"""
SiteGuard AI — Database Wipe & Reinitialization Utility
Backs up current SQLite database and initializes clean schema with default admin account.
"""

import os
import sys
import shutil
import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from database.models import init_db, db_session, User, get_db_path

def reset_database():
    db_path = Path(get_db_path())
    print("=" * 60)
    print("      SITEGUARD AI — DATABASE RESET & SEED TOOL")
    print("=" * 60)
    print(f"[*] Target Database: {db_path.resolve()}")

    if db_path.exists():
        backup_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = db_path.parent / f"safety_monitor_backup_{backup_ts}.db"
        shutil.copy2(db_path, backup_path)
        print(f"[*] Created safe backup at: {backup_path.name}")
        
        try:
            db_path.unlink()
            print(f"[✓] Removed existing database file.")
        except Exception as e:
            print(f"[!] Warning deleting file: {e}")

    # Reinitialize fresh schema
    print("\n[*] Initializing fresh tables and indices...")
    init_db()

    # Seed initial admin user if not present
    session = db_session()
    try:
        admin = session.query(User).filter_by(username="admin").first()
        if not admin:
            admin = User(
                username="admin",
                role="ADMIN",
                full_name="SiteGuard Safety Administrator",
                email="admin@siteguard.ai",
                is_active=True
            )
            session.add(admin)
            session.commit()
            print(f"[✓] Seeded default administrator account: 'admin' (Role: ADMIN)")
        else:
            print(f"[*] Administrator account verified.")
    finally:
        session.close()

    print("\n[✓] Database reset and initialization complete!\n")

if __name__ == "__main__":
    reset_database()
