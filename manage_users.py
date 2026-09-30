"""
User administration CLI for SiteGuard AI.

  python manage_users.py list
  python manage_users.py add <username> <ADMIN|SAFETY_OFFICER|SUPERVISOR|VIEWER> [full name]
  python manage_users.py passwd <username>
"""
import sys
import getpass
from werkzeug.security import generate_password_hash
from database.db import init_db, get_db_connection
from database.models import UserModel

ROLES = ("ADMIN", "SAFETY_OFFICER", "SUPERVISOR", "VIEWER")

def prompt_password():
    pw = getpass.getpass("Password: ")
    if len(pw) < 8:
        sys.exit("Password must be at least 8 characters.")
    if pw != getpass.getpass("Repeat password: "):
        sys.exit("Passwords do not match.")
    return pw

def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    init_db()
    cmd = argv[1]

    if cmd == "list":
        for u in UserModel.get_all():
            print(f"{u['username']:<20} {u['role']:<16} {u['full_name']}")
    elif cmd == "add" and len(argv) >= 4:
        username, role = argv[2], argv[3].upper()
        if role not in ROLES:
            sys.exit(f"Role must be one of: {', '.join(ROLES)}")
        full_name = " ".join(argv[4:]) or username
        UserModel.create(username, prompt_password(), full_name, role)
        print(f"Created {username} ({role}).")
    elif cmd == "passwd" and len(argv) == 3:
        conn = get_db_connection()
        cur = conn.execute("UPDATE users SET password_hash = ? WHERE username = ?",
                           (generate_password_hash(prompt_password()), argv[2]))
        conn.commit()
        conn.close()
        print("Password updated." if cur.rowcount else f"No user named {argv[2]}.")
    else:
        sys.exit(__doc__)

if __name__ == "__main__":
    main(sys.argv)
