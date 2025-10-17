import os
import sqlite3
import hashlib
from pathlib import Path
from typing import Optional, List

# Default to persisted location inside uploads volume; allow override via env
DB_PATH = os.getenv('USERS_DB_PATH', '/app/data/users.db')

class AuthService:
    def __init__(self, db_path=DB_PATH):
        # Ensure parent directory exists so SQLite can create the file on first run
        parent = Path(db_path).parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Fallback to tmp if parent isn't writable
            tmp_dir = Path('/tmp/app_db')
            tmp_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(tmp_dir / 'users.db')

        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL
                )
            ''')

    def hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    def create_user(self, username: str, password: str, role: str, admin: bool = False) -> bool:
        # Only admin user can create other users
        if not admin and self.user_count() > 0:
            return False
        pw_hash = self.hash_password(password)
        try:
            with self.conn:
                self.conn.execute(
                    'INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)',
                    (username, pw_hash, role)
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def authenticate(self, username: str, password: str) -> Optional[dict]:
        """
        Authenticates a user and returns user information including roles.
        
        Args:
            username (str): The username
            password (str): The password
            
        Returns:
            Optional[dict]: User info with username and roles if authenticated, None otherwise
        """
        pw_hash = self.hash_password(password)
        user = self.conn.execute(
            'SELECT username, role FROM users WHERE username=? AND password_hash=?',
            (username, pw_hash)
        ).fetchone()
        
        if user:
            username, role_string = user
            # Parse roles - assuming comma-separated format like "Marketing & Vertrieb,IT & Technik"
            user_roles = [role.strip() for role in role_string.split(',')] if role_string else []
            
            return {
                'username': username,
                'roles': user_roles,
                'role_string': role_string
            }
        return None

    def user_count(self) -> int:
        res = self.conn.execute('SELECT COUNT(*) FROM users').fetchone()
        return res[0] if res else 0

    def get_users(self) -> List[str]:
        users = self.conn.execute('SELECT username, role FROM users').fetchall()
        return users

    def check_role(self, username: str, role: str) -> bool:
        user = self.conn.execute(
            'SELECT role FROM users WHERE username=?',
            (username,)
        ).fetchone()
        return user and user[0] == role
