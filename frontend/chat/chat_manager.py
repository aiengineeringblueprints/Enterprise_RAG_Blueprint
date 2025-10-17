import os
import sqlite3
import uuid
from pathlib import Path
from typing import List, Dict


class ChatManager:
    def __init__(self, db_path: str = None):
        # Use env override or default to standard app data location
        db_path = db_path or os.getenv('CHATS_DB_PATH', '/app/data/chats.db')

        # Ensure parent dir exists so SQLite can create file
        parent = Path(db_path).parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            tmp_dir = Path('/tmp/app_db')
            tmp_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(tmp_dir / 'chats.db')

        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    chat_id TEXT PRIMARY KEY,
                    user TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chat_id) REFERENCES chats (chat_id)
                )
            """)

    def create_chat(self, user: str, title: str) -> str:
        chat_id = str(uuid.uuid4())
        with self.conn:
            self.conn.execute(
                "INSERT INTO chats (chat_id, user, title) VALUES (?, ?, ?)",
                (chat_id, user, title)
            )
        return chat_id

    def list_chats(self, user: str) -> List[Dict]:
        cur = self.conn.execute(
            "SELECT chat_id, title FROM chats WHERE user=? ORDER BY created_at DESC", (user,)
        )
        return [{"chat_id": row[0], "title": row[1]} for row in cur.fetchall()]

    def get_messages(self, chat_id: str) -> List[Dict]:
        cur = self.conn.execute(
            "SELECT sender, content FROM messages WHERE chat_id=? ORDER BY timestamp ASC", (chat_id,)
        )
        return [{"role": row[0], "content": row[1]} for row in cur.fetchall()]

    def add_message(self, chat_id: str, sender: str, content: str):
        with self.conn:
            self.conn.execute(
                "INSERT INTO messages (chat_id, sender, content) VALUES (?, ?, ?)",
                (chat_id, sender, content)
            )
