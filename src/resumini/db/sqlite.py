import json
import sqlite3


class SQLiteStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row

    def initialize(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS materias (
                nombre TEXT PRIMARY KEY,
                metadata_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS preferences (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                query TEXT,
                timestamp TEXT,
                metadata_json TEXT
            );
        """)
        self._conn.commit()

    def add_materia(self, nombre: str, metadata: dict):
        self._conn.execute(
            "INSERT OR REPLACE INTO materias (nombre, metadata_json) VALUES (?, ?)",
            (nombre, json.dumps(metadata, ensure_ascii=False)),
        )
        self._conn.commit()

    def get_materia(self, nombre: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM materias WHERE nombre = ?", (nombre,)
        ).fetchone()
        if not row:
            return None
        meta = json.loads(row["metadata_json"])
        meta["nombre"] = row["nombre"]
        return meta

    def list_materias(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM materias").fetchall()
        result = []
        for row in rows:
            meta = json.loads(row["metadata_json"])
            meta["nombre"] = row["nombre"]
            result.append(meta)
        return result

    def set_preference(self, key: str, value: str):
        self._conn.execute(
            "INSERT OR REPLACE INTO preferences (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def get_preference(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM preferences WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def add_session(self, session_id: str, query: str, timestamp: str, metadata: dict | None = None):
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        self._conn.execute(
            "INSERT OR REPLACE INTO sessions (session_id, query, timestamp, metadata_json) VALUES (?, ?, ?, ?)",
            (session_id, query, timestamp, meta_json),
        )
        self._conn.commit()

    def get_recent_sessions(self, limit: int = 10) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM sessions ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self):
        self._conn.close()
