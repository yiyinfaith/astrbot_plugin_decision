"""Durable, scoped contact facts. No platform or model identities are stored here.

Transactions are short and contain no await: reservations, callback claims and
delivery acknowledgements are atomic even when native and recovery timers race.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager


class LifeLedger:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.transaction() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS profiles (
                    umo TEXT, persona TEXT, data TEXT NOT NULL,
                    PRIMARY KEY (umo, persona));
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY, umo TEXT NOT NULL, persona TEXT NOT NULL,
                    due REAL NOT NULL, status TEXT NOT NULL, data TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS task_scope ON tasks(umo,persona,due);
                CREATE INDEX IF NOT EXISTS task_due ON tasks(status,due);
                CREATE TABLE IF NOT EXISTS contacts (
                    id TEXT PRIMARY KEY, umo TEXT, persona TEXT, at REAL,
                    kind TEXT, status TEXT);
                CREATE INDEX IF NOT EXISTS contact_scope ON contacts(umo,persona,at);
                CREATE TABLE IF NOT EXISTS thoughts (
                    id TEXT PRIMARY KEY, persona TEXT, created REAL,
                    expires REAL, data TEXT);
                CREATE INDEX IF NOT EXISTS thought_persona ON thoughts(persona,created);
                CREATE TABLE IF NOT EXISTS shares (
                    thought TEXT, umo TEXT, status TEXT,
                    PRIMARY KEY(thought,umo));
                CREATE TABLE IF NOT EXISTS signals (
                    umo TEXT, persona TEXT, id TEXT, at REAL,
                    PRIMARY KEY(umo,persona,id));
                CREATE TABLE IF NOT EXISTS idea_attempts (
                    persona TEXT, node TEXT, at REAL,
                    PRIMARY KEY(persona,node));
            """)

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA busy_timeout=5000")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def new_id():
        return uuid.uuid4().hex

    @staticmethod
    def profile(db, umo, persona, now, minimum, maximum):
        row = db.execute("SELECT data FROM profiles WHERE umo=? AND persona=?", (umo, persona)).fetchone()
        data = json.loads(row[0]) if row else {
            "created": now, "baseline": min(maximum, minimum * 2),
            "interval": min(maximum, minimum * 2), "paused_until": 0,
            "last_sent": 0, "last_incoming": 0, "last_positive": 0,
            "unanswered": 0, "awaiting": False, "negative_applied": False,
            "feedback": "", "feedback_at": 0,
        }
        data["baseline"] = min(maximum, max(minimum, data["baseline"]))
        data["interval"] = min(maximum, max(minimum, data["interval"]))
        return data

    @staticmethod
    def save_profile(db, umo, persona, data):
        db.execute("INSERT OR REPLACE INTO profiles VALUES (?,?,?)", (umo, persona, json.dumps(data, ensure_ascii=False)))

    @staticmethod
    def task(row):
        return {**json.loads(row["data"]), **{key: row[key] for key in ("id", "umo", "persona", "due", "status")}}

    @staticmethod
    def save_task(db, task):
        db.execute("INSERT OR REPLACE INTO tasks VALUES (?,?,?,?,?,?)", (
            task["id"], task["umo"], task["persona"], task["due"], task["status"],
            json.dumps(task, ensure_ascii=False),
        ))

    def tasks(self, umo=None, persona=None, *, active=False, limit=100):
        where, args = [], []
        for key, val in (("umo", umo), ("persona", persona)):
            if val is not None:
                where.append(f"{key}=?")
                args.append(val)
        if active:
            where.append("status IN ('pending','preparing','ready','sending','suspended','awaiting_delivery')")
        with self.transaction() as db:
            return [self.task(row) for row in db.execute(
                "SELECT * FROM tasks" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY due DESC LIMIT ?",
                (*args, limit),
            )]

    def get_task(self, ident):
        with self.transaction() as db:
            row = db.execute("SELECT * FROM tasks WHERE id=?", (ident,)).fetchone()
            return self.task(row) if row else None

    def prune(self, now):
        with self.transaction() as db:
            db.execute("DELETE FROM signals WHERE at<?", (now - 7 * 86400,))
            db.execute("DELETE FROM idea_attempts WHERE at<?", (now - 7 * 86400,))
            db.execute("DELETE FROM contacts WHERE at<? AND status NOT IN ('sending','unknown')", (now - 40 * 86400,))
            db.execute("DELETE FROM shares WHERE thought IN (SELECT id FROM thoughts WHERE expires<?)", (now - 7 * 86400,))
            db.execute("DELETE FROM thoughts WHERE expires<?", (now - 7 * 86400,))
            db.execute("DELETE FROM shares WHERE NOT EXISTS(SELECT 1 FROM thoughts WHERE id=shares.thought)")
            # Keep recent completed facts for follow-up questions, never drop
            # unfinished commitments or delivery-unknown records silently.
            db.execute("DELETE FROM tasks WHERE due<? AND status IN ('delivered','cancelled','expired')", (now - 40 * 86400,))
