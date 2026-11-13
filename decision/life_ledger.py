"""Durable, scoped contact facts. No platform or model identities are stored here.

Transactions are short and contain no await: reservations, callback claims and
delivery acknowledgements are atomic even when native and recovery timers race.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager


# Wording in decision records written before 0.4.15.
# Only the plugin's own phrases: error text inside a reason may name the
# user's providers or models, which must stay as they are.
_LEGACY_AUDIT_WORDS = (
    ("原生 JEV DecisionService", "原生决策服务 DecisionService"),
    ("原生 JEV Decision Provider", "原生决策 Provider"),
    ("AstrBot JEV ", "AstrBot decision "),
    ("Jev 未通过", "决策未通过"),
    ("Jev 通过", "决策通过"),
    ("Jev 判断", "决策判断"),
)


def _reword_legacy_audit(db) -> None:
    """Reword only the plugin-written reason/stage fields, once."""

    rows = db.execute(
        "SELECT id, data FROM audit WHERE data GLOB '*Jev*' OR data GLOB '*JEV*'"
    ).fetchall()
    for row in rows:
        try:
            entry = json.loads(row["data"])
        except (TypeError, ValueError):
            continue
        if not isinstance(entry, dict):
            continue
        changed = False
        for key in ("reason", "stage"):
            text = entry.get(key)
            if not isinstance(text, str):
                continue
            new = text
            for old, replacement in _LEGACY_AUDIT_WORDS:
                new = new.replace(old, replacement)
            if new != text:
                entry[key] = new
                changed = True
        if changed:
            db.execute(
                "UPDATE audit SET data=? WHERE id=?",
                (json.dumps(entry, ensure_ascii=False), row["id"]),
            )


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
                CREATE TABLE IF NOT EXISTS conversation_calls (
                    id TEXT PRIMARY KEY, umo TEXT, persona TEXT, at REAL,
                    kind TEXT, status TEXT);
                CREATE INDEX IF NOT EXISTS conversation_call_scope ON conversation_calls(umo,persona,at);
                CREATE TABLE IF NOT EXISTS topics (
                    id TEXT PRIMARY KEY, umo TEXT, persona TEXT, sender TEXT,
                    due REAL, expires REAL, status TEXT, data TEXT);
                CREATE INDEX IF NOT EXISTS topic_scope ON topics(umo,persona,status,due);
                CREATE TABLE IF NOT EXISTS utterances (
                    id TEXT PRIMARY KEY, umo TEXT, persona TEXT, at REAL, data TEXT);
                CREATE INDEX IF NOT EXISTS utterance_scope ON utterances(umo,persona,at);
                CREATE TABLE IF NOT EXISTS persona_states (
                    persona TEXT PRIMARY KEY, data TEXT);
                CREATE TABLE IF NOT EXISTS audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, umo TEXT, at REAL,
                    sent INTEGER, data TEXT);
                CREATE INDEX IF NOT EXISTS audit_scope ON audit(umo,at);
            """)
        # executescript() commits, so the one-time rewording gets its own
        # transaction: all rows and the version mark land together.
        with self.transaction() as db:
            if db.execute("PRAGMA user_version").fetchone()[0] < 1:
                _reword_legacy_audit(db)
                db.execute("PRAGMA user_version=1")

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
        row = db.execute(
            "SELECT data FROM profiles WHERE umo=? AND persona=?", (umo, persona)
        ).fetchone()
        data = (
            json.loads(row[0])
            if row
            else {
                "created": now,
                "baseline": min(maximum, minimum * 2),
                "interval": min(maximum, minimum * 2),
                "paused_until": 0,
                "last_sent": 0,
                "last_incoming": 0,
                "last_positive": 0,
                "unanswered": 0,
                "awaiting": False,
                "negative_applied": False,
                "feedback": "",
                "feedback_at": 0,
                "relationship_score": 20.0,
                "relationship_updated_at": now,
                "response_latency_seconds": 0.0,
                "response_samples": 0,
                "response_last_at": 0,
                "recent_intents": [],
            }
        )
        data["baseline"] = min(maximum, max(minimum, data["baseline"]))
        data["interval"] = min(maximum, max(minimum, data["interval"]))
        return data

    @staticmethod
    def save_profile(db, umo, persona, data):
        db.execute(
            "INSERT OR REPLACE INTO profiles VALUES (?,?,?)",
            (umo, persona, json.dumps(data, ensure_ascii=False)),
        )

    @staticmethod
    def task(row):
        return {
            **json.loads(row["data"]),
            **{key: row[key] for key in ("id", "umo", "persona", "due", "status")},
        }

    @staticmethod
    def save_task(db, task):
        db.execute(
            "INSERT OR REPLACE INTO tasks VALUES (?,?,?,?,?,?)",
            (
                task["id"],
                task["umo"],
                task["persona"],
                task["due"],
                task["status"],
                json.dumps(task, ensure_ascii=False),
            ),
        )

    def tasks(self, umo=None, persona=None, *, active=False, limit=100):
        where, args = [], []
        for key, val in (("umo", umo), ("persona", persona)):
            if val is not None:
                where.append(f"{key}=?")
                args.append(val)
        if active:
            where.append(
                "status IN ('pending','preparing','ready','sending','suspended','awaiting_delivery')"
            )
        with self.transaction() as db:
            return [
                self.task(row)
                for row in db.execute(
                    "SELECT * FROM tasks"
                    + (" WHERE " + " AND ".join(where) if where else "")
                    + " ORDER BY due DESC LIMIT ?",
                    (*args, limit),
                )
            ]

    def get_task(self, ident):
        with self.transaction() as db:
            row = db.execute("SELECT * FROM tasks WHERE id=?", (ident,)).fetchone()
            return self.task(row) if row else None

    def add_audit(self, umo, at, sent, entry):
        with self.transaction() as db:
            db.execute(
                "INSERT INTO audit(umo,at,sent,data) VALUES (?,?,?,?)",
                (umo, at, 1 if sent else 0, json.dumps(entry, ensure_ascii=False)),
            )

    def audits(self, umo=None, limit=60):
        with self.transaction() as db:
            rows = db.execute(
                "SELECT data FROM audit"
                + (" WHERE umo=?" if umo else "")
                + " ORDER BY at DESC, id DESC LIMIT ?",
                ((umo, limit) if umo else (limit,)),
            ).fetchall()
        return [json.loads(row[0]) for row in reversed(rows)]

    def prune(self, now):
        with self.transaction() as db:
            # Decision records are for troubleshooting: two weeks, and at most
            # 500 per conversation.
            db.execute("DELETE FROM audit WHERE at<?", (now - 14 * 86400,))
            db.execute(
                """DELETE FROM audit WHERE id IN (
                SELECT id FROM (SELECT id, ROW_NUMBER() OVER (
                    PARTITION BY umo ORDER BY at DESC, id DESC) AS rank
                    FROM audit) WHERE rank>500)"""
            )
            db.execute("DELETE FROM signals WHERE at<?", (now - 7 * 86400,))
            db.execute("DELETE FROM idea_attempts WHERE at<?", (now - 7 * 86400,))
            db.execute("DELETE FROM conversation_calls WHERE at<?", (now - 7 * 86400,))
            # Duplicate suppression deliberately compares the last four,
            # regardless of age. Retention must not erase that small baseline.
            db.execute(
                """DELETE FROM utterances WHERE at<? AND id NOT IN (
                SELECT id FROM (SELECT id, ROW_NUMBER() OVER (
                    PARTITION BY umo,persona ORDER BY at DESC,rowid DESC) AS rank
                    FROM utterances) WHERE rank<=4)""",
                (now - 40 * 86400,),
            )
            db.execute("DELETE FROM topics WHERE expires<?", (now - 7 * 86400,))
            db.execute(
                "DELETE FROM contacts WHERE at<? AND status NOT IN ('sending','unknown')",
                (now - 40 * 86400,),
            )
            db.execute(
                "DELETE FROM shares WHERE thought IN (SELECT id FROM thoughts WHERE expires<?)",
                (now - 7 * 86400,),
            )
            db.execute("DELETE FROM thoughts WHERE expires<?", (now - 7 * 86400,))
            db.execute(
                "DELETE FROM shares WHERE NOT EXISTS(SELECT 1 FROM thoughts WHERE id=shares.thought)"
            )
            # Keep recent completed facts for follow-up questions, never drop
            # unfinished commitments or delivery-unknown records silently.
            db.execute(
                "DELETE FROM tasks WHERE due<? AND status IN ('delivered','cancelled','expired')",
                (now - 40 * 86400,),
            )
