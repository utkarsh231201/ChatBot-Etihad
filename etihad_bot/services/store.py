"""SQLite store for LLM-call logs, interaction records and user feedback."""
import json
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager

from etihad_bot.config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, user_id TEXT, conversation_id TEXT, purpose TEXT, model TEXT,
    input_tokens INTEGER, output_tokens INTEGER, latency_ms REAL,
    cost_usd REAL, status TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS interactions (
    message_id TEXT PRIMARY KEY,
    ts REAL, user_id TEXT, conversation_id TEXT, question TEXT, rewritten_query TEXT,
    intent TEXT, answer TEXT, status TEXT,
    chunks_json TEXT, citations_json TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, user_id TEXT, conversation_id TEXT, message_id TEXT,
    rating INTEGER, comment TEXT, question TEXT, answer TEXT, chunks_json TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY, user_id TEXT, created REAL, expires REAL
);
CREATE INDEX IF NOT EXISTS idx_calls_conv ON llm_calls(conversation_id);
CREATE INDEX IF NOT EXISTS idx_inter_conv ON interactions(conversation_id);
"""

_lock = threading.Lock()
_initialised: set[str] = set()


@contextmanager
def conn():
    path = get_settings().app_db
    with _lock:
        c = sqlite3.connect(path)
        c.row_factory = sqlite3.Row
        if path not in _initialised:
            c.executescript(SCHEMA)
            _initialised.add(path)
        try:
            yield c
            c.commit()
        finally:
            c.close()


def log_llm_call(*, user_id, conversation_id, purpose, model, input_tokens, output_tokens,
                 latency_ms, cost_usd, status, error=None):
    with conn() as c:
        c.execute(
            "INSERT INTO llm_calls (ts,user_id,conversation_id,purpose,model,input_tokens,"
            "output_tokens,latency_ms,cost_usd,status,error) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), user_id, conversation_id, purpose, model, input_tokens,
             output_tokens, latency_ms, cost_usd, status, error),
        )


def save_interaction(*, message_id, user_id, conversation_id, question, rewritten_query, intent,
                     answer, status, chunks, citations):
    with conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO interactions (message_id,ts,user_id,conversation_id,question,"
            "rewritten_query,intent,answer,status,chunks_json,citations_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (message_id, time.time(), user_id, conversation_id, question, rewritten_query,
             intent, answer, status, json.dumps(chunks), json.dumps(citations)),
        )


def save_feedback(*, message_id: str, rating: int, comment: str = "") -> bool:
    """Store a thumbs up (1) / down (-1) together with the chunks that produced the answer."""
    with conn() as c:
        row = c.execute("SELECT * FROM interactions WHERE message_id=?", (message_id,)).fetchone()
        if row is None:
            return False
        c.execute(
            "INSERT INTO feedback (ts,user_id,conversation_id,message_id,rating,comment,"
            "question,answer,chunks_json) VALUES (?,?,?,?,?,?,?,?,?)",
            (time.time(), row["user_id"], row["conversation_id"], message_id, rating, comment,
             row["question"], row["answer"], row["chunks_json"]),
        )
        return True


def list_conversations(user_id: str) -> list[dict]:
    with conn() as c:
        rows = c.execute(
            "SELECT conversation_id, MIN(ts) AS started, MAX(ts) AS last, COUNT(*) AS turns, "
            "(SELECT question FROM interactions i2 WHERE i2.conversation_id=i.conversation_id "
            " ORDER BY ts LIMIT 1) AS first_question "
            "FROM interactions i WHERE user_id=? GROUP BY conversation_id ORDER BY last DESC",
            (user_id,)).fetchall()
        return [dict(r) for r in rows]


# ---- sessions (sign-in tokens) ------------------------------------------------------------
def create_session(user_id: str, ttl_seconds: int = 24 * 3600) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    with conn() as c:
        c.execute("DELETE FROM sessions WHERE expires < ?", (now,))
        c.execute("INSERT INTO sessions VALUES (?,?,?,?)", (token, user_id, now, now + ttl_seconds))
    return token


def user_for_token(token: str) -> str | None:
    with conn() as c:
        row = c.execute("SELECT user_id FROM sessions WHERE token=? AND expires > ?",
                        (token, time.time())).fetchone()
        return row["user_id"] if row else None


def delete_session(token: str) -> None:
    with conn() as c:
        c.execute("DELETE FROM sessions WHERE token=?", (token,))


# ---- ownership + deletion -----------------------------------------------------------------
def conversation_owner(conversation_id: str) -> str | None:
    with conn() as c:
        row = c.execute("SELECT user_id FROM interactions WHERE conversation_id=? LIMIT 1",
                        (conversation_id,)).fetchone()
        return row["user_id"] if row else None


def message_owner(message_id: str) -> str | None:
    with conn() as c:
        row = c.execute("SELECT user_id FROM interactions WHERE message_id=?",
                        (message_id,)).fetchone()
        return row["user_id"] if row else None


def delete_conversation_data(conversation_id: str) -> int:
    """Remove the stored messages and feedback of a conversation. `llm_calls` (token/cost
    accounting, no message text) is kept."""
    with conn() as c:
        c.execute("DELETE FROM feedback WHERE conversation_id=?", (conversation_id,))
        return c.execute("DELETE FROM interactions WHERE conversation_id=?",
                         (conversation_id,)).rowcount

