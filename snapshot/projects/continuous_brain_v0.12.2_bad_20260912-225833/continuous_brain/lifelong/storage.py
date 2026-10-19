"""Single-writer, transactional lifetime and model persistence on Windows/WSL."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sqlite3

import torch


class LifetimeStore:
    def __init__(self, directory):
        self.directory = Path(directory).expanduser().resolve()
        # A new format must never be mixed into one of the existing child states.
        if any((self.directory / name).exists() for name in ("infant_training.pt", "individual.sqlite3")):
            raise ValueError("Choose a NEW v0.12 state directory; legacy state found here")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / "writer.lock").open("a+b")
        self.lock.seek(0)
        self.lock.write(b"0")
        self.lock.flush()
        self.lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError) as exc:
            self.lock.close()
            raise RuntimeError("This individual already has a running writer. Stop it first.") from exc
        try:
            self.connection = sqlite3.connect(self.directory / "lifetime.sqlite3")
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS current_state (
                    id INTEGER PRIMARY KEY CHECK (id=1), payload BLOB NOT NULL, summary TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS previous_state (
                    id INTEGER PRIMARY KEY CHECK (id=1), payload BLOB NOT NULL, summary TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, step INTEGER NOT NULL,
                    kind TEXT NOT NULL, provenance TEXT NOT NULL, content TEXT NOT NULL);
            """)
        except BaseException:
            self.lock.close()
            raise

    def load(self):
        row = self.connection.execute("SELECT payload FROM current_state WHERE id=1").fetchone()
        # Only the application's own local transaction store is deserialized.
        return torch.load(io.BytesIO(row[0]), map_location="cpu", weights_only=False) if row else None

    def commit(self, payload, summary, events):
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        text = json.dumps(summary, allow_nan=False)
        rows = [(int(e["step"]), e["kind"], e["provenance"], json.dumps(e["content"], allow_nan=False))
                for e in events]
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO previous_state SELECT * FROM current_state WHERE id=1")
            self.connection.execute("INSERT OR REPLACE INTO current_state VALUES (1, ?, ?)", (buffer.getvalue(), text))
            self.connection.executemany("INSERT INTO events (step,kind,provenance,content) VALUES (?,?,?,?)", rows)

    def snapshot(self, destination):
        destination = Path(destination).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=False)
        target = sqlite3.connect(destination / "lifetime.sqlite3")
        try:
            self.connection.backup(target)
        finally:
            target.close()

    def finish_save(self, expected_summary):
        """Verify the closed database before the CLI acknowledges a saved session.

        Keep the writer lock throughout. A clean shutdown folds the WAL into the
        main database, closes the writing connection, then reads the same state
        that a new process will load. A stale or incomplete save is an error,
        never a successful pause message.
        """
        expected = json.loads(json.dumps(expected_summary, allow_nan=False))
        row = self.connection.execute("SELECT summary FROM current_state WHERE id=1").fetchone()
        if row is None or json.loads(row[0]) != expected:
            raise RuntimeError("Save verification failed: the committed state differs from the session")
        busy, _, _ = self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if busy:
            raise RuntimeError("Save verification could not finish: another database reader is still active")
        self.connection.close()
        self.connection = None
        if read_summary(self.directory) != expected:
            raise RuntimeError("Save verification failed: reopening the database returned a different state")

    def close(self):
        if getattr(self, "connection", None) is not None:
            self.connection.close()
            self.connection = None
        if not self.lock.closed:
            self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def read_summary(directory):
    path = Path(directory).expanduser().resolve() / "lifetime.sqlite3"
    if not path.is_file():
        raise FileNotFoundError("No local individual at " + str(path.parent))
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        row = connection.execute("SELECT summary FROM current_state WHERE id=1").fetchone()
        return json.loads(row[0]) if row else {"status": "initializing"}
    finally:
        connection.close()


def read_events(directory, limit=10):
    path = Path(directory).expanduser().resolve() / "lifetime.sqlite3"
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        return [{"step": s, "kind": k, "provenance": p, "content": json.loads(c)}
                for s, k, p, c in connection.execute(
                    "SELECT step,kind,provenance,content FROM events ORDER BY id DESC LIMIT ?", (int(limit),))]
    finally:
        connection.close()
