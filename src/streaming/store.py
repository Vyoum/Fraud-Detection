"""Idempotent SQLite projection for the dashboard; Kafka remains the event log."""

import argparse
import json
import sqlite3
from pathlib import Path

from confluent_kafka import Consumer, KafkaException

from src.streaming.config import DECISIONS, decision_db, kafka_config
from src.streaming.events import decode


def open_store(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("""CREATE TABLE IF NOT EXISTS decisions (
        decision_id TEXT PRIMARY KEY, transaction_id INTEGER NOT NULL,
        action TEXT NOT NULL, score REAL NOT NULL, received_at TEXT NOT NULL,
        payload TEXT NOT NULL)""")
    connection.commit()
    return connection


def save_decision(connection, record: dict):
    # Commit BEFORE acknowledging Kafka. Crash/replay repeats this insert,
    # whose primary key makes delivery idempotent even across replay runs.
    with connection:
        connection.execute("INSERT OR IGNORE INTO decisions VALUES (?, ?, ?, ?, ?, ?)", (
            record["decision_id"], record["transaction_id"], record["action"],
            record["score"], record["timestamp"], json.dumps(record, allow_nan=False),
        ))


def read_decisions(path: Path, limit: int = 200) -> tuple[dict, list[dict]]:
    if not path.exists():
        return {}, []
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    try:
        counts = dict(connection.execute("SELECT action, count(*) FROM decisions GROUP BY action"))
        rows = connection.execute("SELECT payload FROM decisions ORDER BY rowid DESC LIMIT ?", (limit,))
        return counts, [json.loads(row[0]) for row in rows]
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=decision_db())
    parser.add_argument("--group", default="fraud-dashboard-v1")
    args = parser.parse_args()
    connection = open_store(args.db)
    consumer = Consumer({**kafka_config(), "group.id": args.group, "enable.auto.commit": False,
                         "auto.offset.reset": "earliest", "isolation.level": "read_committed"})
    consumer.subscribe([DECISIONS])
    print(f"Projecting {DECISIONS} into {args.db}", flush=True)
    try:
        while True:
            message = consumer.poll(1)
            if message is None:
                continue
            if message.error():
                raise KafkaException(message.error())
            save_decision(connection, decode(message.value()))
            consumer.commit(message=message, asynchronous=False)
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        connection.close()


if __name__ == "__main__":
    main()
