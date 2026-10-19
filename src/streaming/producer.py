"""Replay real held-out observations from an exported bundle into Kafka."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from confluent_kafka import Producer

from src import entities
from src.streaming.artifacts import file_hash
from src.streaming.config import TRANSACTIONS, kafka_config
from src.streaming.events import encode, validate_transaction


def replay_events(bundle: Path, limit: int | None = None):
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    replay_path = bundle / "replay.parquet"
    if file_hash(replay_path) != manifest["files"]["replay.parquet"]:
        raise ValueError("Replay checksum mismatch")
    frame = pd.read_parquet(replay_path).sort_values(["TransactionDT", "TransactionID"], kind="stable")
    if limit is not None:
        frame = frame.head(limit)
    uids = entities.resolve_entities(frame)
    # pandas converts missing values to JSON null, without non-standard NaNs.
    # double_precision=15 preserves float32 values from the research loader.
    for tx in json.loads(frame.to_json(orient="records", double_precision=15)):
        event = {"schema_version": 1, "event_id": f"ieee-cis:{tx['TransactionID']}",
                 "transaction": tx}
        validate_transaction(event)
        uid = uids.loc[tx["TransactionID"]]
        key = str(uid) if pd.notna(uid) else str(tx["TransactionID"])
        yield key, event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rate", type=float, default=10, help="Transactions/second; 0 means unthrottled")
    args = parser.parse_args()
    if args.rate < 0 or (args.limit is not None and args.limit < 1):
        parser.error("rate must be non-negative and limit must be positive")
    producer = Producer({**kafka_config(), "enable.idempotence": True, "acks": "all"})
    failures = []

    def delivered(error, message):
        if error:
            failures.append(str(error))

    count = 0
    try:
        for key, event in replay_events(args.bundle, args.limit):
            while True:
                try:
                    producer.produce(TRANSACTIONS, key=key, value=encode(event), on_delivery=delivered)
                    break
                except BufferError:
                    producer.poll(0.1)
            producer.poll(0)
            if failures:
                raise RuntimeError(f"Kafka delivery failed: {failures[0]}")
            count += 1
            if args.rate:
                time.sleep(1 / args.rate)
    finally:
        remaining = producer.flush(30)
    if failures or remaining:
        raise RuntimeError(f"Replay incomplete: {len(failures)} failures, {remaining} undelivered messages")
    print(f"Delivered {count} real held-out transactions to {TRANSACTIONS}")


if __name__ == "__main__":
    main()
