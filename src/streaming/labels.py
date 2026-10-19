"""Publish real, externally observed labels from JSONL; never infer label latency."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from confluent_kafka import Producer

from src.streaming.config import LABELS, kafka_config
from src.streaming.events import encode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    producer = Producer({**kafka_config(), "enable.idempotence": True, "acks": "all"})
    failures = []

    def delivered(error, message):
        if error:
            failures.append(str(error))

    try:
        with args.input.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                tid = record.get("transaction_id")
                if isinstance(tid, bool) or not isinstance(tid, int):
                    raise ValueError("transaction_id must be an integer")
                if type(record.get("isFraud")) is not int or record["isFraud"] not in (0, 1):
                    raise ValueError("isFraud must be explicitly 0 or 1")
                observed = datetime.fromisoformat(record["label_observed_at"].replace("Z", "+00:00"))
                if observed.tzinfo is None:
                    raise ValueError("label_observed_at needs a timezone")
                record = {"schema_version": 1, "transaction_id": tid, "isFraud": record["isFraud"],
                          "label_observed_at": observed.isoformat()}
                while True:
                    try:
                        producer.produce(LABELS, key=str(tid), value=encode(record), on_delivery=delivered)
                        break
                    except BufferError:
                        producer.poll(0.1)
                producer.poll(0)
                if failures:
                    raise RuntimeError(failures[0])
    finally:
        remaining = producer.flush(30)
    if failures or remaining:
        raise RuntimeError(f"Label delivery incomplete: {len(failures)} errors, {remaining} pending")


if __name__ == "__main__":
    main()
