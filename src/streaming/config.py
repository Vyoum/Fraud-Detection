"""Shared defaults; Kafka settings can also be supplied through a JSON file."""

import json
import os
from pathlib import Path

TRANSACTIONS = "fraud.transactions"
DECISIONS = "fraud.decisions"
LABELS = "fraud.labels"
ERRORS = "fraud.errors"
TOPICS = (TRANSACTIONS, DECISIONS, LABELS, ERRORS)
ROOT = Path(__file__).resolve().parents[2]


def kafka_config() -> dict:
    config = {}
    if filename := os.environ.get("KAFKA_CONFIG_FILE"):
        config = json.loads(Path(filename).read_text(encoding="utf-8"))
    config["bootstrap.servers"] = os.environ.get(
        "KAFKA_BOOTSTRAP_SERVERS", config.get("bootstrap.servers", "localhost:9092")
    )
    return config


def decision_db() -> Path:
    return Path(os.environ.get("FRAUD_DECISION_DB", str(ROOT / "runtime" / "decisions.sqlite")))
