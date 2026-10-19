"""Versioned JSON contracts. Validation failures are safe to dead-letter."""

import json
import math
from datetime import datetime, timezone


class InvalidEvent(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def encode(value: dict) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")


def decode(raw: bytes) -> dict:
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    try:
        value = json.loads(raw, parse_constant=reject_constant)
    except (ValueError, UnicodeDecodeError) as exc:
        raise InvalidEvent("Event must be valid JSON with finite numbers") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise InvalidEvent("Expected an object with schema_version=1")
    return value


def validate_transaction(event: dict) -> dict:
    if event.get("schema_version") != 1:
        raise InvalidEvent("Unsupported schema_version")
    if not isinstance(event.get("event_id"), str) or not event["event_id"]:
        raise InvalidEvent("event_id must be a non-empty string")
    tx = event.get("transaction")
    if not isinstance(tx, dict):
        raise InvalidEvent("transaction must be an object")
    if "isFraud" in tx:
        raise InvalidEvent("isFraud belongs in the labels topic, never in scoring events")
    tid = tx.get("TransactionID")
    if isinstance(tid, bool) or not isinstance(tid, int):
        raise InvalidEvent("TransactionID must be an integer")
    dt = tx.get("TransactionDT")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt < 0:
        raise InvalidEvent("TransactionDT must be a finite non-negative number")
    for key, value in tx.items():
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise InvalidEvent(f"{key}: expected a scalar or null")
        if isinstance(value, float) and not math.isfinite(value):
            raise InvalidEvent(f"{key}: non-finite number")
    return tx
