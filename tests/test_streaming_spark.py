"""Optional local Spark transformation checks; no Kafka, HDFS, or generated abuse."""

import json
import os
from datetime import datetime

import pytest

if os.environ.get("RUN_SPARK_TESTS") != "1":
    pytest.skip("Set RUN_SPARK_TESTS=1 with Java and PySpark installed", allow_module_level=True)

pytest.importorskip("pyspark")
from pyspark.sql import SparkSession
from jobs.archive_stream import archive_rows
from jobs.prepare_history import prepare_history


@pytest.fixture(scope="module")
def spark():
    session = (SparkSession.builder.master("local[1]").appName("fraud-streaming-tests")
               .config("spark.sql.shuffle.partitions", "1").getOrCreate())
    session.conf.set("spark.sql.session.timeZone", "UTC")
    yield session
    session.stop()


def test_archive_and_label_cutoff(spark):
    observation = {"schema_version": 1, "event_id": "fixture:1", "transaction": {
        "TransactionID": 1, "TransactionDT": 86400, "card1": 1,
    }}
    label = {"schema_version": 1, "transaction_id": 1, "isFraud": 0,
             "label_observed_at": "2026-01-01T00:00:00Z"}
    late_label = {**label, "isFraud": 1, "label_observed_at": "2026-01-03T00:00:00Z"}
    # Duplicate delivery and later label correction exercise data mechanics.
    payloads = [("fraud.transactions", observation), ("fraud.transactions", observation),
                ("fraud.labels", label), ("fraud.labels", late_label), ("fraud.transactions", {})]
    rows = [(topic, 0, i, datetime(2026, 1, 1), b"key", json.dumps(payload).encode())
            for i, (topic, payload) in enumerate(payloads)]
    raw = spark.createDataFrame(rows, "topic string, partition int, offset long, timestamp timestamp, key binary, value binary")
    archive = archive_rows(raw)
    assert archive.count() == 5
    assert archive.filter("NOT envelope_valid").count() == 1
    assert archive.filter("topic = 'fraud.transactions' AND envelope_valid").first()["event_day"] == 1
    manifest = {"input_columns": ["TransactionDT", "card1"],
                "schema": {"TransactionDT": {"kind": "numeric"}, "card1": {"kind": "numeric"}}}
    history = prepare_history(archive, manifest, "2026-01-02T00:00:00Z").collect()
    assert len(history) == 1
    assert history[0]["isFraud"] == 0
    assert prepare_history(archive, manifest, "2026-01-01T00:00:00Z").count() == 0
