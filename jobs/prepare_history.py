"""Flatten archived observations and join explicit, available fraud labels.

No missing label is treated as non-fraud. The cutoff controls which labels
are known to this batch, not per-training-row causal feature computation.
Output is a new immutable Parquet history accepted by streaming.export.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from pyspark.sql import SparkSession, Window, functions as F
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType


def prepare_history(archive, manifest: dict, labels_as_of: str):
    fields = [StructField("TransactionID", LongType())]
    for name in manifest["input_columns"]:
        spec = manifest["schema"][name]
        # IEEE-CIS categories are text. Preserve numeric categories if a future
        # bundle explicitly records numeric values.
        numeric = spec["kind"] == "numeric" or (
            spec.get("categories") and isinstance(spec["categories"][0], (int, float)))
        fields.append(StructField(name, DoubleType() if numeric else StringType()))
    schema = StructType(fields)
    transactions = archive.filter((F.col("topic") == "fraud.transactions") & F.col("envelope_valid"))
    transactions = transactions.withColumn("tx", F.from_json(F.get_json_object("payload", "$.transaction"), schema))
    ordering = Window.partitionBy("tx.TransactionID").orderBy("kafka_timestamp", "partition", "offset")
    transactions = transactions.withColumn("rn", F.row_number().over(ordering)).filter("rn = 1").select("tx.*")

    labels = archive.filter((F.col("topic") == "fraud.labels") & F.col("envelope_valid"))
    labels = labels.select(
        F.get_json_object("payload", "$.transaction_id").cast("long").alias("TransactionID"),
        F.get_json_object("payload", "$.isFraud").cast("int").alias("isFraud"),
        F.to_timestamp(F.get_json_object("payload", "$.label_observed_at")).alias("observed_at"),
        "kafka_timestamp", "partition", "offset",
    ).filter(F.col("observed_at") < F.to_timestamp(F.lit(labels_as_of)))
    ordering = Window.partitionBy("TransactionID").orderBy(
        F.col("observed_at").desc(), F.col("kafka_timestamp").desc(), F.col("partition").desc(), F.col("offset").desc())
    labels = labels.withColumn("rn", F.row_number().over(ordering)).filter("rn = 1")
    return transactions.join(labels.select("TransactionID", "isFraud"), "TransactionID", "inner")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="hdfs://namenode:9000/fraud/events")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels-as-of", required=True, help="Exclusive ISO-8601 label availability cutoff, with timezone")
    parser.add_argument("--output", required=True, help="New hdfs:// or file:// directory")
    args = parser.parse_args()
    try:
        if datetime.fromisoformat(args.labels_as_of.replace("Z", "+00:00")).tzinfo is None:
            raise ValueError("timezone required")
    except ValueError:
        parser.error("--labels-as-of must be an ISO-8601 timestamp with a timezone")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    spark = SparkSession.builder.appName("fraud-prepare-history").getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    try:
        history = prepare_history(spark.read.parquet(args.archive), manifest, args.labels_as_of)
        if history.limit(1).count() == 0:
            raise ValueError("No explicitly labeled transactions available before cutoff")
        history.write.mode("errorifexists").parquet(args.output)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
