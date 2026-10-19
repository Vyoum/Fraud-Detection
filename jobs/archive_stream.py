"""Archive Kafka records to HDFS Parquet, retaining Kafka coordinates and raw JSON.

Run with spark-submit and the matching spark-sql-kafka package (see Compose).
Each query owns its checkpoint; retain both checkpoint and output together.
"""

import argparse

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType, StructField, StructType


def archive_rows(raw):
    schema = StructType([
        StructField("schema_version", IntegerType()),
        StructField("event_id", StringType()),
        StructField("decision_id", StringType()),
        StructField("transaction_id", LongType()),
        StructField("transaction_dt", DoubleType()),
        StructField("label_observed_at", StringType()),
        StructField("isFraud", IntegerType()),
        StructField("error", StringType()),
        StructField("transaction", StructType([
            StructField("TransactionID", LongType()),
            StructField("TransactionDT", DoubleType()),
        ])),
    ])
    rows = raw.select("topic", "partition", "offset", F.col("timestamp").alias("kafka_timestamp"),
                      F.col("key").cast("string").alias("key"),
                      F.col("value").cast("string").alias("payload"))
    rows = rows.withColumn("parsed", F.from_json("payload", schema))
    rows = rows.withColumn("event_id", F.col("parsed.event_id"))
    rows = rows.withColumn("transaction_id", F.coalesce("parsed.transaction.TransactionID", "parsed.transaction_id"))
    rows = rows.withColumn("transaction_dt", F.coalesce("parsed.transaction.TransactionDT", "parsed.transaction_dt"))
    valid = (F.col("parsed.schema_version") == 1) & (
        ((F.col("topic") == "fraud.transactions") & F.col("event_id").isNotNull()
         & F.col("transaction_id").isNotNull() & (F.col("transaction_dt") >= 0))
        | ((F.col("topic") == "fraud.decisions") & F.col("parsed.decision_id").isNotNull())
        | ((F.col("topic") == "fraud.labels") & F.col("transaction_id").isNotNull()
           & F.col("parsed.isFraud").isin(0, 1) & F.to_timestamp("parsed.label_observed_at").isNotNull())
        | ((F.col("topic") == "fraud.errors") & F.col("parsed.error").isNotNull())
    )
    rows = rows.withColumn("envelope_valid", F.coalesce(valid, F.lit(False)))
    # IEEE-CIS uses relative seconds, not Unix time. Do not invent a calendar date.
    rows = rows.withColumn("event_day", F.coalesce(F.floor(F.col("transaction_dt") / 86400).cast("long"), F.lit(-1)))
    return rows.drop("parsed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-servers", default="kafka:19092")
    parser.add_argument("--output", default="hdfs://namenode:9000/fraud/events")
    parser.add_argument("--checkpoint", default="hdfs://namenode:9000/fraud/checkpoints/events-v1")
    parser.add_argument("--available-now", action="store_true", help="Drain current backlog and exit")
    args = parser.parse_args()
    spark = SparkSession.builder.appName("fraud-kafka-archive").getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    try:
        raw = (spark.readStream.format("kafka")
               .option("kafka.bootstrap.servers", args.bootstrap_servers)
               .option("subscribe", "fraud.transactions,fraud.decisions,fraud.labels,fraud.errors")
               .option("kafka.isolation.level", "read_committed")
               .option("startingOffsets", "earliest")
               .option("failOnDataLoss", "true")
               .option("maxOffsetsPerTrigger", 10000).load())
        writer = (archive_rows(raw).writeStream.format("parquet").outputMode("append")
                  .option("path", args.output).option("checkpointLocation", args.checkpoint)
                  .partitionBy("topic", "event_day"))
        writer = writer.trigger(availableNow=True) if args.available_now else writer.trigger(processingTime="30 seconds")
        writer.start().awaitTermination()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
