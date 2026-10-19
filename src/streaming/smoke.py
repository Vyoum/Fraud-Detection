"""One real Kafka round trip, comparing the worker result with local scoring."""

import argparse
import time
import uuid
from pathlib import Path

from confluent_kafka import Consumer, KafkaException, Producer, TopicPartition

from src.streaming.artifacts import ScoringBundle
from src.streaming.config import DECISIONS, TRANSACTIONS, kafka_config
from src.streaming.events import decode, encode
from src.streaming.producer import replay_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    key, event = next(replay_events(args.bundle, 1))
    expected = ScoringBundle(args.bundle).score(event)
    consumer = Consumer({**kafka_config(), "group.id": f"fraud-smoke-{uuid.uuid4().hex}",
                         "enable.auto.commit": False, "isolation.level": "read_committed"})
    producer = Producer({**kafka_config(), "enable.idempotence": True})
    try:
        metadata = consumer.list_topics(DECISIONS, timeout=15)
        topic = metadata.topics[DECISIONS]
        if topic.error:
            raise KafkaException(topic.error)
        positions = []
        for partition in topic.partitions:
            _, high = consumer.get_watermark_offsets(TopicPartition(DECISIONS, partition), timeout=15)
            positions.append(TopicPartition(DECISIONS, partition, high))
        consumer.assign(positions)
        delivered = []
        producer.produce(TRANSACTIONS, key=key, value=encode(event),
                         on_delivery=lambda err, msg: delivered.append((err, msg)))
        remaining = producer.flush(30)
        if remaining or not delivered or delivered[0][0]:
            raise RuntimeError("Smoke observation was not delivered")
        source_message = delivered[0][1]
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            message = consumer.poll(1)
            if message is None:
                continue
            if message.error():
                raise KafkaException(message.error())
            record = decode(message.value())
            source = record.get("source", {})
            if source == {"topic": TRANSACTIONS, "partition": source_message.partition(), "offset": source_message.offset()}:
                if (record["decision_id"] != expected["decision_id"]
                        or record["action"] != expected["action"]
                        or abs(record["score"] - expected["score"]) > 1e-12):
                    raise AssertionError("Worker result differs from local bundle scoring")
                print(f"PASS: transaction {record['transaction_id']}, score={record['score']:.8f}, action={record['action']}")
                return
        raise TimeoutError("No matching decision: check scorer logs, bundle path, and fraud.errors")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
