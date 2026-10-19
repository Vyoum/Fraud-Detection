"""Transactional Kafka consume -> score -> decision (or error) -> offset commit."""

import argparse
import base64
from pathlib import Path

from confluent_kafka import Consumer, KafkaException, Producer, TopicPartition

from src.streaming.artifacts import ScoringBundle
from src.streaming.config import DECISIONS, ERRORS, TRANSACTIONS, kafka_config
from src.streaming.events import InvalidEvent, decode, encode, utc_now


def process_message(message, bundle, consumer, producer):
    """Commit output and consumed offset in ONE Kafka transaction.

    Invalid input is dead-lettered. Infrastructure/model failures propagate,
    aborting the transaction so a restart can retry the uncommitted record.
    Consumers of these outputs must use isolation.level=read_committed.
    """
    raw = message.value()
    source = {"topic": message.topic(), "partition": message.partition(), "offset": message.offset()}
    try:
        event = decode(raw or b"")
        output = bundle.score(event)
        output["source"] = source
        topic = DECISIONS
        key = output["decision_id"]
    except InvalidEvent as exc:
        output = {"schema_version": 1, "error": str(exc), "source": source,
                  "raw_base64": base64.b64encode(raw or b"").decode("ascii"),
                  "timestamp": utc_now()}
        topic = ERRORS
        key = f"{message.topic()}:{message.partition()}:{message.offset()}"

    producer.begin_transaction()
    try:
        producer.produce(topic, key=key, value=encode(output))
        producer.send_offsets_to_transaction(
            [TopicPartition(message.topic(), message.partition(), message.offset() + 1)],
            consumer.consumer_group_metadata(),
        )
        producer.commit_transaction()
    except BaseException:
        producer.abort_transaction()
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--group", default="fraud-scorers-v1")
    parser.add_argument("--worker-id", default="scorer-1", help="Stable and unique for each concurrent worker")
    args = parser.parse_args()
    bundle = ScoringBundle(args.bundle)
    consumer = Consumer({**kafka_config(), "group.id": args.group, "enable.auto.commit": False,
                         "auto.offset.reset": "earliest", "isolation.level": "read_committed"})
    producer = Producer({**kafka_config(), "transactional.id": f"{args.group}-{args.worker_id}",
                         "enable.idempotence": True, "acks": "all"})
    try:
        producer.init_transactions()
        consumer.subscribe([TRANSACTIONS])
        print(f"Scoring {TRANSACTIONS} with model {bundle.manifest['model_version'][:12]}", flush=True)
        while True:
            message = consumer.poll(1)
            if message is None:
                continue
            if message.error():
                raise KafkaException(message.error())
            output = process_message(message, bundle, consumer, producer)
            print(f"{output.get('transaction_id', 'invalid')} {output.get('action', output.get('error'))}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
