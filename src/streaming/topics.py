"""Create the four Kafka topics (safe to run more than once)."""

import argparse

from confluent_kafka import KafkaException
from confluent_kafka.admin import AdminClient, NewTopic

from src.streaming.config import TOPICS, kafka_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partitions", type=int, default=1)
    parser.add_argument("--replication-factor", type=int, default=1)
    args = parser.parse_args()
    if args.partitions < 1 or args.replication_factor < 1:
        parser.error("partitions and replication factor must be positive")
    admin = AdminClient(kafka_config())
    futures = admin.create_topics([
        NewTopic(name, num_partitions=args.partitions, replication_factor=args.replication_factor)
        for name in TOPICS
    ])
    from confluent_kafka import KafkaError
    for name, future in futures.items():
        try:
            future.result()
            print(f"Created {name}")
        except KafkaException as exc:
            if exc.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
                raise
            print(f"Already exists: {name}")


if __name__ == "__main__":
    main()
