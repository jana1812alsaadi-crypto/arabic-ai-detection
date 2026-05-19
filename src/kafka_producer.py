# Kafka producer that simulates a live stream of Arabic abstracts.
# Reads a CSV (specified column) and sends one message per (1/rate) seconds.

import argparse
import json
import time
import csv
from kafka import KafkaProducer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/by_polishing.csv",
                        help="Input CSV file path")
    parser.add_argument("--topic", default="arabic-texts",
                        help="Kafka topic name")
    parser.add_argument("--bootstrap", default="localhost:9092",
                        help="Kafka bootstrap servers")
    parser.add_argument("--rate", type=float, default=1.0,
                        help="Messages per second")
    parser.add_argument("--limit", type=int, default=100,
                        help="Maximum messages to send")
    parser.add_argument("--column", default="original_abstract",
                        help="CSV column to send as message text")
    args = parser.parse_args()

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
    )

    sleep_per_msg = 1.0 / args.rate if args.rate > 0 else 0

    print(f"Producer started")
    print(f"  File:    {args.input}")
    print(f"  Topic:   {args.topic}")
    print(f"  Rate:    {args.rate} msg/sec")
    print(f"  Limit:   {args.limit} messages")

    sent = 0
    start = time.time()

    with open(args.input, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if sent >= args.limit:
                break
            text = row.get(args.column, "").strip()
            if not text:
                continue

            # Tag the true source so we can measure streaming accuracy later.
            payload = {
                "id": sent,
                "text": text,
                "true_source": "human" if "original" in args.column else "ai",
                "timestamp": time.time(),
            }
            producer.send(args.topic, payload)
            sent += 1

            if sent % 10 == 0:
                print(f"  Sent {sent} messages...")
            time.sleep(sleep_per_msg)

    producer.flush()
    elapsed = time.time() - start
    print(f"Done: sent {sent} messages in {elapsed:.1f}s ({sent/elapsed:.1f} msg/s)")
    producer.close()


if __name__ == "__main__":
    main()
