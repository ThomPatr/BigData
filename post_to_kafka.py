from kafka import KafkaProducer
import time
import random
import json
import csv
import config

# Let's create a Kafka producer
producer = KafkaProducer(
    bootstrap_servers=config.BOOTSTRAP_SERVER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8'))

# Read the CSV file and send each row as a message to the Kafka topic
def post_to_kafka():

    try:
        with open(config.PREDICTION_PATH + "/prediction.csv", newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            for row in rows:
                # Send the row as a message to the Kafka topic
                print(row)
                producer.send(config.TOPIC_NAME, value=row)
                print(f"\nInviato: {row}")
                time.sleep(random.randint(0, 5))
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

if __name__ == "__main__":
    time.sleep(10) # Wait for the Kafka topic to be created
    while True:
        post_to_kafka()
    