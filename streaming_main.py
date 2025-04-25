from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql.functions import expr, from_json, col
from pyspark.sql.types import StructType
from pyspark.sql.functions import when
import json

import config

if __name__ == '__main__':

    spark = (SparkSession
             .builder
             .appName("Real-time Intrusion Detection System")
             .config("spark.streaming.stopGracefullyOnShutdown", True)
             .config("spark.sql.streaming.schemeInference", True)
             #.config("spark.jars.packages", "org.apache.spark:spark-streaming-kafka-0-10_2.12:3.3.0")
             .config("spark.sql.shuffle.partitions", 4)
             .master("local[*]")
             .getOrCreate())
    
    model = PipelineModel.load(config.MODEL_PATH + "/bestModel")

    kafka_df = (spark
                .readStream
                .format("kafka")
                .option("kafka.bootstrap.servers", "localhost:9092")
                .option("subscribe", "traffic-data") # Topic name
                .option("startingOffsets", "earliest") # Start from the earliest message
                .load())
    
    kafka_df.printSchema()

    # Parse the value from binary to string into kafka_json_df
    kafka_json_df = kafka_df.withColumn("value", expr("cast(value as string)"))
    
    with open("schema.json", "r") as f:
        schema_str = f.read()
    
    # Recreate the schema from the JSON string
    json_schema = StructType.fromJson(json.loads(schema_str))

    # Apply the schema to the payload to read the data: read the json payload from the column "value"
    selected_columns = ['Flow ID', 'Protocol', 'Src Port', 'Fwd Pkts/s', 'Bwd Header Len', 'Flow Duration', 'Init Bwd Win Byts', 'Flow IAT Max', 'Bwd Pkts/s', 'Subflow Fwd Pkts', 'Tot Fwd Pkts', 'Fwd Header Len', 'Flow Pkts/s', 'Flow IAT Std', 'Dst Port', 'Flow IAT Mean']
    streaming_df = kafka_json_df.withColumn("values_json", from_json(col("value"), json_schema)).selectExpr("values_json.*").select(selected_columns) # Select all the columns from the JSON payload
    streaming_df = streaming_df.fillna(0) # Fill the null values with 0
    streaming_df.printSchema()

    streaming_df = streaming_df.withColumn("flow_id_backup", col("Flow ID"))

    # Apply the model to the streaming data
    prediction = model.transform(streaming_df)
    prediction.printSchema()

    prediction = prediction.withColumn("Flow ID", col("flow_id_backup"))
    prediction = prediction.select("Flow ID", "prediction")

    prediction = prediction.withColumn("message", when(col("prediction") != 3.0, "Attenzione! Intrusione rilevata!").otherwise("Nessuna intrusione rilevata."))

    query = (prediction
             .writeStream
             .outputMode("append")
             .format("console")
             .option("truncate", "false") # Do not truncate the output
             .trigger(processingTime="10 seconds")
             .start())

    try:
        query.awaitTermination() # Wait for the termination of the query
    except Exception as e:
        print(f"Si è verificato un errore: {e}")


# Comando per eseguire lo script
""" spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0 \
  streaming_main.py """
