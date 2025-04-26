from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.tuning import TrainValidationSplitModel
from pyspark.sql.functions import expr, from_json, col
from pyspark.sql.types import StructType
from pyspark.sql.functions import when
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
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
    
    model = TrainValidationSplitModel.load(config.MODEL_PATH).bestModel

    kafka_df = (spark
                .readStream
                .format("kafka")
                .option("kafka.bootstrap.servers", "localhost:9092")
                .option("subscribe", "traffic-data") # Topic name
                .option("startingOffsets", "latest")
                .load())
    
    print("Schema del DataFrame di Kafka:")
    kafka_df.printSchema()

    # Parse the value from binary to string into kafka_json_df
    kafka_json_df = kafka_df.withColumn("value", expr("cast(value as string)"))
    
    with open("schema.json", "r") as f:
        schema_str = f.read()
    
    # Recreate the schema from the JSON string
    json_schema = StructType.fromJson(json.loads(schema_str))
    string_schema = StructType([StructField(el.name, StringType(), True) for el in json_schema])

    # Apply the schema to the payload to read the data: read the json payload from the column "value"
    streaming_df = kafka_df.selectExpr("CAST(value AS STRING) as json_string").select(from_json(col("json_string"), string_schema).alias("data")).select("data.*")

    for el in json_schema:
        if isinstance(el.dataType, IntegerType):
            streaming_df = streaming_df.withColumn(el.name, col(el.name).cast("int"))
        elif isinstance(el.dataType, DoubleType):
            streaming_df = streaming_df.withColumn(el.name, col(el.name).cast("double"))


    print("Schema del DataFrame di Kafka dopo la lettura del payload JSON:")
    streaming_df.printSchema()

    streaming_df = streaming_df.withColumn("flow_id_backup", col("Flow ID"))
    streaming_df = streaming_df.withColumn("label_verification", col("label_indexed"))

    # Apply the model to the streaming data
    prediction = model.transform(streaming_df)
    print("Schema del DataFrame di Kafka dopo la predizione:")
    prediction.printSchema()

    prediction = prediction.withColumn("Flow ID", col("flow_id_backup"))
    prediction = prediction.withColumn("label_indexed", col("label_verification"))
    prediction = prediction.select("Flow ID", "prediction", "label_indexed") # We added the label column to the prediction DataFrame for verification purposes

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
