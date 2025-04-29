from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import rand
import os
import shutil
import json
import config
from classification import Classification
from preprocessing import Preprocessing
from methods import Methods

if __name__=='__main__':

    methods = Methods()

    # Create the Spark session
    spark = (SparkSession.builder
                .appName("InSDN Analysis")
                #.config("spark.driver.memory", "12g")
                #.config("spark.executor.memory", "12g")
                .config("spark.sql.debug.maxToStringFields", 1000) # To avoid truncation of long strings
                .getOrCreate())

    data = methods.load_data(spark, config.PATH)

    print("\n###### DATA PREPRATION ######")
    print("\n1. Data Cleaning")
    preprocessor = Preprocessing(data)
    data_cleaned = preprocessor.clean_data()

    label_indexer = StringIndexer(inputCol="Label", outputCol="label_indexed")
    data_cleaned = label_indexer.fit(data_cleaned).transform(data_cleaned) # Transform the label column into a numerical column
    # Divide the dataset into train, test and prediction sets (the prediction set will be used to test streaming)
    print("\nSuddividiamo il dataset in tre porzioni per il training del modello, il testing delle performance e la predizione in real-time.")
    train, test, prediction = methods.stratified_split(data_cleaned, "Label", (0.8, 0.18, 0.02), seed=42)

    json_schema = prediction.schema.json() # Get the schema of the DataFrame in JSON format
    with open("schema.json", "w") as f:
        f.write(json_schema)

    # prediction = prediction.drop("Label") # Drop the label column from the prediction set
    prediction = prediction.orderBy(rand())

    temp_dir = "new_datasets/streaming_temp"
    prediction.write.mode("overwrite").options(header=True).csv(temp_dir) # Save the prediction set to a CSV file
    final_dir = "new_datasets/streaming"
    final_filename = "prediction.csv"

    os.makedirs(final_dir, exist_ok=True) # Create the directory if it doesn't exist
    for file in os.listdir(temp_dir):
        if file.endswith(".csv"):
            shutil.move(os.path.join(temp_dir, file), os.path.join(final_dir, final_filename))
    shutil.rmtree(temp_dir) # Remove the temporary directory

    print(f"Il numero totale di record nel dataset di training è: {train.count()}.")
    print(f"Il numero totale di record nel dataset di test è: {test.count()}.")
    print(f"Il numero totale di record nel dataset di prediction è: {prediction.count()}.")

    print("\n2. Feature Selection and Data Balancing")
    selected_features = preprocessor.preprocessing(spark, train)

    # This is the train set that will be used to train the model
    balanced_train = methods.load_data(spark, config.BALANCED_TRAIN_PATH)

    print("\n###### DATA ANALYSIS ######")
    classificator = Classification()
    classificator.best_model(balanced_train, test, selected_features)
    


