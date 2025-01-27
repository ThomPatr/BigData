from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql.functions import col

if __name__ == '__main__':
    spark = (SparkSession.builder
             .appName("InSDN")
             .getOrCreate())

    # Load dataset
    input_directory = "/home/ilaria/Scaricati/InSDN_DatasetCSV"
    data = spark.read.csv(input_directory, header=True, inferSchema=True)
    data.show()
    data = data.withColumnRenamed("brute-force-attack", "label")  # Adjust "label_column" to the dataset's label column

    # Preprocessing
    features = [col for col in data.columns if col != "label"]
    assembler = VectorAssembler(inputCols=features, outputCol="featuresAssembled")
    scaler = StandardScaler(inputCol="featuresAssembled", outputCol="features_standard")

    # Split dataset into train and test
    train, test = data.randomSplit([0.8, 0.2], seed=42)

    # Models and pipelines
    rf = RandomForestClassifier(featuresCol="features_standard", labelCol="label")
    lr = LogisticRegression(featuresCol="features_standard", labelCol="label")
    gbt = GBTClassifier(featuresCol="features_standard", labelCol="label")

    rf_pipeline = Pipeline(stages=[assembler, scaler, rf])
    lr_pipeline = Pipeline(stages=[assembler, scaler, lr])
    gbt_pipeline = Pipeline(stages=[assembler, scaler, gbt])

    # Train-validation split setup
    paramGrid_rf = ParamGridBuilder().addGrid(rf.numTrees, [10, 50]).build()
    tvs_rf = TrainValidationSplit(estimator=rf_pipeline, estimatorParamMaps=paramGrid_rf,
                                  evaluator=MulticlassClassificationEvaluator(labelCol="label",
                                                                              predictionCol="prediction",
                                                                              metricName="accuracy"),
                                  trainRatio=0.8)

    paramGrid_lr = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01]).build()
    tvs_lr = TrainValidationSplit(estimator=lr_pipeline, estimatorParamMaps=paramGrid_lr,
                                  evaluator=MulticlassClassificationEvaluator(labelCol="label",
                                                                              predictionCol="prediction",
                                                                              metricName="accuracy"),
                                  trainRatio=0.8)

    paramGrid_gbt = ParamGridBuilder().addGrid(gbt.maxDepth, [5, 10]).build()
    tvs_gbt = TrainValidationSplit(estimator=gbt_pipeline, estimatorParamMaps=paramGrid_gbt,
                                   evaluator=MulticlassClassificationEvaluator(labelCol="label",
                                                                               predictionCol="prediction",
                                                                               metricName="accuracy"),
                                   trainRatio=0.8)

    # Train and evaluate models
    rf_model = tvs_rf.fit(train)
    rf_predictions = rf_model.transform(test)

    lr_model = tvs_lr.fit(train)
    lr_predictions = lr_model.transform(test)

    gbt_model = tvs_gbt.fit(train)
    gbt_predictions = gbt_model.transform(test)

    # Evaluate models
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

    rf_accuracy = evaluator.evaluate(rf_predictions)
    lr_accuracy = evaluator.evaluate(lr_predictions)
    gbt_accuracy = evaluator.evaluate(gbt_predictions)

    print("Random Forest Accuracy:", rf_accuracy)
    print("Logistic Regression Accuracy:", lr_accuracy)
    print("Gradient Boosted Trees Accuracy:", gbt_accuracy)

    # Anomaly Detection with K-Means
    kmeans = KMeans(featuresCol="features_standard", k=2)
    kmeans_pipeline = Pipeline(stages=[assembler, scaler, kmeans])
    kmeans_model = kmeans_pipeline.fit(train)
    kmeans_predictions = kmeans_model.transform(test)


    # Evaluate clustering
    def evaluate_clustering(predictions):
        cluster_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                         metricName="accuracy")
        accuracy = cluster_eval.evaluate(predictions)
        print("Clustering Accuracy:", accuracy)


    evaluate_clustering(kmeans_predictions)

    # Save models
    rf_model.write().overwrite().save("rf_model")
    lr_model.write().overwrite().save("lr_model")
    gbt_model.write().overwrite().save("gbt_model")