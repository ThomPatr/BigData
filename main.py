from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.feature import StringIndexer

# Avvia Spark
spark = (SparkSession.builder
         .appName("InSDN Optimized")
         .getOrCreate())

# Configura Spark per limitare l'output del piano di esecuzione
spark.conf.set("spark.sql.debug.maxToStringFields", 100)

# Carica il dataset
input_directory = "/home/mariarosa/Scaricati/InSDN_DatasetCSV"
data = spark.read.csv(input_directory, header=True, inferSchema=True)
data.show()

# Seleziona solo le colonne più utili
selected_columns = [
    "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "Fwd Pkt Len Mean", "Bwd Pkt Len Mean", "Flow Byts/s", "Label"
]
data = data.select(selected_columns)

# Campiona il dataset per ridurre il carico di memoria (usa solo il 10%)
# data = data.sample(fraction=0.1, seed=42)

# Limita il numero di righe caricate per evitare blocchi
# data = data.limit(50000)

# Converte la colonna "Label" in numerico
indexer = StringIndexer(inputCol="Label", outputCol="label_indexed")
data = indexer.fit(data).transform(data)

# Mostra le etichette distinte
data.select("label_indexed", "Label").distinct().show()

# Pre-elaborazione delle feature
features = [col for col in data.columns if col != "Label" and col != "label_indexed"]
assembler = VectorAssembler(inputCols=features, outputCol="featuresAssembled")
scaler = StandardScaler(inputCol="featuresAssembled", outputCol="features_standard")

# Dividi il dataset in training e test
train, test = data.randomSplit([0.8, 0.2], seed=42)

# Definisci i modelli
rf = RandomForestClassifier(featuresCol="features_standard", labelCol="label_indexed")
dt = DecisionTreeClassifier(featuresCol="features_standard", labelCol="label_indexed")

# Crea le pipeline per entrambi i modelli
rf_pipeline = Pipeline(stages=[assembler, scaler, rf])
dt_pipeline = Pipeline(stages=[assembler, scaler, dt])

# Imposta la Train-Validation Split per ottimizzare i modelli
paramGrid_rf = ParamGridBuilder().addGrid(rf.numTrees, [10, 50]).build()
tvs_rf = TrainValidationSplit(
    estimator=rf_pipeline,
    estimatorParamMaps=paramGrid_rf,
    evaluator=MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="accuracy"),
    trainRatio=0.8
)

paramGrid_dt = ParamGridBuilder().addGrid(dt.maxDepth, [5, 10]).build()
tvs_dt = TrainValidationSplit(
    estimator=dt_pipeline,
    estimatorParamMaps=paramGrid_dt,
    evaluator=MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="accuracy"),
    trainRatio=0.8
)

# Addestra i modelli
rf_model = tvs_rf.fit(train)
dt_model = tvs_dt.fit(train)

# Effettua predizioni
rf_predictions = rf_model.transform(test)
rf_predictions.show()
dt_predictions = dt_model.transform(test)
dt_predictions.show()

# Valuta i modelli
evaluator = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="accuracy")
rf_accuracy = evaluator.evaluate(rf_predictions)
dt_accuracy = evaluator.evaluate(dt_predictions)

print("Random Forest Accuracy:", rf_accuracy)
print("Gradient Boosted Trees Accuracy:", dt_accuracy)

# Salva il dataset ridotto in formato Parquet per future analisi
# data.write.mode("overwrite").parquet("reduced_dataset.parquet")

# Salva i modelli
rf_model.write().overwrite().save("rf_model")
dt_model.write().overwrite().save("dt_model")

# Arresta Spark
spark.stop()