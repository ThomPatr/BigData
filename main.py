from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import when

# Avvia Spark
spark = (SparkSession.builder
         .appName("InSDN Optimized")
         .getOrCreate())

# Configura Spark per limitare l'output del piano di esecuzione
spark.conf.set("spark.sql.debug.maxToStringFields", 100)

# Carica il dataset
input_directory = "/home/ilaria/Scaricati/InSDN_DatasetCSV"
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
data.select("label_indexed").distinct().show()

# Crea una colonna binaria per il modello (0 e 1)
data = data.withColumn("binary_label", when(data["label_indexed"] > 0, 1).otherwise(0))

# Pre-elaborazione delle feature
features = [col for col in data.columns if col != "Label" and col != "label_indexed"]
assembler = VectorAssembler(inputCols=features, outputCol="featuresAssembled")
scaler = StandardScaler(inputCol="featuresAssembled", outputCol="features_standard")

# Dividi il dataset in training e test
train, test = data.randomSplit([0.8, 0.2], seed=42)

# Definisci i modelli
rf = RandomForestClassifier(featuresCol="features_standard", labelCol="binary_label")
gbt = GBTClassifier(featuresCol="features_standard", labelCol="binary_label")

# Crea le pipeline per entrambi i modelli
rf_pipeline = Pipeline(stages=[assembler, scaler, rf])
gbt_pipeline = Pipeline(stages=[assembler, scaler, gbt])

# Imposta la Train-Validation Split per ottimizzare i modelli
paramGrid_rf = ParamGridBuilder().addGrid(rf.numTrees, [10, 50]).build()
tvs_rf = TrainValidationSplit(
    estimator=rf_pipeline,
    estimatorParamMaps=paramGrid_rf,
    evaluator=MulticlassClassificationEvaluator(labelCol="binary_label", predictionCol="prediction", metricName="accuracy"),
    trainRatio=0.8
)

paramGrid_gbt = ParamGridBuilder().addGrid(gbt.maxDepth, [5, 10]).build()
tvs_gbt = TrainValidationSplit(
    estimator=gbt_pipeline,
    estimatorParamMaps=paramGrid_gbt,
    evaluator=MulticlassClassificationEvaluator(labelCol="binary_label", predictionCol="prediction", metricName="accuracy"),
    trainRatio=0.8
)

# Addestra i modelli
rf_model = tvs_rf.fit(train)
gbt_model = tvs_gbt.fit(train)

# Effettua predizioni
rf_predictions = rf_model.transform(test)
rf_predictions.show()
gbt_predictions = gbt_model.transform(test)
gbt_predictions.show()

# Valuta i modelli
evaluator = MulticlassClassificationEvaluator(labelCol="binary_label", predictionCol="prediction", metricName="accuracy")
rf_accuracy = evaluator.evaluate(rf_predictions)
gbt_accuracy = evaluator.evaluate(gbt_predictions)

print("Random Forest Accuracy:", rf_accuracy)
print("Gradient Boosted Trees Accuracy:", gbt_accuracy)

# Salva il dataset ridotto in formato Parquet per future analisi
# data.write.mode("overwrite").parquet("reduced_dataset.parquet")

# Salva i modelli
rf_model.write().overwrite().save("rf_model")
gbt_model.write().overwrite().save("gbt_model")

# Arresta Spark
spark.stop()
