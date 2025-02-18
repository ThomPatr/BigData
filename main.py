from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, MultilayerPerceptronClassifier
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
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
data.select("label_indexed", "Label").distinct().show()

# Pre-elaborazione delle feature
features = [col for col in data.columns if col != "Label" and col != "label_indexed"]
assembler = VectorAssembler(inputCols=features, outputCol="featuresAssembled") # aggrega le feature numeriche in un unico vettore
scaler = StandardScaler(inputCol="featuresAssembled", outputCol="features_standard") # normalizza i dati

# Testa il VectorAssembler
assembled_data = assembler.transform(data)
assembled_data.show(5)

# Testa il StandardScaler
scaled_data = scaler.fit(assembled_data).transform(assembled_data)
scaled_data.show(5)

# Definisce i modelli per la regressione
## RandomForestClassifier: Modello basato su struttura con più alberi decisionali indipendeti
rf = RandomForestClassifier(featuresCol="features_standard", labelCol="label_indexed")

## DecisionTreeClassifier: Modello basato su struttura ad albero decisionale
dt = DecisionTreeClassifier(featuresCol="features_standard", labelCol="label_indexed")

## Multilayer Perceptron Classifier
# Configura le dimensioni del layer della rete neurale (numero di neuroni per layer)
    # Il primo numero è il numero di feature (dimensione del vettore di input)
    # Il secondo numero è il numero di classi (output)
    # Il terzo numero è la dimensione dell'output, ossia il numero di classi target
layers = [len(features), 5, 4, 8]  # Input | Hidden Layer | Hidden Layer | Output (8 classi)
mlp = MultilayerPerceptronClassifier(featuresCol="features_standard", labelCol="label_indexed", layers=layers, blockSize=128, maxIter=100)

## LinearRegression
lr = LinearRegression(featuresCol="features_standard", labelCol="label_indexed")


# Crea le pipeline per entrambi i modelli per includere la preparazione dei dati nei modelli
rf_pipeline = Pipeline(stages=[assembler, scaler, rf])
dt_pipeline = Pipeline(stages=[assembler, scaler, dt])
lr_pipeline = Pipeline(stages=[assembler, scaler, lr])
mlp_pipeline = Pipeline(stages=[assembler, scaler, mlp])


## Validazione dei modelli utile per selezionare i modelli/paramentri ottimali
# TrainValidationSplit esegue una suddivisione dei dati in training e test utilizzando il trainRatio
# Imposta la Train-Validation Split per ottimizzare i modelli
paramGrid_rf = ParamGridBuilder().addGrid(rf.numTrees, [10, 50]).build() # costruisce una griglia di parametri
tvs_rf = TrainValidationSplit(
    estimator=rf_pipeline, # pipeline da ottimizzare
    estimatorParamMaps=paramGrid_rf,
    # evaluator {RegressionEvaluator, BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
    evaluator=MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="accuracy"), # metrica per valutare le prestazioni
    trainRatio=0.8
)

paramGrid_dt = ParamGridBuilder().addGrid(dt.maxDepth, [5, 10]).build()
tvs_dt = TrainValidationSplit(
    estimator=dt_pipeline,
    estimatorParamMaps=paramGrid_dt,
    evaluator=MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="accuracy"),
    trainRatio=0.8
)

paramGrid_mlp = ParamGridBuilder().addGrid(mlp.maxIter, [50, 100]).addGrid(mlp.blockSize, [64, 128]).build()
tvs_mlp = TrainValidationSplit(
    estimator=mlp_pipeline,
    estimatorParamMaps=paramGrid_mlp,
    evaluator=MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="accuracy"),
    trainRatio=0.8
)

paramGrid_lr = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01]).addGrid(lr.elasticNetParam, [0.5, 0.8]).build()
tvs_lr = TrainValidationSplit(
    estimator=lr_pipeline,  # pipeline da ottimizzare
    estimatorParamMaps=paramGrid_lr,
    evaluator=RegressionEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="rmse"),  # metrica per la regressione
    trainRatio=0.8
)


# Addestra i modelli

# Dividi il dataset in training e test
train, test = data.randomSplit([0.8, 0.2], seed=42)
print(f"Training set count: {train.count()}, Test set count: {test.count()}")

rf_model = tvs_rf.fit(train) # fit(), dato un DataFrame, addestra il modello per produrre un Transformer
dt_model = tvs_dt.fit(train)
mlp_model = tvs_mlp.fit(train)
lr_model = tvs_lr.fit(train)


# Effettua predizioni
print("Random Forest")
rf_predictions = rf_model.transform(test) # transform() trasforma il DataFrame in un altro aggiungendo nuove colonne, in questo caso viene aggiunta la colonna della predezione
rf_predictions.show()
print("Decision Tree")
dt_predictions = dt_model.transform(test)
dt_predictions.show()
print("Multilayer Perceptron Classifier")
mlp_predictions = mlp_model.transform(test)
mlp_predictions.show()
print("Linear Regression")
lr_predictions = lr_model.transform(test)
lr_predictions.show()


# Valuta i modelli
evaluator = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="accuracy") # valuta le prestazioni
rf_accuracy = evaluator.evaluate(rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)
dt_accuracy = evaluator.evaluate(dt_predictions)
print("Decision Tree Accuracy:", dt_accuracy)
mlp_accuracy = evaluator.evaluate(mlp_predictions)
print("Multilayer Perceptron Accuracy:", mlp_accuracy)
evaluator = RegressionEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="rmse")  # RMSE per la regressione
lr_rmse = evaluator.evaluate(lr_predictions)
print("Linear Regression RMSE:", lr_rmse)


# Salva il dataset ridotto in formato Parquet per future analisi
# data.write.mode("overwrite").parquet("reduced_dataset.parquet")

# Salva i modelli
rf_model.write().overwrite().save("rf_model")
dt_model.write().overwrite().save("dt_model")
mlp_model.write().overwrite().save("mlp_model")
lr_model.write().overwrite().save("lr_model")

# Arresta Spark
spark.stop()
