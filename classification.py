from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, MultilayerPerceptronClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import config

class Classification():

    def __init__(self):
        pass

    def rf_classification(self, train, test, final_features):
        # Random Forest Classification

        print("\n1. RANDOM FOREST CLASSIFICATION")
        assembler = VectorAssembler(inputCols=final_features, outputCol="features_assembled")
        scaler = StandardScaler(inputCol="features_assembled", outputCol="features_scaled")

        if config.pca_bool:
            print("Utilizziamo la PCA.")
            pca = PCA(k=5, inputCol="features_scaled", outputCol="pca_features") # Reduce the dimensionality of the features to 5 components
            rf = RandomForestClassifier(featuresCol="pca_features", labelCol="label_indexed")
            pipeline_rf = Pipeline(stages=[assembler, scaler, pca, rf])
        else:
            print("Non utilizziamo la PCA.")
            rf = RandomForestClassifier(featuresCol="features_scaled", labelCol="label_indexed")
            pipeline_rf = Pipeline(stages=[assembler, scaler, rf])

        paramGrid_rf = (ParamGridBuilder()
                        .addGrid(rf.numTrees, [10, 50])
                        .addGrid(rf.maxDepth, [5, 10])
                        .build())
        
        evaluator_rf = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="accuracy")
        
        if config.cross_validation_bool:
            print("Utilizziamo la cross-validation.")
            validator = CrossValidator(estimator=pipeline_rf,
                                    estimatorParamMaps=paramGrid_rf,
                                    evaluator=evaluator_rf,
                                    numFolds=5) # 5-fold cross-validation
        else:
            print("Utilizziamo la train-validation split.")
            validator = TrainValidationSplit(estimator=pipeline_rf,
                                            estimatorParamMaps=paramGrid_rf,
                                            evaluator=evaluator_rf,
                                            trainRatio=0.8)

        model = validator.fit(train)
        predictions = model.transform(test)

        accuracy = evaluator_rf.evaluate(predictions)
        precision = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="weightedPrecision").evaluate(predictions)
        recall = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="weightedRecall").evaluate(predictions)
        f1 = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="f1").evaluate(predictions)

        print(f"Random Forest Accuracy: {accuracy:.2f}")
        print(f"Random Forest Precision: {precision:.2f}")
        print(f"Random Forest Recall: {recall:.2f}")
        print(f"Random Forest F1 Score: {f1:.2f}")

        # Let's represent the confusion matrix
        predictions_pd = predictions.toPandas()
        conf_matrix = confusion_matrix(predictions_pd["label_indexed"], predictions_pd["prediction"])
        labels = sorted(predictions_pd["label_indexed"].unique()) # Get the unique labels

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Random Forest - Confusion Matrix")
        plt.tight_layout()
        plt.savefig(config.FIGURES_PATH + "/random_forest_confusion_matrix.png", dpi=300)
        plt.close()

        return model, accuracy, precision, recall, f1

    def dt_classification(self, train, test, final_features):
        # Decision Tree Classification

        print("\n2. DECISION TREE CLASSIFICATION")
        assembler = VectorAssembler(inputCols=final_features, outputCol="features_assembled")
        scaler = StandardScaler(inputCol="features_assembled", outputCol="features_scaled")

        if config.pca_bool:
            pca = PCA(k=5, inputCol="features_scaled", outputCol="pca_features")
            dt = DecisionTreeClassifier(featuresCol="pca_features", labelCol="label_indexed")
            pipeline_dt = Pipeline(stages=[assembler, scaler, pca, dt])
        else:
            dt = DecisionTreeClassifier(featuresCol="features_scaled", labelCol="label_indexed")
            pipeline_dt = Pipeline(stages=[assembler, scaler, dt])

        paramGrid_dt = (ParamGridBuilder()
                        .addGrid(dt.maxDepth, [5, 10])
                        .build())
        
        evaluator_dt = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="accuracy")

        if config.cross_validation_bool:
            validator = CrossValidator(estimator=pipeline_dt,
                                    estimatorParamMaps=paramGrid_dt,
                                    evaluator=evaluator_dt,
                                    numFolds=5)
        else:
            validator = TrainValidationSplit(estimator=pipeline_dt,
                                            estimatorParamMaps=paramGrid_dt,
                                            evaluator=evaluator_dt,
                                            trainRatio=0.8)
        model = validator.fit(train)
        predictions = model.transform(test)

        accuracy = evaluator_dt.evaluate(predictions)
        precision = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="weightedPrecision").evaluate(predictions)
        recall = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="weightedRecall").evaluate(predictions)
        f1 = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="f1").evaluate(predictions)

        print(f"Decision Tree Accuracy: {accuracy:.2f}")
        print(f"Decision Tree Precision: {precision:.2f}")
        print(f"Decision Tree Recall: {recall:.2f}")
        print(f"Decision Tree F1 Score: {f1:.2f}")

        # Let's represent the confusion matrix
        predictions_pd = predictions.toPandas()
        conf_matrix = confusion_matrix(predictions_pd["label_indexed"], predictions_pd["prediction"])
        labels = sorted(predictions_pd["label_indexed"].unique())

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Decision Tree - Confusion Matrix")
        plt.tight_layout()
        plt.savefig(config.FIGURES_PATH + "/decision_tree_confusion_matrix.png", dpi=300)
        plt.close()

        return model, accuracy, precision, recall, f1
    
    def mlp_classification(self, train, test, final_features):
        # Multilayer Perceptron Classification

        print("\n3. MULTILAYER PERCEPTRON CLASSIFICATION")
        assembler = VectorAssembler(inputCols=final_features, outputCol="features_assembled")
        scaler = StandardScaler(inputCol="features_assembled", outputCol="features_scaled")

        num_classes = train.select("label_indexed").distinct().count()
        if config.pca_bool:
            pca = PCA(k=5, inputCol="features_scaled", outputCol="pca_features")
            mlp = MultilayerPerceptronClassifier(
                    featuresCol="pca_features",
                    labelCol="label_indexed",
                    layers=[5, 10, 5, num_classes],
                    blockSize=128,
                    maxIter=100,
                    seed=42
            )
            pipeline_mlp = Pipeline(stages=[assembler, scaler, pca, mlp])
        else:
            mlp = MultilayerPerceptronClassifier(
                    featuresCol="features_scaled",
                    labelCol="label_indexed",
                    layers=[len(final_features), 10, 5, num_classes],
                    blockSize=128,
                    maxIter=100,
                    seed=42
            )
            pipeline_mlp = Pipeline(stages=[assembler, scaler, mlp])

        paramGrid_mlp = (ParamGridBuilder()
                             .addGrid(mlp.maxIter, [25, 50])
                             .addGrid(mlp.blockSize, [32, 64])
                             .build())
        
        evaluator_mlp = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="accuracy")

        if config.cross_validation_bool:
            validator = CrossValidator(estimator=pipeline_mlp,
                                    estimatorParamMaps=paramGrid_mlp,
                                    evaluator=evaluator_mlp,
                                    numFolds=5)
        else:
            validator = TrainValidationSplit(estimator=pipeline_mlp,
                                            estimatorParamMaps=paramGrid_mlp,
                                            evaluator=evaluator_mlp,
                                            trainRatio=0.8)
        
        model = validator.fit(train)
        predictions = model.transform(test)

        accuracy = evaluator_mlp.evaluate(predictions)
        precision = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="weightedPrecision").evaluate(predictions)
        recall = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="weightedRecall").evaluate(predictions)
        f1 = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="f1").evaluate(predictions)

        print(f"Multilayer Perceptron Accuracy: {accuracy:.2f}")
        print(f"Multilayer Perceptron Precision: {precision:.2f}")
        print(f"Multilayer Perceptron Recall: {recall:.2f}")
        print(f"Multilayer Perceptron F1 Score: {f1:.2f}")

        # Let's represent the confusion matrix
        predictions_pd = predictions.toPandas()
        conf_matrix = confusion_matrix(predictions_pd["label_indexed"], predictions_pd["prediction"])
        labels = sorted(predictions_pd["label_indexed"].unique())

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Multilayer Perceptron - Confusion Matrix")
        plt.tight_layout()
        plt.savefig(config.FIGURES_PATH + "/mlp_confusion_matrix.png", dpi=300)
        plt.close()

        return model, accuracy, precision, recall, f1
    
    def best_model(self, train, test, final_features):

        results = {}

        # Random Forest
        rf_model, rf_accuracy, rf_precision, rf_recall, rf_f1 = self.rf_classification(train, test, final_features)
        results["Random Forest"] = {
            "Model": rf_model,
            "Accuracy": rf_accuracy,
            "Precision": rf_precision,
            "Recall": rf_recall,
            "F1 Score": rf_f1
        }

        # Decision Tree
        dt_model, dt_accuracy, dt_precision, dt_recall, dt_f1 = self.dt_classification(train, test, final_features)
        results["Decision Tree"] = {
            "Model": dt_model,
            "Accuracy": dt_accuracy,
            "Precision": dt_precision,
            "Recall": dt_recall,
            "F1 Score": dt_f1
        }

        # Multilayer Perceptron
        mlp_model, mlp_accuracy, mlp_precision, mlp_recall, mlp_f1 = self.mlp_classification(train, test, final_features)
        results["Multilayer Perceptron"] = {
            "Model": mlp_model,
            "Accuracy": mlp_accuracy,
            "Precision": mlp_precision,
            "Recall": mlp_recall,
            "F1 Score": mlp_f1
        }

        # Find the best model based on accuracy
        best_model_name = max(results, key=lambda x: results[x]["Accuracy"])
        print(f"\nIl modello migliore è: {best_model_name}.")

        model = results[best_model_name]["Model"]
        model.write().overwrite().save(config.MODEL_PATH)