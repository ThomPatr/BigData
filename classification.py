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
        pca = PCA(k=5, inputCol="features_scaled", outputCol="pca_features") # Reduce the dimensionality of the features to 5 components

        rf = RandomForestClassifier(featuresCol="pca_features", labelCol="label_indexed")

        paramGrid_rf_pca = (ParamGridBuilder()
                            .addGrid(rf.numTrees, [10, 50])
                            .addGrid(rf.maxDepth, [5, 10])
                            .build())
        
        evaluator_rf_pca = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="accuracy")
        
        pipeline_rf_pca = Pipeline(stages=[assembler, scaler, pca, rf])
        cv_rf_pca = CrossValidator(estimator=pipeline_rf_pca,
                                   estimatorParamMaps=paramGrid_rf_pca,
                                   evaluator=evaluator_rf_pca,
                                   numFolds=5) # 5-fold cross-validation
        
        
        """ tvs_rf_pca = TrainValidationSplit(estimator=pipeline_rf_pca,
                                         estimatorParamMaps=paramGrid_rf_pca,
                                         evaluator=evaluator_rf_pca,
                                         trainRatio=0.8) """

        model = cv_rf_pca.fit(train)
        predictions = model.transform(test)

        accuracy = evaluator_rf_pca.evaluate(predictions)
        precision = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="weightedPrecision").evaluate(predictions)
        recall = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="weightedRecall").evaluate(predictions)
        f1 = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="f1").evaluate(predictions)

        print(f"Random Forest with PCA Accuracy: {accuracy:.2f}")
        print(f"Random Forest with PCA Precision: {precision:.2f}")
        print(f"Random Forest with PCA Recall: {recall:.2f}")
        print(f"Random Forest with PCA F1 Score: {f1:.2f}")

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
        pca = PCA(k=5, inputCol="features_scaled", outputCol="pca_features")

        dt = DecisionTreeClassifier(featuresCol="pca_features", labelCol="label_indexed")

        paramGrid_dt_pca = (ParamGridBuilder()
                            .addGrid(dt.maxDepth, [5, 10])
                            .build())
        
        evaluator_dt_pca = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="accuracy")
        pipeline_dt_pca = Pipeline(stages=[assembler, scaler, pca, dt])
        cv_dt_pca = CrossValidator(estimator=pipeline_dt_pca,
                                   estimatorParamMaps=paramGrid_dt_pca,
                                   evaluator=evaluator_dt_pca,
                                   numFolds=5)
        """ tvs_dt_pca = TrainValidationSplit(estimator=pipeline_dt_pca,
                                         estimatorParamMaps=paramGrid_dt_pca,
                                         evaluator=evaluator_dt_pca,
                                         trainRatio=0.8) """
        model = cv_dt_pca.fit(train)
        predictions = model.transform(test)

        accuracy = evaluator_dt_pca.evaluate(predictions)
        precision = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="weightedPrecision").evaluate(predictions)
        recall = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="weightedRecall").evaluate(predictions)
        f1 = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="f1").evaluate(predictions)

        print(f"Decision Tree with PCA Accuracy: {accuracy:.2f}")
        print(f"Decision Tree with PCA Precision: {precision:.2f}")
        print(f"Decision Tree with PCA Recall: {recall:.2f}")
        print(f"Decision Tree with PCA F1 Score: {f1:.2f}")

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
        pca = PCA(k=5, inputCol="features_scaled", outputCol="pca_features")

        num_classes = train.select("label_indexed").distinct().count()

        mlp = MultilayerPerceptronClassifier(
                featuresCol="pca_features",
                labelCol="label_indexed",
                layers=[5, 10, 5, num_classes],
                blockSize=128,
                maxIter=100,
                seed=42
        )

        paramGrid_mlp_pca = (ParamGridBuilder()
                             .addGrid(mlp.maxIter, [25, 50])
                             .addGrid(mlp.blockSize, [32, 64])
                             .build())
        
        evaluator_mlp_pca = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="accuracy")
        pipeline_mlp_pca = Pipeline(stages=[assembler, scaler, pca, mlp])
        cv_mlp_pca = CrossValidator(estimator=pipeline_mlp_pca,
                                   estimatorParamMaps=paramGrid_mlp_pca,
                                   evaluator=evaluator_mlp_pca,
                                   numFolds=5)
        """ tvs_mlp_pca = TrainValidationSplit(estimator=pipeline_mlp_pca,
                                         estimatorParamMaps=paramGrid_mlp_pca,
                                         evaluator=evaluator_mlp_pca,
                                         trainRatio=0.8) """
        
        model = cv_mlp_pca.fit(train)
        predictions = model.transform(test)

        accuracy = evaluator_mlp_pca.evaluate(predictions)
        precision = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="weightedPrecision").evaluate(predictions)
        recall = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="weightedRecall").evaluate(predictions)
        f1 = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="f1").evaluate(predictions)

        print(f"Multilayer Perceptron with PCA Accuracy: {accuracy:.2f}")
        print(f"Multilayer Perceptron with PCA Precision: {precision:.2f}")
        print(f"Multilayer Perceptron with PCA Recall: {recall:.2f}")
        print(f"Multilayer Perceptron with PCA F1 Score: {f1:.2f}")

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