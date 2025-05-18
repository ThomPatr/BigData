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
import time
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

import config

class Classification():

    def __init__(self):
        pass

    def plot_roc_curve(self, preds_pd, model_name):
        y_true = preds_pd["label_indexed"].values.astype(int)
        y_score = np.vstack(preds_pd["probability"].apply(lambda x: x.toArray()).values)

        classes = np.unique(y_true)
        y_bin = label_binarize(y_true, classes=classes)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        # For each class
        for i in range(len(classes)):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(10, 7))

        for i in range(len(classes)):
            plt.plot(fpr[i], tpr[i],
                    label=f"Class {classes[i]} (AUC = {roc_auc[i]:.6f})")

        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} Multi-class ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()



    def rf_classification(self, train, test, final_features):
        # Random Forest Classification

        print("\n1. RANDOM FOREST CLASSIFICATION")
        assembler = VectorAssembler(inputCols=final_features, outputCol="features_assembled")
        scaler = StandardScaler(inputCol="features_assembled", outputCol="features_scaled")

        if config.pca_bool:
            # print("Utilizziamo la PCA.")
            pca = PCA(k=5, inputCol="features_scaled", outputCol="pca_features") # Reduce the dimensionality of the features to 5 components
            rf = RandomForestClassifier(featuresCol="pca_features", labelCol="label_indexed")
            pipeline_rf = Pipeline(stages=[assembler, scaler, pca, rf])
        else:
            # print("Non utilizziamo la PCA.")
            rf = RandomForestClassifier(featuresCol="features_scaled", labelCol="label_indexed")
            pipeline_rf = Pipeline(stages=[assembler, scaler, rf])

        paramGrid_rf = (ParamGridBuilder()
                        .addGrid(rf.numTrees, [10, 50])
                        .addGrid(rf.maxDepth, [5, 10])
                        .build())
        
        evaluator_rf = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="accuracy")
        
        if config.cross_validation_bool:
            # print("Utilizziamo la cross-validation.")
            validator = CrossValidator(estimator=pipeline_rf,
                                    estimatorParamMaps=paramGrid_rf,
                                    evaluator=evaluator_rf,
                                    numFolds=5) # 5-fold cross-validation
        else:
            # print("Utilizziamo la train-validation split.")
            validator = TrainValidationSplit(estimator=pipeline_rf,
                                            estimatorParamMaps=paramGrid_rf,
                                            evaluator=evaluator_rf,
                                            trainRatio=0.8)
        start_time = time.time()
        model = validator.fit(train)
        train_time = time.time() - start_time
        print(f"Tempo di addestramento del modello: {train_time:.2f} secondi")
        predictions = model.transform(test)

        accuracy = evaluator_rf.evaluate(predictions)
        precision = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="weightedPrecision").evaluate(predictions)
        recall = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="weightedRecall").evaluate(predictions)
        f1 = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="f1").evaluate(predictions)

        print(f"Random Forest Accuracy: {accuracy:.6f}")
        print(f"Random Forest Precision: {precision:.6f}")
        print(f"Random Forest Recall: {recall:.6f}")
        print(f"Random Forest F1 Score: {f1:.6f}")

        # Let's represent the confusion matrix
        predictions_pd = predictions.toPandas()
        self.plot_roc_curve(predictions_pd, "Random Forest")
        plt.savefig(config.FIGURES_PATH + "/rf_roc_curve.png", dpi=300)
        plt.show()
        plt.close()

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
        plt.show()
        plt.close()

        return model, accuracy, precision, recall, f1, train_time

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
        
        start_time = time.time()
        model = validator.fit(train)
        train_time = time.time() - start_time
        print(f"Tempo di addestramento del modello: {train_time:.2f} secondi")
        predictions = model.transform(test)

        accuracy = evaluator_dt.evaluate(predictions)
        precision = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="weightedPrecision").evaluate(predictions)
        recall = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="weightedRecall").evaluate(predictions)
        f1 = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="f1").evaluate(predictions)

        print(f"Decision Tree Accuracy: {accuracy:.6f}")
        print(f"Decision Tree Precision: {precision:.6f}")
        print(f"Decision Tree Recall: {recall:.6f}")
        print(f"Decision Tree F1 Score: {f1:.6f}")

        # Let's represent the confusion matrix
        predictions_pd = predictions.toPandas()
        self.plot_roc_curve(predictions_pd, "Decision Tree")
        plt.savefig(config.FIGURES_PATH + "/dt_roc_curve.png", dpi=300)
        plt.show()
        plt.close()

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
        plt.show()
        plt.close()

        return model, accuracy, precision, recall, f1, train_time
    
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
        
        start_time = time.time()
        model = validator.fit(train)
        train_time = time.time() - start_time
        print(f"Tempo di addestramento del modello: {train_time:.2f} secondi")
        predictions = model.transform(test)

        accuracy = evaluator_mlp.evaluate(predictions)
        precision = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="weightedPrecision").evaluate(predictions)
        recall = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="weightedRecall").evaluate(predictions)
        f1 = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="f1").evaluate(predictions)

        print(f"Multilayer Perceptron Accuracy: {accuracy:.6f}")
        print(f"Multilayer Perceptron Precision: {precision:.6f}")
        print(f"Multilayer Perceptron Recall: {recall:.6f}")
        print(f"Multilayer Perceptron F1 Score: {f1:.6f}")

        # Let's represent the confusion matrix
        predictions_pd = predictions.toPandas()
        self.plot_roc_curve(predictions_pd, "Multilayer Perceptron")
        plt.savefig(config.FIGURES_PATH + "/mlp_roc_curve.png", dpi=300)
        plt.show()
        plt.close()

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
        plt.show()
        plt.close()

        return model, accuracy, precision, recall, f1, train_time
    
    def best_model(self, train, test, final_features):

        results = {}

        # Random Forest
        rf_model, rf_accuracy, rf_precision, rf_recall, rf_f1, rf_train_time = self.rf_classification(train, test, final_features)
        results["Random Forest"] = {
            "Model": rf_model,
            "Accuracy": rf_accuracy,
            "Precision": rf_precision,
            "Recall": rf_recall,
            "F1 Score": rf_f1,
            "Train Time": rf_train_time
        }

        # Decision Tree
        dt_model, dt_accuracy, dt_precision, dt_recall, dt_f1, dt_train_time = self.dt_classification(train, test, final_features)
        results["Decision Tree"] = {
            "Model": dt_model,
            "Accuracy": dt_accuracy,
            "Precision": dt_precision,
            "Recall": dt_recall,
            "F1 Score": dt_f1,
            "Train Time": dt_train_time
        }

        # Multilayer Perceptron
        mlp_model, mlp_accuracy, mlp_precision, mlp_recall, mlp_f1, mlp_train_time = self.mlp_classification(train, test, final_features)
        results["Multilayer Perceptron"] = {
            "Model": mlp_model,
            "Accuracy": mlp_accuracy,
            "Precision": mlp_precision,
            "Recall": mlp_recall,
            "F1 Score": mlp_f1,
            "Train Time": mlp_train_time
        }

        # Find the best model based on accuracy
        best_model_name = max(results, key=lambda x: results[x]["Accuracy"])
        print(f"\nIl modello migliore è: {best_model_name}.")

        model = results[best_model_name]["Model"]
        model.write().overwrite().save(config.MODEL_PATH)

        model_names = list(results.keys())
        training_times = [results[name]["Train Time"] for name in model_names]
        sorted_data = sorted(zip(training_times, model_names))
        training_times_sorted, model_names_sorted = zip(*sorted_data)

        plt.figure(figsize=(10, 6))
        bars = plt.barh(model_names_sorted, training_times_sorted, color='cornflowerblue')
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                    f'{width:.2f}s', va='center', fontsize=9)

        plt.xlabel("Training Time (seconds)")
        plt.ylabel("Model")
        plt.title("Comparison of Model Training Times")
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(config.FIGURES_PATH + "/time_comparison.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()