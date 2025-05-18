from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, VarianceThresholdSelector
from pyspark.ml.stat import Correlation
from pyspark.sql.functions import col, count, isnan, when, abs
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config 

class Preprocessing():

    def  __init__(self):
        pass

    def assemble(self, columns):
        assembler = VectorAssembler(inputCols=columns, outputCol="features_assembled")

        return assembler


    def nullCount(self, df):
        # Count the number of null values in each column
        nullCounts = df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns])
        row = nullCounts.collect()[0]

        totalNulls = sum(row.asDict().values())
        print(f"Il numero totale di valori nulli nel dataset è: {totalNulls}")
        return totalNulls
    
    def duplicateCounts(self, df):
        # Count the number of duplicate rows
        duplicateCounts = df.groupBy(df.columns).agg(count("*").alias("count")).filter(col("count") > 1).count()
        return duplicateCounts

    def clean_data(self, df):
        
        # Analyze the data and clean it
        print(f"\nIl numero totale di record nel dataset è: {df.count()}")
        print(f"Verifichiamo che non ci siano valori mancanti all'interno del dataset.")
        totalNulls = self.nullCount(df)

        if totalNulls > 0:
            print("Procediamo con l'eliminazione dei record con valori nulli.")
            df = df.na.drop(subset=['Label'])
            totalNulls = self.nullCount(df)

        duplicateCounts = self.duplicateCounts(df)
        print(f"Il numero totale di record duplicati nel dataset è: {duplicateCounts}")

        print(f"\nIl numero totale di colonne presenti all'interno del dataset è: {len(df.columns)}. Sarà necessario rimuovere le colonne che non sono utili per l'analisi.")
        print(f"Le colonne presenti all'interno del dataset sono: {df.columns}")
        print("Analizziamo il numero di valori che appartengono a ciascuna classe della variabile target Label.")
        df.groupBy("Label").count().show()
        print("Si osserva che il DATASET è fortemente SBILANCIATO.")
        return df
    
    def apply_variance_selector(self,train_df, columns):
        # Apply variance selector to the columns
        print("\nProcediamo con la VARIANCE THRESHOLD SELECTION.")
        selector = VarianceThresholdSelector(featuresCol="features_assembled", varianceThreshold=0.5, outputCol="var_selected_features")

        model = selector.fit(train_df) 
        train_df = model.transform(train_df) # Add a new column with the selected features

        indices = model.selectedFeatures # Get the indices of the selected features
        selected_features = [columns[i] for i in indices] 

        print(f"\nIl numero di feature selezionate dopo la Variance Threshold Selection è: {len(selected_features)}.")
        print(f"\nLe feature selezionate dopo la Variance Threshold Selection sono: {selected_features}.")

        return train_df, selected_features
    
    def apply_correlation_selector(self, spark, train_df, selected_features):
        # Apply correlation selector to the feature selected by the variance threshold
        print("\nProcediamo con la CORRELATION-BASED FEATURE SELECTION.")
        target_col = ["label_indexed"]
        cols = ["var_selected_features"]

        df_corr = train_df.select(target_col + cols)
        assembler = VectorAssembler(inputCols=cols+target_col, outputCol="features")
        df_corr = assembler.transform(df_corr)

        selected_features.append("label_indexed")

        # Calculate the correlation matrix between the features and the target variable
        corr_matrix =  Correlation.corr(df_corr, column="features", method="spearman") # Spearman correlation measures monotonic relationships, whether linear or not
        corr_matrix_sns = corr_matrix.collect()[0][0].toArray() # Convert to numpy array
        corr_matrix_df = pd.DataFrame(data = corr_matrix_sns, columns=selected_features, index=selected_features) # Convert to pandas DataFrame

        plt.figure(figsize=(50, 50))
        sns.heatmap(corr_matrix_df,
                    xticklabels=corr_matrix_df.columns.values,
                    yticklabels=corr_matrix_df.columns.values,
                    cmap="coolwarm",
                    annot=True)
        plt.title("Matrice di correlazione - Metodo di Spearman", fontsize=30)
        plt.savefig(config.FIGURES_PATH + "/correlation_matrix.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        # plt.show()

        # Select features with high correlation with the target variable
        corr_matrix_table = corr_matrix.head() # head() returns a list of Row objects
        # Let's get the ist of correlations with the target variable
        # corr_matrix_table[0][i, len(selected_features)-1] returns the correlation between the i-th feature and the target variable, which is the last column of the matrix
        target_corr_list = [corr_matrix_table[0][i, len(selected_features)-1] for i in range(len(selected_features))][:-1]
        correlation_data = [(selected_features[i], float(target_corr_list[i])) for i in range(len(selected_features)-1)] # List of tuples (feature, correlation)

        # Create a DataFrame with the correlation information
        correlation_df = spark.createDataFrame(correlation_data, ["feature", "correlation"])
        correlation_df = correlation_df.withColumn("abs_correlation", abs(col("correlation"))) # Add a new column with the absolute value of the correlation
        correlation_df = correlation_df.orderBy(col("abs_correlation").desc()) # Order by absolute correlation
        correlation_df.show(truncate=False)

        # Select only the first 15 features with the highest absolute correlation
        selected_features = [row.feature for row in correlation_df.head(15)]
        print(f"\nIl numero di feature selezionate dopo la Correlation-based Feature Selection è: {len(selected_features)}.")
        print(f"\nLe feature selezionate dopo la Correlation-based Feature Selection sono: {selected_features}.")       

        return selected_features
    
    def smoteen_balancing(self, train_df, final_features):

        print("\nProcediamo con il BILANCIAMENTO del dataset utilizzando il metodo SMOTEENN.\n")
        train_pd = train_df.select("features_assembled", "label_indexed").toPandas() # Convert to pandas DataFrame, select only the assembled features and the label
        train_pd["features_array"] = train_pd["features_assembled"].apply(lambda x: np.array(x.toArray())) # Convert the features column to a numpy array

        X_train = np.vstack(train_pd["features_array"].values) # Stack the features arrays in sequence vertically
        y_train = train_pd["label_indexed"].values # Get the labels

        # --- GRAPH PRE-RESAMPLING ---
        # Bar plot of the class distribution
        plt.figure(figsize=(8, 5))
        orig_counts = pd.Series(y_train).value_counts().sort_index()
        sns.barplot(x=orig_counts.index, y=orig_counts.values, hue=orig_counts.index, palette="viridis", legend=False)
        plt.title("Distribuzione delle classi prima del bilanciamento")
        plt.xlabel("Label Indexed")
        plt.ylabel("Count")
        plt.savefig(config.FIGURES_PATH + "/before_smoteen.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

        # --- SMOTEEN BALANCING ---
        min_neighbors = min(pd.Series(y_train).value_counts().min(), 2) # Minimum number of neighbors for SMOTEEN
        smote = SMOTE(k_neighbors=min_neighbors, random_state=42)
        enn = EditedNearestNeighbours(n_neighbors=min_neighbors) 
        smote_enn = SMOTEENN(sampling_strategy="auto", smote=smote, enn=enn, random_state=42)
        X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train) # Resample the data

        print("Numero di record e di colonne prima del bilanciamento:", X_train.shape)
        print("Numero di record e di colonne dopo il bilanciamento:", X_resampled.shape)
        print(f"Il numero di record per ciascuna classe dopo il bilanciamento del training dataset è:\n{pd.Series(y_resampled).value_counts().sort_index().to_string(index=True, header=False)}")

        # Recreate the DataFrame with the resampled data
        resampled_df = pd.DataFrame(X_resampled, columns=final_features)
        resampled_df["label_indexed"] = y_resampled

        resampled_df.to_csv(config.BALANCED_TRAIN_PATH, index=False) # Save the resampled data to a CSV file
        print("Il dataset bilanciato è stato salvato in ./train_dataset/train.csv.")

        # --- GRAPH POST-RESAMPLING ---
        # Bar plot of the class distribution
        plt.figure(figsize=(8, 5))
        resampled_counts = pd.Series(y_resampled).value_counts().sort_index()
        sns.barplot(x=resampled_counts.index, y=resampled_counts.values, hue=resampled_counts.index, palette="viridis", legend=False)
        plt.title("Distribuzione delle classi dopo il bilanciamento")
        plt.xlabel("Label Indexed")
        plt.ylabel("Count")
        plt.savefig(config.FIGURES_PATH + "/after_smoteen.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    def preprocessing(self, spark, train_df):
        # Preprocess the data
        # 1. Remove unnecessary columns
        train_df = train_df.drop("Src Port")
        train_df = train_df.drop("Dst Port")
        numeric_columns = [col for col, dtype in train_df.dtypes if dtype != "string" and col != "label_indexed"]
        categorical_columns = [col for col, dtype in train_df.dtypes if dtype == "string" and col != "Label"]

        print(f"\nEliminiamo le colonne che non sono utili per la classificazione: {categorical_columns + ['Src Port', 'Dst Port']}.")
        train_df = train_df.drop(*categorical_columns)

        # 2. Modify the dataframe: we need to convert the Label column to a numerical format and assemble the features into a single vector
        initial_assembler = self.assemble(numeric_columns)
        initial_train_df = initial_assembler.transform(train_df)

        # Show the conversion of the Label column
        print("Conversione della colonna Label in formato numerico:")
        initial_train_df.select("Label", "label_indexed").distinct().show()

        # 3. Feature selection
        # 3.1 Variance Threshold Selection
        initial_train_df, selected_features = self.apply_variance_selector(initial_train_df, numeric_columns)
        # 3.2 Correlation-based Feature Selection
        final_selected_features = self.apply_correlation_selector(spark, initial_train_df, selected_features)

        new_assembler = self.assemble(final_selected_features)
        new_train_df = new_assembler.transform(train_df)

        # 4. Handle classes imbalance
        self.smoteen_balancing(new_train_df, final_selected_features)

        return final_selected_features

        
        