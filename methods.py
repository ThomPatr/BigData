from pyspark.sql import DataFrame

class Methods():

    def __init__(self):
        pass

    def load_data(self, spark, path: str):
        df = spark.read.csv(path, header=True, inferSchema=True)
        return df

    def stratified_split(self, df: DataFrame, label_col: str, weights: tuple = (0.8, 0.18, 0.02), seed: int = 42):
        # Split the DataFrame into train, test, and prediction sets based on stratified sampling, meaning that the distribution of the labels in the original DataFrame is preserved in the splits.
        labels = df.select(label_col).distinct().rdd.flatMap(lambda x: x).collect()

        train_df = []
        test_df = []
        prediction_df = []

        for label in labels:
            # Filter the DataFrame for the current label
            class_df = df.filter(df[label_col] == label)
            splits = class_df.randomSplit(weights, seed)
            train_df.append(splits[0]) # We create a list of DataFrames for each label
            test_df.append(splits[1])
            prediction_df.append(splits[2])

        train = train_df[0] # Initialize the first DataFrame (related to the first label)
        test = test_df[0]
        prediction = prediction_df[0]

        for i in range(1, len(train_df)):
            train = train.union(train_df[i]) # Union the DataFrames for each label
            test = test.union(test_df[i])
            prediction = prediction.union(prediction_df[i])
        return train, test, prediction
    

