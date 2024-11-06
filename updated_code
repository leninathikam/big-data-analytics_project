from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.types import IntegerType, DoubleType
import matplotlib.pyplot as plt
import numpy as np

# Start Spark session
spark = SparkSession.builder.appName("TerrorismPrediction").getOrCreate()

# Load dataset (adjust path if necessary)
file_path = "/content/globalterrorismdb_0718dist.csv"
data = spark.read.csv(file_path, header=True, inferSchema=True)

# Convert necessary columns from string to numeric types and handle nulls
data = data.withColumn("nkill", col("nkill").cast(DoubleType())).fillna({"nkill": 0})
data = data.withColumn("nwound", col("nwound").cast(DoubleType())).fillna({"nwound": 0})
data = data.withColumn("property", col("property").cast(IntegerType())).fillna({"property": 0})  # Assuming binary (0 or 1)
data = data.withColumn("suicide", col("suicide").cast(IntegerType())).fillna({"suicide": 0})    # Assuming binary (0 or 1)
data = data.withColumn("success", col("success").cast(IntegerType())).fillna({"success": 0})    # Assuming binary (0 or 1)

# Define categorical columns to index
categorical_cols = ["country_txt", "region_txt", "city", "attacktype1_txt", "targtype1_txt", "weaptype1_txt"]

# Index categorical columns
indexers = [StringIndexer(inputCol=col, outputCol=col+"_indexed").fit(data) for col in categorical_cols]
for indexer in indexers:
    data = indexer.transform(data)

# Assemble feature columns into a single vector
feature_cols = ["iyear", "imonth", "iday", "country_txt_indexed", "region_txt_indexed", "city_indexed",
                "attacktype1_txt_indexed", "targtype1_txt_indexed", "weaptype1_txt_indexed", "nkill", "nwound",
                "property", "suicide", "success"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Check if columns are correctly cast and assembled, and handle nulls in feature columns
for col_name in feature_cols:
    data = data.withColumn(col_name, col(col_name).cast(DoubleType())).fillna(0)

data = assembler.transform(data)

# Define label column (modify this based on the target variable)
data = data.withColumn("label", col("success"))

# Split data into training and testing sets (70% training, 30% testing)
train_data, test_data = data.randomSplit([0.7, 0.3], seed=42)

# Initialize logistic regression model
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Initialize Neural Network (Multilayer Perceptron)
mlp = MultilayerPerceptronClassifier(featuresCol="features", labelCol="label", maxIter=100)

# Set up parameter grid for tuning Logistic Regression
param_grid_lr = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

# Set up parameter grid for tuning Neural Network (MLP)
param_grid_mlp = ParamGridBuilder() \
    .addGrid(mlp.layers, [[len(feature_cols), 5, 4, 3, 2], [len(feature_cols), 10, 5, 2]]) \
    .build()

# Cross-validator for model tuning Logistic Regression
cv_lr = CrossValidator(estimator=lr,
                       estimatorParamMaps=param_grid_lr,
                       evaluator=BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC"),
                       numFolds=5)

# Cross-validator for model tuning Neural Network
cv_mlp = CrossValidator(estimator=mlp,
                        estimatorParamMaps=param_grid_mlp,
                        evaluator=BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC"),
                        numFolds=5)

# Train models with cross-validation
cv_model_lr = cv_lr.fit(train_data)
cv_model_mlp = cv_mlp.fit(train_data)

# Make predictions on the test dataset
predictions_lr = cv_model_lr.transform(test_data)
predictions_mlp = cv_model_mlp.transform(test_data)

# Evaluate accuracy and AUC for Logistic Regression
evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

accuracy_lr = evaluator_accuracy.evaluate(predictions_lr)
accuracy_mlp = evaluator_accuracy.evaluate(predictions_mlp)

auc_lr = evaluator_auc.evaluate(predictions_lr)
auc_mlp = evaluator_auc.evaluate(predictions_mlp)

# Display evaluation metrics for both models
print(f"Logistic Regression - Test Accuracy: {accuracy_lr:.2f}")
print(f"Logistic Regression - Test AUC: {auc_lr:.2f}")
print(f"Neural Network (MLP) - Test Accuracy: {accuracy_mlp:.2f}")
print(f"Neural Network (MLP) - Test AUC: {auc_mlp:.2f}")

# Plotting accuracy and AUC comparison between the models
models = ['Logistic Regression', 'Neural Network (MLP)']
accuracy_scores = [accuracy_lr, accuracy_mlp]
auc_scores = [auc_lr, auc_mlp]

# Plot Accuracy
plt.figure(figsize=(10, 5))
plt.bar(models, accuracy_scores, color=['blue', 'green'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.show()

# Plot AUC
plt.figure(figsize=(10, 5))
plt.bar(models, auc_scores, color=['blue', 'green'])
plt.title('Model AUC Comparison')
plt.ylabel('AUC')
plt.show()

# Stop Spark session
spark.stop()
