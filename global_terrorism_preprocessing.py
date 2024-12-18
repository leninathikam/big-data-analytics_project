from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, count, sum, max, min, year, month, stddev, round
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Global Terrorism Data Preprocessing and EDA") \
    .getOrCreate()

# Load the CSV file with the correct delimiter and valid path
df = spark.read.csv("/opt/spark/global_terrorism.csv", header=True, inferSchema=True)

# Show the initial dataframe structure
df.show(5)

# Check the schema to understand data types
df.printSchema()

# Convert relevant columns to appropriate data types if necessary
df = df.withColumn("year", col("year").cast("int")) \
       .withColumn("casualties", col("casualties").cast("float")) \
       .withColumn("nkill", col("nkill").cast("float")) \
       .withColumn("nwound", col("nwound").cast("float"))

# Handle missing or invalid values
df = df.withColumn("casualties", when(col("casualties").isNull(), 0).otherwise(col("casualties"))) \
       .withColumn("nkill", when(col("nkill").isNull(), 0).otherwise(col("nkill"))) \
       .withColumn("nwound", when(col("nwound").isNull(), 0).otherwise(col("nwound")))

# Drop rows with critical missing values (e.g., country, attack type)
df = df.na.drop(subset=["country", "attacktype"])

# Summary statistics for numeric columns
df.describe().show()

# Generate insights (initial summary-based EDA)
# 1. Count of incidents by year
df.groupBy("year").agg(count("*").alias("incident_count")).orderBy("year").show()

# 2. Total casualties per year
df.groupBy("year").agg(mean("casualties").alias("average_casualties"),
                       sum("casualties").alias("total_casualties")).orderBy("year").show()

# 3. Distribution of attack types
df.groupBy("attacktype").count().orderBy("count", ascending=False).show()

# 4. Count of attacks by region
df.groupBy("region").count().orderBy("count", ascending=False).show()

# 5. Maximum casualties in a single attack
df.select(max("casualties").alias("max_casualties")).show()

# 6. Minimum casualties in a single attack
df.select(min("casualties").alias("min_casualties")).show()

# 7. Total number of attacks and average casualties per attack type
df.groupBy("attacktype").agg(count("*").alias("total_attacks"),
                             mean("casualties").alias("average_casualties")).orderBy("total_attacks", ascending=False).show()

# 8. Top 5 countries with the most attacks
df.groupBy("country").count().orderBy("count", ascending=False).limit(5).show()

# 9. Yearly trends of attacks
df.groupBy("year").agg(count("*").alias("total_attacks")).orderBy("year").show()

# Step 10: Extract month from the date column (if available)
# Count of attacks by month (assuming a date column 'date' exists)
if 'date' in df.columns:
    df = df.withColumn("month", month(col("date")))
    df.groupBy("month").count().orderBy("month").show()

df_pd = df.toPandas()

#histogram
numerical_columns = ["casualties", "nkill", "nwound"]

for col_name in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df_pd[col_name], bins=30, kde=True)
    plt.title(f"Distribution of {col_name}")
    plt.xlabel(col_name)
    plt.ylabel("Frequency")
    plt.show()

# frequency distribution of categorical predictors with bar plots
categorical_columns = ["attacktype", "region", "country"]

for col_name in categorical_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(y=df_pd[col_name], order=df_pd[col_name].value_counts().index)
    plt.title(f"Frequency Distribution of {col_name}")
    plt.xlabel("Count")
    plt.ylabel(col_name)
    plt.show()

#boxplots to check for outliers among numerical features
for col_name in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df_pd[col_name])
    plt.title(f"Boxplot of {col_name}")
    plt.xlabel(col_name)
    plt.show()

# Step 4: Generate a correlation matrix to examine relationships between numerical predictors
correlation_matrix = df_pd[numerical_columns].corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix of Numerical Features")
plt.show()

# Saving the correlation matrix to a CSV file
correlation_matrix.to_csv("/opt/spark/correlation_matrix.csv")

# Saving the preprocessed data
df.write.csv("/opt/spark/preprocessed_global_terrorism_data.csv", header=True)

# Stopping the Spark session
spark.stop()
