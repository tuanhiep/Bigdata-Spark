#  Copyright (c) 2020. Tuan-Hiep TRAN
from pathlib import Path
import shutil
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer
from pyspark.ml import Pipeline

# Question 1. (Holdout)
# Create a training set that contains 80% of the data and export it to a .csv file called training.csv. Create a
# test set that contains the remaining 20% and export it to a .csv file called testing.csv.
# Prepare training and test data.

# Check if folder csv exists, then delete it
# csv_path = Path('csv')
# if csv_path.exists() and csv_path.is_dir():
#     shutil.rmtree(csv_path)

# Define spark session
spark = SparkSession \
    .builder \
    .appName("Spark Application") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
raw_data = spark.read.format("csv").option("inferSchema", "true").load("resource/covtype.data")
train, test = raw_data.randomSplit([0.8, 0.2], seed=12345)
# Create training set and testing set
train.repartition(1).write.format("com.databricks.spark.csv").save("csv/training.csv")
test.repartition(1).write.format("com.databricks.spark.csv").save("csv/testing.csv")

# Question 2. (Predicting Forest Cover)
# Train the following classifiers on training.csv using Spark MLlib:

# - Multi-nominal Logistic Regression


assembler = VectorAssembler(inputCols=train.columns[:-1], outputCol='features')
assembler.transform(train)

lr = LogisticRegression(maxIter=10,
                        regParam=0.3,
                        elasticNetParam=0.8,
                        labelCol=raw_data.columns[-1])
pipeline_lr = Pipeline(stages=[assembler, lr])
model_lr = pipeline_lr.fit(train)
trainingSummary = model_lr.stages[1].summary
training_accuracy = trainingSummary.accuracy
print("Logistic Regression gives the accuracy of training set: %f" % training_accuracy)
# Predict
predictions = model_lr.transform(test)
# Select example rows to display.
predictions.select("prediction", raw_data.columns[-1], "features").show(5)
# Select (prediction, true label) and compute the accuracy of testing set
evaluator = MulticlassClassificationEvaluator(
    labelCol=raw_data.columns[-1], predictionCol="prediction", metricName="accuracy")
testing_accuracy = evaluator.evaluate(predictions)
print("Logistic Regression gives the accuracy of testing set %f " % testing_accuracy)
predictions.select("prediction", raw_data.columns[-1]).repartition(1).write.format("com.databricks.spark.csv").save(
    "csv/predictions_logistic_regression.csv")

# - Decision Tree Classifier

dt = DecisionTreeClassifier(labelCol=raw_data.columns[-1])
pipeline_dt = Pipeline(stages=[assembler, dt])
model_dt = pipeline_dt.fit(train)
# Predict
predictions = model_dt.transform(test)
# Select example rows to display.
predictions.select("prediction", raw_data.columns[-1], "features").show(5)
# Select (prediction, true label) and compute the accuracy of testing set
evaluator = MulticlassClassificationEvaluator(
    labelCol=raw_data.columns[-1], predictionCol="prediction", metricName="accuracy")
testing_accuracy = evaluator.evaluate(predictions)
print("Decision Tree gives the accuracy of testing set %f" % testing_accuracy)
predictions.select("prediction", raw_data.columns[-1]).repartition(1).write.format("com.databricks.spark.csv").save(
    "csv/predictions_decision_tree.csv")

# # - Random Forest Classifier

rf = RandomForestClassifier(labelCol=raw_data.columns[-1], numTrees=10)
pipeline_rf = Pipeline(stages=[assembler, rf])
model_rf = pipeline_rf.fit(train)
# Predict
predictions = model_rf.transform(test)
# Select example rows to display.
predictions.select("prediction", raw_data.columns[-1], "features").show(5)
# Select (prediction, true label) and compute the accuracy of testing set
evaluator = MulticlassClassificationEvaluator(
    labelCol=raw_data.columns[-1], predictionCol="prediction", metricName="accuracy")
testing_accuracy = evaluator.evaluate(predictions)
print("Random Forest gives the accuracy of testing set %f" % testing_accuracy)
predictions.select("prediction", raw_data.columns[-1]).repartition(1).write.format("com.databricks.spark.csv").save(
    "csv/predictions_random_forest.csv")
