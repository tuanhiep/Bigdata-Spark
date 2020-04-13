#Install the dependencies (Apache Spark 3.0-Hadoop 2.7, Java 8, and FindSpark) in Colab environment:
# !rm -r spark*
# !apt-get install openjdk-8-jdk-headless -qq
# !wget -q http://mirrors.koehn.com/apache/spark/spark-3.0.0-preview2/spark-3.0.0-preview2-bin-hadoop3.2.tgz
# !tar xf spark-3.0.0-preview2-bin-hadoop3.2.tgz
# !pip install -q findspark

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.0.0-preview2-bin-hadoop3.2"

# import findspark
# findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()
spark

# import matplotlib.pyplot as plt
# import numpy as np

####Data
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

row = spark.read.csv("/content/covtype.data", inferSchema=True, header=False)
train, test = row.randomSplit([0.8, 0.2], 1234)
print("Train size: ", train.count())
print("Test size: ", test.count())

assembler = VectorAssembler(
    inputCols=row.schema.names[0:-1],
    outputCol="features")

train = assembler.transform(train)
test = assembler.transform(test)

evaluator = MulticlassClassificationEvaluator(labelCol=row.schema.names[-1], predictionCol="prediction", metricName="accuracy")

#### Logistic Regression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#training
lr = LogisticRegression(maxIter=100, featuresCol="features", labelCol=row.schema.names[-1])
lrModel = lr.fit(train)

#testing
predictions = lrModel.transform(test)
# predictions.select(row.schema.names[-1], 'rawPrediction', 'prediction', 'probability').show(1000)

accuracy = evaluator.evaluate(predictions)
print("(Logistic Regression) Testing Accuracy = %g " % accuracy)

#### Decision Tree
from pyspark.ml.classification import DecisionTreeClassifier


#training
dt = DecisionTreeClassifier(featuresCol="features", labelCol=row.schema.names[-1], maxDepth=10)
dtModel = dt.fit(train)

#testing
predictions = dtModel.transform(test)
# predictions.select(row.schema.names[-1], 'rawPrediction', 'prediction', 'probability').show(1000)

accuracy = evaluator.evaluate(predictions)
print("(Decision Tree) Testing Accuracy = %g " % accuracy)

#### Random Forest
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#training
rf = RandomForestClassifier(featuresCol="features", labelCol=row.schema.names[-1], maxDepth=10, numTrees=100)
rfModel = rf.fit(train)

#testing
predictions = rfModel.transform(test)
# predictions.select(row.schema.names[-1], 'rawPrediction', 'prediction', 'probability').show(1000)

accuracy = evaluator.evaluate(predictions)
print("(Random Forest) Testing Accuracy = %g " % accuracy)

