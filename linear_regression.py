#Install the dependencies (Apache Spark 3.0-Hadoop 2.7, Java 8, and FindSpark) in Colab environment:
# !rm -r spark*
# !apt-get install openjdk-8-jdk-headless -qq
# !wget -q http://mirrors.koehn.com/apache/spark/spark-3.0.0-preview2/spark-3.0.0-preview2-bin-hadoop3.2.tgz
# !tar xf spark-3.0.0-preview2-bin-hadoop3.2.tgz
# !pip install -q findspark

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.0.0-preview2-bin-hadoop3.2"

import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()
spark

import matplotlib.pyplot as plt
import numpy as np

from pyspark.sql.functions import lit
import time

row = spark.read.csv("/content/kc_house_data.csv", inferSchema=True, header=True)
#add intercept column
row = row.withColumn("intercept", lit(1))
print("Correlation price, bedrooms:",row.stat.corr('price','bedrooms'))
print("Correlation price, bathrooms:",row.stat.corr('price','bathrooms'))
print("Correlation price, sqft_living:",row.stat.corr('price','sqft_living'))
print("Correlation price, sqft_lot:",row.stat.corr('price','sqft_lot'))
print("Correlation price, floors:",row.stat.corr('price','floors'))
print("Correlation price, waterfront:",row.stat.corr('price','waterfront'))
print("Correlation price, view:",row.stat.corr('price','view'))
print("Correlation price, condition:",row.stat.corr('price','condition'))
print("Correlation price, grade:",row.stat.corr('price','grade'))
print("Correlation price, sqft_above:",row.stat.corr('price','sqft_above'))
print("Correlation price, sqft_basement:",row.stat.corr('price','sqft_basement'))
print("Correlation price, yr_built:",row.stat.corr('price','yr_built'))
print("Correlation price, yr_renovated:",row.stat.corr('price','yr_renovated'))
print("Correlation price, zipcode:",row.stat.corr('price','zipcode'))
print("Correlation price, lat:",row.stat.corr('price','lat'))
print("Correlation price, long:",row.stat.corr('price','long'))
print("Correlation price, sqft_living15:",row.stat.corr('price','sqft_living15'))
print("Correlation price, sqft_lot15:",row.stat.corr('price','sqft_lot15'))

#remove features with correlations < 0.5
row = row.drop('date', 'bedrooms', 'sqft_lot','condition', 'sqft_basement','yr_built','yr_renovated','zipcode', 'lat','long','sqft_lot15')
train, test = row.randomSplit([0.8, 0.2], time.time())

X_with_y = train.drop('id')
X = X_with_y.drop('price')

#train linear regression
XTX = X.rdd.map(lambda s: (1, np.outer(np.array(s).astype(np.float), np.array(s).astype(np.float)))).reduceByKey(lambda a, b: a + b).collect()[0][1]
inverse_XTX = np.linalg.inv(np.asarray(XTX))
XTy = X_with_y.rdd.map(lambda s: (1, np.outer(np.array(s).astype(np.float)[1:],np.array(s).astype(np.float)[0]))).reduceByKey(lambda a, b: a + b).collect()[0][1]
w = np.matmul(inverse_XTX,np.asarray(XTy))

##check that XTX is correct
# dftoarray = np.array(X.select('*').collect()).astype(np.float)
# localXTX = np.matmul(np.transpose(dftoarray),dftoarray)
# localXTX

#test the trained LR model
Xtest = test.drop('price')
predicted_y = Xtest.rdd.map(lambda s: (s['id'],np.matmul(np.array(s).astype(np.float)[1:],w))).collect()
print(predicted_y)

# !rm -r output
debug = X_with_y.rdd.map(lambda s: (1, np.array(s).astype(np.float)))
debug.saveAsTextFile("output")

