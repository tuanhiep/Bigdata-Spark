#  Copyright (c) 2020. Tuan Hiep TRAN
# Question 1: In Spark, implement the standard k-means via MapReduce .
# You should not use the Spark MLlib clustering library for this problem.
# Question 2:
# Use a synthetic dataset S3 (s3.csv) to test the algorithm implemented in Question 1. S3 is a 2-d dataset
# which has 5000 data points and k=15 Gaussian clusters with different degree of cluster overlap. Use the
# algorithm in Question 1 with k=15 to cluster this dataset. Then draw a scatter plot using cluster IDs as labels
# (legends) to display the results.


import argparse
import numpy as np
from pyspark import Row
from pyspark.sql import SparkSession
from kmeans.my_kmeans import myKmeans, get_closest_centroid
import matplotlib.pyplot as plt

# parse the arguments from command line
parser = argparse.ArgumentParser(description="Dimensionality Reduction")
parser.add_argument("-data", "--dataSet", type=str, required=True, help="name of the data set for this program")
parser.add_argument("-k", "--numberCenters", type=int, required=True, help="number K of centers for Kmeans")
parser.add_argument("-cd", "--convergeDistance", type=float, default=6,
                    help="Threshold to stop the iterations of Kmeans")
args = parser.parse_args()
if args.convergeDistance is None or args.dataSet is None or args.numberCenters is None:
    parser.error("Kmeans requires --dataSet and --numberCenters adn --convergeDistance")



# Define spark session
spark = SparkSession \
    .builder \
    .appName("Spark Application") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
# Load and parse the data
lines = spark.sparkContext.textFile(args.dataSet)
rdd = lines.map(lambda line: np.array([float(x) for x in line.split(',')]))
K = args.numberCenters
converge_distance = args.convergeDistance
k_means_model = myKmeans(rdd, K, converge_distance)
centroids = k_means_model.get_centroids()

print("Final centroids: " + str(centroids))

predictions = rdd.map(lambda p: Row(x1= float(p[0]),x2=float(p[1]), cluster_id=float( get_closest_centroid(p, centroids))))
print(rdd.take(2))
print(predictions.take(2))
df=spark.createDataFrame(predictions).toPandas()
# Another approach
# df.select('*').limit(1000).createOrReplaceTempView("predictions")
# df=df.take(10000)
# d = spark.sql("SELECT * FROM predictions WHERE cluster_id={}").format(label)

# Scatter plot
plt.clf()
for label in range(15):
    d = df[df['cluster_id'] == label]
    plt.scatter(d['x1'], d['x2'])
plt.title('K-means')
plt.show()



