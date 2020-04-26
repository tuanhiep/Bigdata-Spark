#  Copyright (c) 2020. Tuan Hiep TRAN
from math import sqrt

import matplotlib.pyplot as plt
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import Rating
from pyspark.sql import SparkSession
import pandas as pd
from pyspark import Row


# Question 1 (k-means):
# We want to cluster the movies using k-means implemented in Spark MLlib.
# First, get the movie latent factors from the utility matrix (rating matrix). The ALS model returned contains the
# latent factors in two RDDs of key-value pairs (called userFeatures and productFeatures) with the user or movie ID as
# the key and the factor as the value. Set the number of latent factors to be 50.
# Second, the movie latent factors will be used as training input for k-means. Set the number of clusters k=5.

# Function to evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x ** 2 for x in (point - center)]))


# Define spark session
spark = SparkSession \
    .builder \
    .appName("Spark Application") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
sc = spark.sparkContext
rawData = sc.textFile("../resource/ml-100k/u.data")
print(rawData.first())
rawRatings = rawData.map(lambda s: s.split("\t")[0:3])
print(rawRatings.first())
ratings = rawRatings.map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
print(ratings.first())
model = ALS.train(ratings, 50, 10)
# model.userFeatures().collect()
parsedData = model.productFeatures().map(lambda tuple: tuple[1])
print(parsedData.take(2))
# Build the model (cluster the data)
clusters = KMeans.train(parsedData, 5, maxIterations=10, initializationMode="random")

# Within Set Sum of Squared Errors
WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

# Save and load model using KmeansModel of MLlib 
# clusters.save(sc, "../target/KMeansModel")
# sameModel = KMeansModel.load(sc, "../target/KMeansModel")
# Get final result with cluster id for each movie id

cluster_ids = model.productFeatures().map(lambda tuple: (tuple[0], clusters.predict(tuple[1])))
print(cluster_ids.take(2))
df = spark.createDataFrame(cluster_ids).toPandas()
df.columns = ["movieID", "clusterID"]
df.to_csv("../resource/kmeans_movie_cluster.csv")

# Question 2 (Visualization):
# Use PCA in Spark MLlib to reduce the dimension of the movie factors to 2D and use a scatterplot to visualize
# the movies. In the scatterplot, use cluster IDs as labels (legends).

# *PCA
mat = RowMatrix(parsedData)
# Compute the top 2 principal components.
# Principal components are stored in a local dense matrix.
pc = mat.computePrincipalComponents(2)

# Project the rows to the linear space spanned by the top 2 principal components.
projected = mat.multiply(pc)
X = spark.createDataFrame(projected.rows.map(lambda p: Row(x1=float(p[0]), x2=float(p[1])))).toPandas()
Y = spark.createDataFrame(cluster_ids.map(lambda r: Row(clusterID=int(r[1])))).toPandas()
df = pd.concat([X, Y], axis=1)
# * Scatter plot
plt.clf()
legends = []
for label in range(5):
    d = df[df['clusterID'] == label]
    plt.scatter(d['x1'], d['x2'])
    legends.append(label)
plt.legend(legends, loc='upper left')
plt.title('K-means')
plt.show()

# Question 3 (Interpreting the movie clusters):
# We want to examine the clustering of movies we have found to see whether there is some meaningful
# interpretation of each cluster, such as a common genre or theme among the movies in the cluster.
# For each movie, get its genre labels from the u.genre file in the dataset. For each cluster, rank the movies
# belonging to that cluster by their Euclidean distances to the cluster center.
# For each cluster, extract the top 20 movies and print out their titles, genre labels, and ranking scores. Based
# on the genre labels, what do think about the quality of the clusters?
