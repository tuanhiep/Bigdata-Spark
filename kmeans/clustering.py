#  Copyright (c) 2020. Tuan Hiep TRAN
from math import sqrt
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import Rating
from pyspark.sql import SparkSession

# Define spark session
spark = SparkSession \
    .builder \
    .appName("Spark Application") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
sc= spark.sparkContext
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


# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x ** 2 for x in (point - center)]))


WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

# Save and load model
clusters.save(sc, "../target/KMeansModel")
sameModel = KMeansModel.load(sc, "../target/KMeansModel")