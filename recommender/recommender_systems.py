

# !rm -r spark*
# !apt-get install openjdk-8-jdk-headless -qq
# !wget -q http://mirrors.koehn.com/apache/spark/spark-3.0.0-preview2/spark-3.0.0-preview2-bin-hadoop3.2.tgz
# !tar xf spark-3.0.0-preview2-bin-hadoop3.2.tgz
# !pip install -q findspark

# import os
# os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
# os.environ["SPARK_HOME"] = "/content/spark-3.0.0-preview2-bin-hadoop3.2"
#
# import findspark
# findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()
sc = spark.sparkContext

import matplotlib.pyplot as plt
import numpy as np

"""#Extracting features from the MovieLens 100k dataset

We will use the same MovieLens dataset [ml-100k.zip](http://files.grouplens.org/datasets/movielens/ml-100k.zip) that we used in the Tutorial 3.

Upload ml-100k.zip to Colab. Unzip it.
"""

# !unzip ml-100k.zip

"""Inspect the raw ratings dataset:"""

rawData = sc.textFile("ml-100k/u.data")
rawData.first()

"""Recall that this dataset consisted of the user id, movie id, rating, timestamp fields separated by a tab ("\t") character. We don't need the time when the rating was made to train our model, so let's simply extract the first three fields:"""

rawRatings = rawData.map(lambda s: s.split("\t")[0:3])
rawRatings.first()

"""We will use Spark's MLlib library to train our model. Let's take a look at what methods are available for us to use and what input is required. First, import the ALS model from MLlib:"""

from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import Rating

"""We need to provide the ALS model with an RDD that consists of Rating records. A Rating class, in turn, is just a wrapper around user id, movie id (called product here), and the actual rating arguments.
We'll create our rating dataset using the map method and transforming the array of IDs and ratings into a Rating object:
"""

ratings = rawRatings.map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
ratings.first()

"""#Training a recommendation model on the MovieLens 100k dataset

Besides Ratings, the other inputs required for our model
are as follows:
*  *rank*: This refers to the number of factors in our ALS model, that is, the number of hidden features in our low-rank approximation matrices. Generally, the greater the number of factors, the better, but this has a direct impact on memory usage, both for computation and to store models for serving, particularly for large number of users or items. A rank in the range of 10 to 200 is usually reasonable.

*  *iterations*: This refers to the number of iterations to run. ALS models will converge to a reasonably good solution after relatively few iterations. Around 10 is often a good default.

We'll use rank of 10, and 10 iterations to illustrate how to train our model:
"""

model = ALS.train(ratings, 10, 10)

"""This returns a MatrixFactorizationModel object, which contains the user and item factors in the form of an RDD of (id, factor) pairs. These are called userFeatures and productFeatures, respectively. For example:"""

model.userFeatures().collect()
model.productFeatures().collect()

"""#Using the recommendation model

Now that we have our trained model, we're ready to use it to make predictions.

##User recommendations

In this case, we would like to generate recommended items for a given user. This usually takes the form of a top-K list, that is, the K items that our model predicts will have the highest probability of the user liking them. This is done by computing the predicted score for each item and ranking the list based on this score.

In matrix factorization, because we are modeling the ratings matrix directly, the predicted score can be computed as the vector dot product between a user-factor vector and an item-factor vector. As MLlib's recommendation model is based on matrix factorization, we can use the
factor matrices computed by our model to compute predicted scores (or ratings) for a user.

The *MatrixFactorizationModel* class has a convenient *predict* method that will compute a predicted score for a given user and item combination:
"""

predictedRating = model.predict(789, 123)
print(predictedRating)

"""This model predicts a rating for user 789 and movie 123.

To generate the top-K recommended items for a user, *MatrixFactorizationModel* provides a convenience method called *recommendProducts*:
"""

userId = 789
num_items = 10
topKRecs = model.recommendProducts(userId, num_items)
print(topKRecs)

"""##Inspecting the recommendations

We can give these recommendations a sense check by taking a quick look at the titles of the movies a user has rated and the recommended movies. First, we need to load the movie data. We'll collect this data as a Map[Int, String] method mapping the movie ID to the title:
"""

movies = sc.textFile("ml-100k/u.item")
titles = movies.map(lambda line: line.split("|")[0:2]).map(lambda array: (int(array[0]), array[1])).collectAsMap()
titles[123]

"""For our user 789, we can find out what movies they have rated, take the 10 movies with the highest rating, and then check the titles. We will do this now by first using the keyBy Spark function to create an RDD of key-value pairs from our ratings RDD, where the key will be the user ID. We will then use the lookup function to return just the ratings for this key (that is, that particular user ID) to the driver:"""

moviesForUser = ratings.keyBy(lambda r: r.user).lookup(789)
print(len(moviesForUser))

"""We see that this user has rated 33 movies.

Next, we will take the 10 movies with the highest ratings by sorting the
moviesForUser collection using the rating field of the Rating object. We will then extract the movie title for the relevant product ID attached to the Rating class from our mapping of movie titles and print out the top 10 titles with their ratings:
"""

moviesForUser.sort(key=lambda r: -r.rating)
sc.parallelize(moviesForUser[0:10]).map(lambda rating: (titles[rating.product], rating.rating)).collect()

"""Now, let's take a look at the top 10 recommendations for this user and see what the titles are using the same approach as the one we used earlier (note that the recommendations are already sorted):"""

sc.parallelize(topKRecs).map(lambda rating: (titles[rating.product], rating.rating)).collect()

"""We leave it to you to decide whether these recommendations make sense.

#Evaluating the performance of recommendation models

##Mean Square Error

We will compute the MSE and RMSE metrics using RegressionMetrics.
"""

from pyspark.mllib.evaluation import RegressionMetrics

usersProducts = ratings.map(lambda r: (r.user, r.product))
predictions = model.predictAll(usersProducts).map(lambda r: ((r.user, r.product), r.rating))
ratingsAndPredictions = ratings.map(lambda r: ((r.user, r.product), r.rating)).join(predictions)
print(predictions.take(1))
print(ratingsAndPredictions.take(1))
predictedAndTrue = ratingsAndPredictions.map (lambda r: (r[1][0], r[1][1]))
regressionMetrics = RegressionMetrics(predictedAndTrue)

print("Mean Squared Error = ", regressionMetrics.meanSquaredError)
print("Root Mean Squared Error ", regressionMetrics.rootMeanSquaredError)

"""#Questions

##Question 1

Predict the rating that user 123 will give to the movie 427. Compute the Squared Error of that prediction.
"""

#Your code here
predictedRating = model.predict(123, 427)
print(predictedRating)
true_rating= ratingsAndPredictions.filter(lambda rap : rap[0][0]==123 and rap[0][1]==427)
predictedAndTrue_123_427 = true_rating.map (lambda r: (r[1][0], r[1][1]))
regressionMetrics_123_427 = RegressionMetrics(predictedAndTrue_123_427)
# Because the RDD has only 1 line so mean squared error is needed squared error
print("Squared Error = ", regressionMetrics_123_427.meanSquaredError)
print("Root Squared Error ", regressionMetrics_123_427.rootMeanSquaredError)

"""
Reference

[Machine Learning with Spark: Create scalable machine learning applications to power a modern data-driven business using Spark](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-spark) by Nick Pentreath.
"""