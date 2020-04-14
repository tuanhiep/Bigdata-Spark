#  Copyright (c) 2020. Tuan Hiep TRAN
import numpy as np

'''
Class to implement Kmeans algorithm in Spark
'''


class myKmeans:
    def __init__(self, rdd, K, converge_distance):
        self.rdd = rdd
        self.K = K
        self.converge_distance = converge_distance

    def get_centroids(self):
        temp_distance = 1.0
        k_points = self.rdd.takeSample(False, self.K, 1)
        while temp_distance > self.converge_distance:
            closest = self.rdd.map(
                lambda p: (get_closest_centroid(p, k_points), (p, 1)))
            point_stats = closest.reduceByKey(
                lambda x_f_1, x_f_2: (x_f_1[0] + x_f_2[0], x_f_1[1] + x_f_2[1]))
            new_centroids = point_stats.map(
                lambda st: (st[0], st[1][0] / st[1][1])).collect()
            temp_distance = sum(np.sum((k_points[index] - centroid) ** 2) for (index, centroid) in new_centroids)
            for (index, centroid) in new_centroids:
                k_points[index] = centroid
        self.k_points = k_points
        return k_points


# To find the index of closest centroid to the point
def get_closest_centroid(point, centers):
    index = 0
    closest_distance = float("+inf")
    for i in range(len(centers)):
        distance = np.sum((point - centers[i]) ** 2)
        if distance < closest_distance:
            closest_distance = distance
            index = i
    return index
