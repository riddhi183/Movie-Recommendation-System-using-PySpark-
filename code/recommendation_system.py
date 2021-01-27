from pyspark import SparkContext
from collections import Counter
import pickle
import json
import re
import numpy as np
import math


sc = SparkContext()
data_file = "data.json"
rating_file = "user_movie_rating.json"

def convert_into_vectors(movie_vector):

    movie_id = movie_vector["movie_id"]
    features = []
    
    for feature, value in item_vector.items():
        if feature == "movie_id":
            continue
        features.append((feature, value))     
    return (movie_id, features)

# normalize item vectors 
def normalize_vector(vector):
    
    movie = vector[0]
    features = vector[1]    
    
    mean = sum(map(lambda x: x[1], features)) / len(features)
    normalized_vector = list(map(lambda x: (x[0], x[1] - mean), features))
    
    return (movie, normalized_vector)

def content_based_recommendation(user_id, movie_id, rating):

    #convert the input to rdd
    rdd_input = sc.parallelize([((user_id, movie_id), rating)])

    # read user rating vectors
    with open(rating_file, 'rb') as fp:
        rating = pickle.load(fp)

    user_movie_rating = sc.parallelize(rating).filter(lambda x: x[0][0] == user_id)
    user_movie_rating = user_movie_rating.union(rdd_input)

    # create a broadcast variable for user_id
    user_id_items = sc.broadcast(set(user_movie_rating.map(lambda x: x[0][1]).collect()))

    # output: (movie, (user, rating))
    item_movie_rating = user_movie_rating.map(lambda x: (x[0][1], (x[0][0], x[1])))
    
    # reading movie vectors
    rdd = sc.textFile(data_filename).map(lambda x: json.loads(x)).map(lambda x: convert(x))

    #output: (movie, (feature, rating))
    normalized_rdd = rdd.map(lambda x: normalize_vector(x)).flatMapValues(list)
    
    # concantenate the two
    joined_rdd = normalized_rdd.join(item_user_rating)
    
    # output: (feature, (user_id, value))
    joined_rdd = joined_rdd.map(lambda x: (x[1][0][0], (x[0], x[1][1][0], x[1][0][1] * x[1][1][1])))\
                            .reduceByKey(lambda x, y: (x[0], x[1], x[2] * y[2]))\
                            .map(lambda x: (x[0], (x[1][1], x[1][2])))
    
    # output: (feature, (item, value))
    item_ratings = normalized_rdd.filter(lambda x: x[0] not in user_id_items.value)\
                                    .map(lambda x: (x[1][0], (x[0], x[1][1])))

    # output: ((user_id, item), value)
    top5rec = item_ratings.join(joined_rdd)\
                            .map(lambda x: ((x[1][1][0], x[1][0][0]), (x[1][0][1] * x[1][1][1])))\
                            .reduceByKey(lambda x,y: x + y)\
                            .top(5, lambda x: x[1])
    
    top5rec = list(map(lambda x: x[0][1], top5rec))

    
    return top5rec
    
print(content_based_recommendation(22, 2334, 3))

