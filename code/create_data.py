#feteching the dataset (using the url)
from urllib.request import urlretrieve
import zipfile

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle


# urlretrieve("http://files.grouplens.org/datasets/movielens/ml-100k.zip", "movielens.zip")
# zip_ref = zipfile.ZipFile('movielens.zip', "r")
# zip_ref.extractall()

#create user_movie_rating
ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv(
    'u.data', sep='\t', names=ratings_cols, encoding='latin-1')


result = {}

for idx in ratings.index:
    row = ratings[ratings.index==idx]
    user = int(row['user_id'].values[0])
    item = int(row['movie_id'].values[0])
    rating = int(row['rating'].values[0])
    result[(user,item)] = rating


users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv(
    'u.user', sep='|', names=users_cols, encoding='latin-1')

with open('user_movie_rating.json', 'wb') as fp:
    pickle.dump(result, fp)

with open('user_movie_rating.json', 'rb') as fp:
    r = pickle.load(fp)


#creating data for features 
cols_movies = [
    'movie_id', 'title', 'release_date', "video_release_date", "imdb_url"
] + [
    "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
movies = pd.read_csv(
    'u.item', sep='|', names=cols_movies, encoding='latin-1')


data = ratings.merge(movies, on='movie_id').merge(users, on='user_id')


#create a Label Encoder

label_encoder = LabelEncoder()

data = data.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')

data = data.drop(['user_id', 'rating'], axis = 1)

result = data.to_json(orient="records")

with open('data.json', 'wb') as fp:
    pickle.dump(result, fp)

with open('dummy_data.json', 'rb') as fp:
    data = pickle.load(fp)

