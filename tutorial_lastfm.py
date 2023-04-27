from implicit.datasets.lastfm import get_lastfm
from implicit.nearest_neighbours import bm25_weight
from implicit.als import AlternatingLeastSquares
from implicit.recommender_base import RecommenderBase
import numpy as np
import pandas as pd
import pickle

artists, users, artist_user_plays = get_lastfm()
# weight the matrix, both to reduce impact of users that have played the same artist thousands of times
# and to reduce the weight given to popular items
artist_user_plays = bm25_weight(artist_user_plays, K1=100, B=0.8)
# get the transpose since the most of the functions in implicit expect (user, item) sparse matrices instead of (item, user)
user_plays = artist_user_plays.T.tocsr()

def get_model() -> RecommenderBase:
    embedding_cache_path = "data/recommendations_lastfm.pkl"
    try:
        model = pd.read_pickle(embedding_cache_path)
    except FileNotFoundError:
        model = AlternatingLeastSquares(factors=64, regularization=0.05, alpha=2.0)
        model.fit(user_plays)
    with open(embedding_cache_path, "wb") as embedding_cache_file:
        pickle.dump(model, embedding_cache_file)
    return model

def recommend_for(userid) -> str:
    ids, scores = get_model().recommend(userid, user_plays[userid], N=10, filter_already_liked_items=False)
    return pd.DataFrame({"artist": artists[ids], "score": scores, "already_liked": np.in1d(ids, user_plays[userid].indices)}).to_html()


def get_similar(itemid) -> str:
    ids, scores = get_model().similar_items(itemid)
    return pd.DataFrame({"artist": artists[ids], "score": scores}).to_html()

# # Make recommendations for the first 1000 users in the dataset
# userids = np.arange(1000)
# ids, scores = model.recommend(userids, user_plays[userids])
# ids, ids.shape
