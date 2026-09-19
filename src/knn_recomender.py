import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class SpotifyRecomender:
    def __init__(self, n_neighb, metr = 'cosine'):
        self.model = NearestNeighbors(n_neighbors=n_neighb, metric=metr, n_jobs=-1)
        self.is_fitted = False

    def fit(self, X_train):
        self.model.fit(X_train)
        self.is_fitted = True

        return self

    def recomend(self, query_vector, n_tracks):
        if not self.is_fitted:
            print("Error: model is not fitted!")
        else:
            distances, indices = self.model.kneighbors(query_vector, n_neighbors=n_tracks)

        return distances[0], indices[0]
        

        

