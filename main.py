import pandas as pd
import numpy as np
import os
from src.make_dataset import make_dataset
from src.knn_recomender import SpotifyRecomender

path = os.getcwd()

path_to_data = path + '/data/spotify_data/'
path_to_save = path + '/data/processed/'

# make_dataset(path_to_data=path_to_data, path_to_save=path_to_save)

df_meta = pd.read_csv(path_to_save + 'meta.csv')
X_train = np.load(path_to_save + 'features.npy')

id = 42

print('Chosen:', df_meta.iloc[id])

query_vector = X_train[id].reshape(1, -1)

model = SpotifyRecomender(n_neighb=5, metr='cosine')
model.fit(X_train)

distances, indices = model.recomend(query_vector=query_vector, n_tracks=50)

recs = df_meta.iloc[indices[1:]].copy()
recs['distances'] = distances[1:]

print(recs[['artists', 'album_name', 'track_name',  'track_genre', 'distances']])
