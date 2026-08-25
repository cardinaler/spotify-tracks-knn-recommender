from preprocessor import SpotifyDataPreprocessor
import pandas as pd
import numpy as np
import os

path = os.getcwd()
path_to_data = path + '/data/spotify_data/'
path_to_save = path + '/data/processed/'
preprocessor = SpotifyDataPreprocessor()

df = pd.read_csv(path_to_data + 'dataset.csv')
X, df_meta = preprocessor.fit_transform(df)

np.save(path_to_save + 'features.npy', X)
df_meta.to_csv(path_to_save + 'meta.csv')
