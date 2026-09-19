from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler, FunctionTransformer
import numpy as np
import pandas as pd


def sin_cos_transformer(x, period) -> np.ndarray:
    x = np.asarray(x)
    sin_key = np.sin(2 * np.pi * x / period)
    cos_key = np.cos(2 * np.pi * x / period)
    return np.column_stack([sin_key, cos_key])


def get_sin_cos_names(transformer, input_features):
    names = []
    for col in input_features:
        names.extend([f"{col}_sin", f"{col}_cos"])
    return names


def bin_time_signature(x) -> np.ndarray:
    x = np.asarray(x)
    x = np.where(x == 4, 1, 0)
    return x.reshape(-1, 1)


class SpotifyDataPreprocessor:
    META_COLS = ['track_id', 'artists',
                 'album_name', 'track_name', 'track_genre']

    def __init__(self, random_state=42):
        self.random_state = random_state

        key_transformer = FunctionTransformer(
            sin_cos_transformer,
            kw_args={'period': 12},
            feature_names_out=get_sin_cos_names
        )

        time_signature_transformer = FunctionTransformer(
            bin_time_signature,
            feature_names_out=lambda trans, in_feat: ['time_signature']
        )

        self.ct = ColumnTransformer(
            transformers=[
                ('drop_cols', 'drop', [
                    'Unnamed: 0', 'track_id', 'artists', 'album_name', 'track_name', 'track_genre']),
                ('std', StandardScaler(), ['popularity', 'energy', 'valence']),
                ('quantile', QuantileTransformer(output_distribution='normal', random_state=self.random_state), [
                 'speechiness', 'acousticness', 'instrumentalness', 'liveness']),
                ('robust', RobustScaler(), [
                 'duration_ms', 'danceability', 'loudness', 'tempo']),
                ('key_cyclical', key_transformer, ['key']),
                ('bin', time_signature_transformer, ['time_signature']),
                ('bool_to_int', FunctionTransformer(np.int64), ['explicit'])

            ],
            remainder='passthrough'
        )

    def clean_data(self, df: pd.DataFrame):
        df_clean = df.copy()
    
        df_clean['artist_norm'] = df_clean['artists'].astype(str).str.strip().str.lower()
        df_clean['track_norm'] = df_clean['track_name'].astype(str).str.strip().str.lower()

        df_clean = df_clean.sort_values(by='popularity', ascending=False)

        df_clean = df_clean.drop_duplicates(
            subset=['artist_norm', 'track_norm'], 
            keep='first'
        )

        df_clean = df_clean.drop(columns=['artist_norm', 'track_norm'])

        mask = (
            (df_clean['duration_ms'] > 30 * 1000) &
            (df_clean['duration_ms'] < 10 * 60 * 1000) &
            (df_clean['loudness'] >= -25) &
            (df_clean['tempo'] >= 1) &
            (df_clean['time_signature'] != 0)
        )
        df_clean = df_clean[mask]

        return df_clean

    def fit_transform(self, df: pd.DataFrame):
        df_clean = self.clean_data(df)
        df_meta = df_clean[self.META_COLS]
        X_scaled = self.ct.fit_transform(df_clean)

        return X_scaled, df_meta

    def trasform(self, df: pd.DataFrame):
        df_clean = self.clean_data(df)
        df_meta = df_clean[self.META_COLS]
        X_scaled = self.ct.fit_transform(df_clean)
        
        return X_scaled, df_meta

    def get_feature_names_out(self):
        return self.ct.get_feature_names_out()
