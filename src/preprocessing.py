from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler, FunctionTransformer
import numpy as np
import pandas as pd

class preprocessing_data:

    @staticmethod
    def sin_cos_transformer(x, period):
        x = np.asarray(x)
        sin_key = np.sin(2 * np.pi * x / period)
        cos_key = np.cos(2 * np.pi * x / period)
        return np.column_stack([sin_key, cos_key])

    @staticmethod
    def get_sin_cos_names(transformer, input_features):
        names = []
        for col in input_features:
            names.extend([f"{col}_sin", f"{col}_cos"])
        return names

    @staticmethod
    def bin_time_signature(x):
        x = np.asarray(x)
        x = np.where(x == 4, 1, 0)
        return x.reshape(-1, 1)

    def __init__(self):
        
        key_transformer = FunctionTransformer(
            self.sin_cos_transformer,
            kw_args={'period':12},
            feature_names_out=self.get_sin_cos_names
        )

        time_signature_transformer = FunctionTransformer(
            self.bin_time_signature,
            feature_names_out = lambda trans, in_feat: ['time_signature']
        )

        self.ct = ColumnTransformer(
            transformers=[
                ('drop_cols', 'drop', ['Unnamed: 0', 'track_id', 'artists', 'album_name', 'track_name', 'track_genre'])
                ('std', StandardScaler(), ['popularity', 'energy', 'valence']),      
                ('quantile', QuantileTransformer(output_distribution='normal', random_state=42), ['speechiness', 'acousticness', 'instrumentalness', 'liveness']),    
                ('robust', RobustScaler(), ['duration_ms', 'danceability', 'loudness', 'tempo']),
                ('key_cyclical', key_transformer, ['key']),
                ('bin', time_signature_transformer, ['time_signature'])
            ],
            remainder='passthrough'
        )

    def fit(self, df : pd.DataFrame):
        df_clean = df.copy()
        df_clean = df_clean[(1000 * 30 < df_clean['duration_ms']) & (df_clean['duration_ms'] < 1000 * 60 * 10) ]
        df_clean = df_clean[df_clean['loudness'] >= -25]
        df_clean = df_clean[df_clean['tempo'] >= 1]

        self.ct.fit(df_clean)
        return self

    def fit_transform(self, df : pd.DataFrame):
        df_clean = df.copy()
        df_clean = df_clean[(1000 * 30 < df_clean['duration_ms']) & (df_clean['duration_ms'] < 1000 * 60 * 10) ]
        df_clean = df_clean[df_clean['loudness'] >= -25]
        df_clean = df_clean[df_clean['tempo'] >= 1]

        return self.ct.fit_transform(df_clean)

    def transform(self, df : pd.DataFrame):
        return self.ct.transform(df)

    def get_feature_names_out(self):
        return self.ct.get_feature_names_out()