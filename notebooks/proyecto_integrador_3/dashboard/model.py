"""
model.py — Definición standalone de AccidentPredictionSystem.

Este módulo debe existir como archivo separado para que joblib pueda
deserializar el objeto correctamente. El archivo .joblib fue guardado
desde un notebook de Colab donde el módulo se registró como '__main__';
el CustomUnpickler en app.py redirige esa referencia a este módulo.
"""

import pandas as pd
import numpy as np
import joblib

try:
    import statsmodels.api as sm
except ImportError:
    sm = None  # predict_frequency no estará disponible sin statsmodels


class AccidentPredictionSystem:
    """
    Sistema predictivo encapsulado.
    Etapa 1: Poisson (ocurrencia)
    Etapa 2: RandomForest Calibrado (severidad)
    100% serializable con joblib.
    """

    def __init__(self, severity_pipeline, poisson_model,
                 county_encoder, iqr_bounds, feature_cols):
        self.severity_pipeline = severity_pipeline
        self.poisson_model     = poisson_model
        self.county_encoder    = county_encoder
        self.iqr_bounds        = iqr_bounds
        self.feature_cols      = feature_cols

    def _clip_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        X_c = X.copy()
        for feat, (lo, hi) in self.iqr_bounds.items():
            if feat in X_c.columns:
                X_c[feat] = X_c[feat].clip(lo, hi)
        return X_c

    def predict_severity(self, X: pd.DataFrame) -> pd.DataFrame:
        X_c   = self._clip_outliers(X[self.feature_cols])
        preds = self.severity_pipeline.predict(X_c)
        proba = self.severity_pipeline.predict_proba(X_c)
        labels = {0: 'Menor', 1: 'Intermedio', 2: 'Mayor'}
        return pd.DataFrame({
            'clase':           [labels[p] for p in preds],
            'prob_Menor':      proba[:, 0].round(4),
            'prob_Intermedio': proba[:, 1].round(4),
            'prob_Mayor':      proba[:, 2].round(4),
        })

    def predict_frequency(self, zone_df: pd.DataFrame) -> np.ndarray:
        if sm is None:
            raise ImportError("statsmodels is required for predict_frequency()")
        df = zone_df.copy()
        df['county_enc'] = self.county_encoder.transform(df[['County']])
        feats = ['temp_mean', 'humidity_mean', 'visibility_mean',
                 'rain_mean', 'Hour', 'Month', 'county_enc']
        X_p = sm.add_constant(df[feats], has_constant='add')
        return self.poisson_model.predict(X_p).values

    def save(self, path: str):
        joblib.dump(self, path)
        print(f"Sistema guardado: {path}")

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)
