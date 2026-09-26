"""
backend/preprocessing.py - Inference Preprocessing Module
Part of the FastAPI backend for Healthcare Disease Risk Prediction.

Imports transformer helpers from training.preprocessing to ensure zero train-serve skew.
"""

import os
import joblib
import pandas as pd
from training.preprocessing import (
    replace_zeros_with_nan,
    create_engineered_features,
    transform_data,
    RAW_NUMERIC_COLS,
    QUIRK_COLS,
    CATEGORICAL_COLS,
    ENGINEERED_NUMERIC_COLS,
)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


class InferencePreprocessor:
    """
    Loads fitted scikit-learn transformers from models/ and provides
    a single .preprocess() method for incoming dictionaries or DataFrames.
    """

    def __init__(self, models_dir: str = None):
        self.models_dir = os.path.abspath(models_dir) if models_dir else DEFAULT_MODELS_DIR
        self.imputer = None
        self.encoder = None
        self.scaler = None
        self.feature_names = None
        self.is_loaded = False
        self.load_artifacts()

    def load_artifacts(self):
        imputer_path = os.path.join(self.models_dir, "imputer.pkl")
        encoder_path = os.path.join(self.models_dir, "encoder.pkl")
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        feat_path = os.path.join(self.models_dir, "feature_columns.pkl")

        if (
            os.path.exists(imputer_path)
            and os.path.exists(encoder_path)
            and os.path.exists(scaler_path)
            and os.path.exists(feat_path)
        ):
            self.imputer = joblib.load(imputer_path)
            self.encoder = joblib.load(encoder_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_names = joblib.load(feat_path)
            self.is_loaded = True
        else:
            self.is_loaded = False

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_loaded:
            self.load_artifacts()
            if not self.is_loaded:
                raise FileNotFoundError(
                    f"Preprocessing artifacts not found in {self.models_dir}. Please run training first."
                )

        return transform_data(
            df, self.imputer, self.encoder, self.scaler, self.feature_names
        )
