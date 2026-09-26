"""
backend/predictor.py - Production Inference & Prediction Engine
Part of the FastAPI backend for Healthcare Disease Risk Prediction.

Implements:
- Singleton RiskPredictor loaded once at application startup
- Transformation via InferencePreprocessor (zero training skew)
- Real-time probability estimation & risk stratification (High, Moderate, Low)
- Local Explainability via SHAP (factors increasing/reducing predicted risk)
- Batch CSV inference processing
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from backend.preprocessing import InferencePreprocessor
from utils.explainability import CDSSExplainer
from utils.validation import check_physiological_warnings


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "diabetes_data.csv")


class RiskPredictor:
    """
    Singleton inference service holding fitted model artifacts, preprocessor,
    and SHAP explainer in memory for low-latency scoring.
    """

    def __init__(self, models_dir: str = None, data_path: str = None):
        self.models_dir = os.path.abspath(models_dir) if models_dir else DEFAULT_MODELS_DIR
        self.data_path = os.path.abspath(data_path) if data_path else DEFAULT_DATA_PATH
        self.model = None
        self.model_info = {}
        self.preprocessor = None
        self.explainer = None
        self.is_ready = False
        self.load()

    def load(self):
        """Loads all artifacts once into memory."""
        model_path = os.path.join(self.models_dir, "disease_model.pkl")
        info_path = os.path.join(self.models_dir, "model_info.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")

        self.model = joblib.load(model_path)
        self.preprocessor = InferencePreprocessor(models_dir=self.models_dir)

        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                self.model_info = json.load(f)

        # Initialize background data for SHAP explainer
        if os.path.exists(self.data_path):
            df_raw = pd.read_csv(self.data_path)
            if "Outcome" in df_raw.columns:
                df_raw = df_raw.drop(columns=["Outcome"])
            bg_transformed = self.preprocessor.transform(df_raw)
            self.explainer = CDSSExplainer(
                self.model, bg_transformed, self.preprocessor.feature_names
            )
        else:
            self.explainer = None

        self.is_ready = True
        print(f"RiskPredictor successfully loaded champion model: {self.model_info.get('model_name', 'Unknown')}")

    def predict_single(self, patient_dict: dict) -> dict:
        """
        Executes end-to-end inference and SHAP explanation for a single patient record.
        """
        if not self.is_ready:
            self.load()

        # Map input to canonical feature columns
        df_patient = pd.DataFrame([
            {
                "Pregnancies": int(patient_dict.get("pregnancies", 0)),
                "Glucose": float(patient_dict.get("glucose", 0.0)),
                "BloodPressure": float(patient_dict.get("blood_pressure", 0.0)),
                "SkinThickness": float(patient_dict.get("skin_thickness", 0.0)),
                "Insulin": float(patient_dict.get("insulin", 0.0)),
                "BMI": float(patient_dict.get("bmi", 0.0)),
                "DiabetesPedigreeFunction": float(patient_dict.get("diabetes_pedigree_function", 0.47)),
                "Age": int(patient_dict.get("age", 30)),
            }
        ])

        # Check physiological plausibility
        warnings = check_physiological_warnings(df_patient.iloc[0].to_dict())

        # Transform features
        df_trans = self.preprocessor.transform(df_patient)

        # Predict probability
        prob = float(self.model.predict_proba(df_trans)[0, 1])
        risk_pct = round(prob * 100, 1)

        # Stratify clinical risk category
        # In CDSS: Prob >= 0.50 -> HIGH_RISK; 0.35 <= prob < 0.50 -> MODERATE; < 0.35 -> LOW
        if prob >= 0.50:
            prediction_str = "HIGH_RISK"
            risk_level = "HIGH"
            recommendation = (
                "Priority medical follow-up recommended: Schedule confirmatory diagnostic testing "
                "(Fasting Plasma Glucose and laboratory HbA1c). Implement structured dietary and physical activity interventions."
            )
            confidence_desc = f"Model indicates high confidence of elevated risk ({risk_pct}% estimated risk)."
        elif prob >= 0.35:
            prediction_str = "HIGH_RISK"
            risk_level = "MODERATE"
            recommendation = (
                "Borderline risk detected: Early prediabetes or metabolic syndrome risk factors present. "
                "Suggest lifestyle modification and re-screening within 6 to 12 months."
            )
            confidence_desc = f"Model indicates intermediate risk ({risk_pct}% estimated risk). Close monitoring advised."
        else:
            prediction_str = "LOW_RISK"
            risk_level = "LOW"
            recommendation = (
                "Normal physiological risk profile: Maintain standard routine preventative wellness visits "
                "and healthy lifestyle habits."
            )
            confidence_desc = f"Model indicates low disease probability ({risk_pct}% estimated risk)."

        # SHAP Explainability
        increasing_factors = []
        reducing_factors = []
        if self.explainer:
            explanation = self.explainer.explain_patient(df_patient, df_trans, top_k=4)
            increasing_factors = explanation["increasing_factors"]
            reducing_factors = explanation["reducing_factors"]

        return {
            "prediction": prediction_str,
            "probability": round(prob, 4),
            "risk_level": risk_level,
            "risk_percentage": risk_pct,
            "confidence_description": confidence_desc,
            "clinical_recommendation": recommendation,
            "warnings": warnings,
            "increasing_factors": increasing_factors,
            "reducing_factors": reducing_factors,
        }

    def predict_batch(self, df_input: pd.DataFrame) -> tuple:
        """
        Executes vectorized batch inference for healthcare staff CSV uploads.
        Returns: (augmented_df, summary_dict)
        """
        if not self.is_ready:
            self.load()

        df_work = df_input.copy()

        # Identify or synthesize patient IDs
        if "patient_id" in [c.lower() for c in df_work.columns]:
            pid_col = [c for c in df_work.columns if c.lower() == "patient_id"][0]
            patient_ids = df_work[pid_col].astype(str).tolist()
        else:
            patient_ids = [f"PT-{i+1:04d}" for i in range(len(df_work))]

        # Transform data
        df_trans = self.preprocessor.transform(df_work)
        probs = self.model.predict_proba(df_trans)[:, 1]

        predictions = []
        risk_levels = []
        for p in probs:
            if p >= 0.50:
                predictions.append("HIGH_RISK")
                risk_levels.append("HIGH")
            elif p >= 0.35:
                predictions.append("HIGH_RISK")
                risk_levels.append("MODERATE")
            else:
                predictions.append("LOW_RISK")
                risk_levels.append("LOW")

        results_df = pd.DataFrame({
            "patient_id": patient_ids,
            "prediction": predictions,
            "probability": np.round(probs, 4),
            "risk_level": risk_levels,
        })

        summary = {
            "total_records": len(results_df),
            "high_risk_count": int((results_df["prediction"] == "HIGH_RISK").sum()),
            "low_risk_count": int((results_df["prediction"] == "LOW_RISK").sum()),
            "average_risk_probability": float(np.round(np.mean(probs), 4)),
        }

        return results_df, summary


# Global singleton instance
predictor_instance = RiskPredictor()
