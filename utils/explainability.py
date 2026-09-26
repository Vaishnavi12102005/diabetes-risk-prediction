"""
utils/explainability.py - Model Explainability & SHAP Module
Part of the Healthcare Disease Risk Prediction & Clinical Decision Support System.

Provides:
- SHAP (SHapley Additive exPlanations) calculation for individual patient predictions
- Identification of clinical risk drivers (factors increasing risk vs factors reducing risk)
- Human-readable clinical explanations for medical professionals
- Global feature importance rankings
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap


class CDSSExplainer:
    """
    Explainability engine for clinical predictions using SHAP.
    """

    def __init__(self, model, background_data: pd.DataFrame, feature_names: list):
        self.model = model
        self.feature_names = feature_names
        
        # Subsample background data to 100 rows for high-speed inference
        if len(background_data) > 100:
            self.background_data = background_data.sample(100, random_state=42)
        else:
            self.background_data = background_data

        if hasattr(model, "coef_"):
            self.explainer = shap.LinearExplainer(model, self.background_data)
        elif hasattr(model, "estimators_") or "XGB" in model.__class__.__name__:
            self.explainer = shap.TreeExplainer(model, self.background_data)
        else:
            self.explainer = shap.KernelExplainer(
                model.predict_proba, self.background_data
            )

    def explain_patient(
        self, patient_raw: pd.DataFrame, patient_transformed: pd.DataFrame, top_k: int = 5
    ) -> dict:
        """
        Calculates local SHAP contributions for a single patient and separates
        features into risk-increasing and risk-reducing factors.
        """
        shap_res = self.explainer(patient_transformed)
        shap_vals = shap_res.values[0]

        factors = []
        for feat_name, s_val in zip(self.feature_names, shap_vals):
            # Clinical feature description
            direction = "INCREASE_RISK" if s_val > 0 else "REDUCE_RISK"
            formatted_name = feat_name.replace("_", " ")

            factors.append(
                {
                    "feature": feat_name,
                    "display_name": formatted_name,
                    "shap_value": float(s_val),
                    "impact_magnitude": float(abs(s_val)),
                    "direction": direction,
                }
            )

        # Sort by impact magnitude
        factors_sorted = sorted(factors, key=lambda x: x["impact_magnitude"], reverse=True)

        increasing_factors = [f for f in factors_sorted if f["direction"] == "INCREASE_RISK"][:top_k]
        reducing_factors = [f for f in factors_sorted if f["direction"] == "REDUCE_RISK"][:top_k]

        return {
            "increasing_factors": increasing_factors,
            "reducing_factors": reducing_factors,
            "all_factors": factors_sorted,
        }

    def get_global_importance(self, X_sample: pd.DataFrame) -> pd.DataFrame:
        """
        Computes mean absolute SHAP values across a cohort to rank global feature importance.
        """
        shap_values = self.explainer(X_sample).values
        mean_abs = np.abs(shap_values).mean(axis=0)
        df_imp = pd.DataFrame(
            {"Feature": self.feature_names, "Mean_Abs_SHAP": mean_abs}
        ).sort_values(by="Mean_Abs_SHAP", ascending=False)
        return df_imp
