"""
utils/validation.py - Clinical Data Validation & Quality Checks
Part of the Healthcare Disease Risk Prediction & Clinical Decision Support System.

Provides:
- Physiological range checking and warning flag generation
- Validation of patient inputs for single and batch predictions
"""

import pandas as pd

# Standard clinical physiological reference intervals for living adult subjects
PHYSIOLOGICAL_RANGES = {
    "Glucose": {"min": 40.0, "max": 400.0, "unit": "mg/dL"},
    "BloodPressure": {"min": 40.0, "max": 200.0, "unit": "mm Hg"},
    "SkinThickness": {"min": 5.0, "max": 80.0, "unit": "mm"},
    "Insulin": {"min": 2.0, "max": 600.0, "unit": "µU/ml"},
    "BMI": {"min": 12.0, "max": 65.0, "unit": "kg/m²"},
    "Age": {"min": 18, "max": 110, "unit": "years"},
    "Pregnancies": {"min": 0, "max": 20, "unit": "count"},
    "DiabetesPedigreeFunction": {"min": 0.05, "max": 2.5, "unit": "score"},
}


def check_physiological_warnings(patient_dict: dict) -> list:
    """
    Checks if any clinical parameters fall outside normal expected physiological bounds.
    Returns a list of advisory warning messages for healthcare providers.
    """
    warnings = []

    for param, limits in PHYSIOLOGICAL_RANGES.items():
        val = patient_dict.get(param)
        if val is None:
            # Check lowercase / snake_case alternative
            val = patient_dict.get(param.lower())

        if val is not None and val > 0:  # Ignore 0 if used as missing placeholder
            if val < limits["min"]:
                warnings.append(
                    f"Atypical low reading: {param} = {val} {limits['unit']} (typical min: {limits['min']})"
                )
            elif val > limits["max"]:
                warnings.append(
                    f"Extreme high reading: {param} = {val} {limits['unit']} (typical max: {limits['max']})"
                )

    return warnings


def validate_batch_dataframe(df: pd.DataFrame) -> tuple:
    """
    Validates uploaded CSV file for required columns.
    Returns: (is_valid, missing_columns)
    """
    required_cols = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]

    # Check case-insensitive match
    df_cols_lower = [c.lower().replace("_", "") for c in df.columns]
    missing = []
    col_mapping = {}

    for req in required_cols:
        matched = False
        for original_col in df.columns:
            if original_col.lower().replace("_", "") == req.lower():
                col_mapping[original_col] = req
                matched = True
                break
        if not matched:
            missing.append(req)

    if missing:
        return False, missing, df

    # Rename to canonical column names
    standardized_df = df.rename(columns=col_mapping)
    return True, [], standardized_df
