"""
preprocessing.py - Data Preprocessing & Feature Engineering Module
Part of the Healthcare Disease Risk Prediction & Clinical Decision Support System.

Provides:
- 0-value replacement (masking biologically impossible zeros as NaN)
- Missing value imputation (median-based, robust against skewed distributions)
- Feature engineering:
    * BMI Category (Underweight, Normal, Overweight, Obese)
    * Age Group (Young, Middle_Age, Senior)
    * Risk Factor Score (composite clinical risk flag count)
- One-Hot Encoding for categorical features
- Feature scaling (RobustScaler or StandardScaler)
- Stratified train-test splitting
- SMOTE class imbalance handling (applied strictly to training data only)
- Standalone transformer methods and artifact serialization
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Clinical features where 0 represents missing/unrecorded measurement
QUIRK_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

RAW_NUMERIC_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

CATEGORICAL_COLS = ["BMI_Category", "Age_Group"]
ENGINEERED_NUMERIC_COLS = ["Risk_Factor_Score"]


def replace_zeros_with_nan(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Replaces 0 with np.nan for physiological features where 0 is biologically invalid.
    """
    df_out = df.copy()
    target_cols = columns if columns is not None else QUIRK_COLS
    for col in target_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].replace(0, np.nan)
    return df_out


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives clinical features from raw/imputed measurements:
    1. BMI_Category: Underweight (<18.5), Normal (18.5-24.9), Overweight (25-29.9), Obese (>=30)
    2. Age_Group: Young (<30), Middle_Age (30-50), Senior (>50)
    3. Risk_Factor_Score: Count of positive ADA clinical risk criteria:
       - High Glucose (>= 140 mg/dL)
       - Obese BMI (>= 30.0 kg/m^2)
       - Hypertension (Diastolic BP >= 80 mm Hg)
       - Elevated Age (>= 45 years)
       - Elevated Genetic Risk (DPF >= 0.5)
    """
    df_out = df.copy()

    # BMI Category
    df_out["BMI_Category"] = pd.cut(
        df_out["BMI"],
        bins=[-np.inf, 18.5, 24.9, 29.9, np.inf],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    ).astype(str)

    # Age Group
    df_out["Age_Group"] = pd.cut(
        df_out["Age"],
        bins=[-np.inf, 29, 50, np.inf],
        labels=["Young", "Middle_Age", "Senior"],
    ).astype(str)

    # Risk Factor Count
    high_glucose = (df_out["Glucose"] >= 140).astype(int)
    high_bmi = (df_out["BMI"] >= 30.0).astype(int)
    high_bp = (df_out["BloodPressure"] >= 80).astype(int)
    high_age = (df_out["Age"] >= 45).astype(int)
    high_dpf = (df_out["DiabetesPedigreeFunction"] >= 0.5).astype(int)

    df_out["Risk_Factor_Score"] = (
        high_glucose + high_bmi + high_bp + high_age + high_dpf
    )

    return df_out


def fit_preprocessor(X_train: pd.DataFrame, scaler_type: str = "robust"):
    """
    Fits imputer, encoder, and scaler strictly on training data.
    Returns:
        imputer, encoder, scaler, feature_names
    """
    X_train_df = X_train.copy()

    # 1. Mask zeros
    X_masked = replace_zeros_with_nan(X_train_df, QUIRK_COLS)

    # 2. Fit median imputer
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_masked[RAW_NUMERIC_COLS])
    X_imputed = pd.DataFrame(
        imputer.transform(X_masked[RAW_NUMERIC_COLS]),
        columns=RAW_NUMERIC_COLS,
        index=X_train_df.index,
    )

    # 3. Derive engineered features
    X_eng = create_engineered_features(X_imputed)

    # 4. Fit OneHotEncoder on categorical columns
    encoder = OneHotEncoder(
        categories=[
            ["Underweight", "Normal", "Overweight", "Obese"],
            ["Young", "Middle_Age", "Senior"],
        ],
        sparse_output=False,
        handle_unknown="ignore",
    )
    encoder.fit(X_eng[CATEGORICAL_COLS])
    encoded_cat_names = encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()

    # 5. Fit Scaler
    num_cols_to_scale = RAW_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS
    if scaler_type == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    scaler.fit(X_eng[num_cols_to_scale])
    feature_names = num_cols_to_scale + encoded_cat_names

    return imputer, encoder, scaler, feature_names


def transform_data(
    X: pd.DataFrame,
    imputer: SimpleImputer,
    encoder: OneHotEncoder,
    scaler,
    feature_names: list,
) -> pd.DataFrame:
    """
    Applies fitted imputer, encoder, and scaler to any incoming patient DataFrame.
    Guarantees output columns match feature_names exactly.
    """
    X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=RAW_NUMERIC_COLS)

    # 1. Mask zeros
    X_masked = replace_zeros_with_nan(X_df, QUIRK_COLS)

    # 2. Impute missing values with training medians
    X_imputed = pd.DataFrame(
        imputer.transform(X_masked[RAW_NUMERIC_COLS]),
        columns=RAW_NUMERIC_COLS,
        index=X_df.index,
    )

    # 3. Derive engineered features
    X_eng = create_engineered_features(X_imputed)

    # 4. Transform categorical columns
    X_cat = encoder.transform(X_eng[CATEGORICAL_COLS])
    encoded_cat_names = encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()
    df_cat = pd.DataFrame(X_cat, columns=encoded_cat_names, index=X_df.index)

    # 5. Scale numerical columns
    num_cols_to_scale = RAW_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS
    X_num = scaler.transform(X_eng[num_cols_to_scale])
    df_num = pd.DataFrame(X_num, columns=num_cols_to_scale, index=X_df.index)

    # Combine into final processed DataFrame
    df_processed = pd.concat([df_num, df_cat], axis=1)
    return df_processed[feature_names]


def prepare_data(
    data_path: str = "data/diabetes_data.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    apply_smote: bool = True,
    scaler_type: str = "robust",
):
    """
    End-to-end dataset preparation pipeline:
    - Loads raw data
    - Removes duplicate records if any
    - Performs stratified train/test split
    - Fits preprocessing transformers strictly on train set
    - Transforms both train and test splits
    - Applies SMOTE strictly to the training split
    - Returns data dictionary with splits, fitted transformers, and feature column list
    """
    df = pd.read_csv(data_path)

    # Duplicate detection & removal
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        print(f"Removing {n_dupes} duplicate rows.")
        df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit transformers strictly on training data
    imputer, encoder, scaler, feature_names = fit_preprocessor(
        X_train, scaler_type=scaler_type
    )

    # Transform splits
    X_train_proc = transform_data(X_train, imputer, encoder, scaler, feature_names)
    X_test_proc = transform_data(X_test, imputer, encoder, scaler, feature_names)

    # Handle class imbalance on training set only
    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)
    else:
        X_train_res, y_train_res = X_train_proc, y_train

    return {
        "X_train": X_train_res,
        "y_train": y_train_res,
        "X_test": X_test_proc,
        "y_test": y_test,
        "imputer": imputer,
        "encoder": encoder,
        "scaler": scaler,
        "feature_names": feature_names,
    }


def save_preprocessing_artifacts(
    imputer: SimpleImputer,
    encoder: OneHotEncoder,
    scaler,
    feature_names: list,
    output_dir: str = "models/",
):
    """
    Exports individual transformer components specified in Section 9 / 16 of the architecture brief:
    - models/scaler.pkl
    - models/encoder.pkl
    - models/feature_columns.pkl
    - models/imputer.pkl
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Scaler
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    # 2. Encoder
    encoder_path = os.path.join(output_dir, "encoder.pkl")
    joblib.dump(encoder, encoder_path)

    # 3. Feature columns
    feat_path = os.path.join(output_dir, "feature_columns.pkl")
    joblib.dump(feature_names, feat_path)

    # 4. Imputer
    imputer_path = os.path.join(output_dir, "imputer.pkl")
    joblib.dump(imputer, imputer_path)

    print(f"Preprocessing artifacts successfully saved to {output_dir}:")
    print(f" - {scaler_path}")
    print(f" - {encoder_path}")
    print(f" - {feat_path}")
    print(f" - {imputer_path}")


if __name__ == "__main__":
    print("Executing preprocessing pipeline...")
    data = prepare_data()
    print("X_train shape (after SMOTE):", data["X_train"].shape)
    print("y_train balance:", data["y_train"].value_counts().to_dict())
    print("X_test shape (un-altered):", data["X_test"].shape)
    print("y_test balance:", data["y_test"].value_counts().to_dict())
    print(f"Features ({len(data['feature_names'])}):", data["feature_names"])

    save_preprocessing_artifacts(
        data["imputer"],
        data["encoder"],
        data["scaler"],
        data["feature_names"],
    )
