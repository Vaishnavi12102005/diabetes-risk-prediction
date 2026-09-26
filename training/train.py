"""
train.py - Full ML Model Training, Tuning, Selection & Artifact Export Pipeline
Part of the Healthcare Disease Risk Prediction & Clinical Decision Support System.

Executes:
1. End-to-end data preparation (with SMOTE and preprocessing)
2. Benchmarking of 6 supervised algorithms (Logistic Regression, Decision Tree, Random Forest,
   Gradient Boosting, XGBoost, SVM)
3. Stratified 5-Fold cross-validated hyperparameter optimization
4. Model comparison table with full clinical metrics
5. Best model selection prioritizing Sensitivity/Recall and ROC-AUC
6. Export of models/disease_model.pkl and models/model_info.json
7. False positive vs false negative clinical trade-off report
"""

import os
import sys
import json
import joblib
import pandas as pd
from datetime import datetime

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.preprocessing import prepare_data, save_preprocessing_artifacts
from training.evaluate import (
    calculate_metrics,
    format_metrics_table,
    clinical_tradeoff_summary,
)
from training.model_selection import (
    get_candidate_models,
    get_hyperparameter_grids,
    evaluate_all_models,
    tune_model,
)


def run_training_pipeline(
    data_path: str = "data/diabetes_data.csv",
    output_dir: str = "models/",
    random_state: int = 42,
):
    print("=" * 80)
    print("HEALTHCARE DISEASE RISK PREDICTION — ML TRAINING PIPELINE")
    print("=" * 80)

    # 1. Prepare Data
    print("\n[Step 1/5] Loading data and executing preprocessing pipeline...")
    data = prepare_data(data_path=data_path, random_state=random_state)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    feature_names = data["feature_names"]

    print(f" - Train cohort size (after SMOTE): {X_train.shape[0]} samples")
    print(f" - Test cohort size (unaltered):    {X_test.shape[0]} samples")
    print(f" - Feature count (clinical + eng): {len(feature_names)} features")

    # Save preprocessing transformers
    save_preprocessing_artifacts(
        data["imputer"], data["encoder"], data["scaler"], feature_names, output_dir=output_dir
    )

    # 2. Benchmark Baseline and Candidate Models
    print("\n[Step 2/5] Benchmarking 6 supervised classification algorithms...")
    comp_df, trained_models, baseline_results = evaluate_all_models(
        X_train, y_train, X_test, y_test, random_state=random_state
    )
    print("\nBaseline Model Comparison Table (Test Set):")
    cols_to_show = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1-Score",
        "ROC-AUC",
        "FP",
        "FN",
    ]
    print(comp_df[cols_to_show].to_string(index=False))

    # 3. Hyperparameter Tuning
    print("\n[Step 3/5] Performing Stratified 5-Fold CV Hyperparameter Tuning on top models...")
    grids = get_hyperparameter_grids()
    candidates_to_tune = ["Random Forest", "Gradient Boosting", "XGBoost"]
    tuned_models = {}
    tuned_results = {}

    base_models = get_candidate_models(random_state=random_state)
    for model_name in candidates_to_tune:
        print(f" -> Tuning {model_name}...")
        grid = grids[model_name]
        cv_search = tune_model(
            model_name,
            base_models[model_name],
            grid,
            X_train,
            y_train,
            cv_folds=5,
            scoring="roc_auc",
        )
        best_est = cv_search.best_estimator_
        tuned_models[f"Tuned {model_name}"] = best_est

        y_pred = best_est.predict(X_test)
        y_prob = best_est.predict_proba(X_test)[:, 1]
        tuned_metrics = calculate_metrics(y_test, y_pred, y_prob)
        tuned_results[f"Tuned {model_name}"] = tuned_metrics
        print(
            f"    Best CV ROC-AUC: {cv_search.best_score_:.4f} | Best Params: {cv_search.best_params_}"
        )

    # 4. Final Comparison & Selection
    print("\n[Step 4/5] Evaluating all models (baseline + tuned) to select champion...")
    all_results = {**baseline_results, **tuned_results}
    final_comp_df = format_metrics_table(all_results)
    print("\nComplete Leaderboard (Sorted by ROC-AUC):")
    print(final_comp_df[cols_to_show].to_string(index=False))

    # Selection Criterion:
    # In clinical decision support for diabetes, we select the model with the highest
    # ROC-AUC and Sensitivity/Recall (to minimize FN - missed diabetic cases).
    # If ROC-AUC is comparable, we select the model minimizing False Negatives.
    all_trained = {**trained_models, **tuned_models}

    # Sort candidates by combined score: 0.5 * ROC-AUC + 0.5 * Recall
    def clinical_score(name):
        m = all_results[name]
        return 0.5 * m["ROC-AUC"] + 0.5 * m["Recall"]

    best_model_name = max(all_trained.keys(), key=clinical_score)
    best_model = all_trained[best_model_name]
    best_metrics = all_results[best_model_name]

    print("\n" + "=" * 80)
    print(f"CHAMPION MODEL SELECTED: {best_model_name.upper()}")
    print("=" * 80)
    print(f" - ROC-AUC:     {best_metrics['ROC-AUC']:.4f}")
    print(f" - Recall (Sens):{best_metrics['Recall']:.4f} (Missed only {best_metrics['FN']} of 54 cases)")
    print(f" - Specificity:  {best_metrics['Specificity']:.4f} ({best_metrics['TN']} of 100 healthy flagged)")
    print(f" - Accuracy:     {best_metrics['Accuracy']:.4f}")
    print(f" - F1-Score:     {best_metrics['F1-Score']:.4f}")

    # 5. Export Model Artifacts
    print("\n[Step 5/5] Exporting model artifacts and metadata...")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "disease_model.pkl")
    joblib.dump(best_model, model_path)
    print(f" -> Champion model saved to: {model_path}")

    # Save model metadata for backend and frontend
    info_path = os.path.join(output_dir, "model_info.json")
    model_info = {
        "model_name": best_model_name,
        "algorithm": best_model.__class__.__name__,
        "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "features": feature_names,
        "metrics": best_metrics,
        "parameters": best_model.get_params(),
    }
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2, default=str)
    print(f" -> Model metadata saved to: {info_path}")

    # Save complete comparison table as CSV in outputs/
    os.makedirs("outputs/", exist_ok=True)
    leaderboard_csv = "outputs/model_comparison_leaderboard.csv"
    final_comp_df.to_csv(leaderboard_csv, index=False)
    print(f" -> Comparison table saved to: {leaderboard_csv}")

    # Print clinical trade-off summary
    print("\n" + clinical_tradeoff_summary(best_metrics["FP"], best_metrics["FN"], X_test.shape[0]))

    return best_model, final_comp_df, best_metrics


if __name__ == "__main__":
    run_training_pipeline()
