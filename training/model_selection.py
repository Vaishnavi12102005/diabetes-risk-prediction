"""
model_selection.py - Algorithm Benchmarking & Hyperparameter Tuning Module
Part of the Healthcare Disease Risk Prediction & Clinical Decision Support System.

Provides:
- Definition of baseline, tree-based, and advanced classifiers:
    1. Baseline: Logistic Regression
    2. Tree-Based: Decision Tree, Random Forest, Gradient Boosting
    3. Advanced: XGBoost, Support Vector Machine (SVM)
- Unified model benchmarking and evaluation
- Systematic hyperparameter tuning with GridSearchCV
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from training.evaluate import calculate_metrics, format_metrics_table


def get_candidate_models(random_state: int = 42) -> dict:
    """
    Instantiates all 6 candidate supervised algorithms required by the project brief.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "XGBoost": XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
        ),
        "SVM": SVC(probability=True, random_state=random_state),
    }


def get_hyperparameter_grids() -> dict:
    """
    Returns search grids for hyperparameter optimization on top candidate architectures.
    """
    return {
        "Random Forest": {
            "n_estimators": [100, 200],
            "max_depth": [4, 6, 8, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"],
        },
        "Gradient Boosting": {
            "n_estimators": [100, 150, 200],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [3, 4, 5],
            "subsample": [0.8, 1.0],
        },
        "XGBoost": {
            "n_estimators": [100, 150],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [3, 4, 5],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
        "Logistic Regression": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "solver": ["lbfgs", "liblinear"],
        },
        "SVM": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
    }


def evaluate_all_models(
    X_train, y_train, X_test, y_test, random_state: int = 42
) -> tuple:
    """
    Fits and evaluates all candidate algorithms on the standardized test set.
    Returns:
        (metrics_table_df, trained_models_dict, detailed_results_dict)
    """
    models = get_candidate_models(random_state=random_state)
    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        metrics = calculate_metrics(y_test, y_pred, y_prob)
        results[name] = metrics

    comparison_df = format_metrics_table(results)
    return comparison_df, trained_models, results


def tune_model(
    model_name: str,
    base_model,
    param_grid: dict,
    X_train,
    y_train,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
    n_jobs: int = -1,
):
    """
    Performs Stratified K-Fold cross-validated hyperparameter optimization.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    return grid_search
