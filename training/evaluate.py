"""
evaluate.py - Clinical Evaluation Metrics & Trade-off Analysis Module
Part of the Healthcare Disease Risk Prediction & Clinical Decision Support System.

Provides:
- Comprehensive medical classification metrics (Accuracy, Precision, Recall/Sensitivity,
  Specificity, F1-Score, ROC-AUC)
- Confusion matrix decomposition (TP, TN, FP, FN)
- Curve generation (ROC Curve, Precision-Recall Curve)
- In-depth clinical trade-off analysis between False Positives and False Negatives
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


def calculate_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Computes all standard and clinical classification metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    roc_auc = (
        roc_auc_score(y_true, y_prob)
        if y_prob is not None and len(np.unique(y_true)) > 1
        else None
    )

    return {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),  # Sensitivity
        "Sensitivity": float(recall),
        "Specificity": float(specificity),
        "F1-Score": float(f1),
        "ROC-AUC": float(roc_auc) if roc_auc is not None else np.nan,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def format_metrics_table(models_metrics: dict) -> pd.DataFrame:
    """
    Formats a dictionary of model results into a sorted comparative DataFrame.
    """
    rows = []
    for model_name, metrics in models_metrics.items():
        row = {"Model": model_name}
        row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
        rows.append(row)

    df = pd.DataFrame(rows)
    if "ROC-AUC" in df.columns:
        df = df.sort_values(by="ROC-AUC", ascending=False)
    return df


def get_roc_data(y_true, y_prob):
    """
    Returns False Positive Rates, True Positive Rates, and thresholds for ROC curve plotting.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": float(auc_score),
    }


def get_pr_data(y_true, y_prob):
    """
    Returns Precision, Recall, and thresholds for Precision-Recall curve plotting.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }


def clinical_tradeoff_summary(fp: int, fn: int, total_tested: int) -> str:
    """
    Generates a structured medical rationale discussing the trade-off
    between False Positives and False Negatives for viva/clinical review.
    """
    text = f"""
================================================================================
CLINICAL DECISION SUPPORT SYSTEM: FALSE POSITIVE VS FALSE NEGATIVE TRADE-OFF
================================================================================
Total Test Cohort Evaluated: {total_tested} patients
- False Negatives (FN): {fn} (Diabetic patients classified as Non-Diabetic)
- False Positives (FP): {fp} (Healthy patients classified as At-Risk)

CLINICAL IMPLICATIONS:
1. False Negatives (Critical Failure Mode):
   - A False Negative means an individual with underlying insulin resistance or
     prediabetes/diabetes is falsely reassured and sent away without intervention.
   - Long-term clinical sequelae include microvascular damage (retinopathy, nephropathy,
     neuropathy) and macrovascular disease (myocardial infarction, stroke).
   - In clinical screening, Minimizing False Negatives (Maximizing Recall / Sensitivity)
     must be prioritized over pure accuracy.

2. False Positives (Manageable Operational Overhead):
   - A False Positive leads to benign confirmatory follow-up (e.g. standard laboratory
     HbA1c or Oral Glucose Tolerance Test [OGTT], lifestyle counselling).
   - While unnecessary testing induces minor anxiety and healthcare resource usage,
     it carries zero physiological harm to the patient.

3. System Recommendation:
   - In a Clinical Decision Support System (CDSS), an optimal probability cutoff
     is typically calibrated between 0.35 - 0.45 to elevate Sensitivity (>80%)
     at the expense of accepting a modest number of easily disproven False Positives.
================================================================================
"""
    return text.strip()
