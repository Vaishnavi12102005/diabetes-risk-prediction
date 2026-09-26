"""
utils/visualization.py - Interactive Clinical Visualizations
Part of the Healthcare Disease Risk Prediction & Clinical Decision Support System.

Provides:
- ROC Curve and PR Curve plotting (Plotly)
- Confusion Matrix heatmap
- SHAP factor contribution breakdown charts
- Styled specifically for a clean, academic/clinical dark dashboard
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Global Plotly layout defaults for dark clinical aesthetic
CLINICAL_LAYOUT = dict(
    font=dict(family="system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif", size=12, color="#94a3b8"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=30, t=40, b=40),
)


def plot_roc_curve(fpr, tpr, roc_auc):
    """
    Creates a clean, minimal ROC curve for clinical model evaluation.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"Logistic Regression (AUC = {roc_auc:.3f})",
            line=dict(color="#38bdf8", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Chance (AUC = 0.500)",
            line=dict(color="#475569", width=1.5, dash="dash"),
        )
    )
    fig.update_layout(
        **CLINICAL_LAYOUT,
        title=dict(text="ROC Characteristic Curve", font=dict(size=14, color="#f8fafc")),
        xaxis=dict(title="False Positive Rate (1 - Specificity)", gridcolor="#1e293b", zeroline=False, color="#94a3b8"),
        yaxis=dict(title="True Positive Rate (Sensitivity)", gridcolor="#1e293b", zeroline=False, color="#94a3b8"),
        legend=dict(x=0.42, y=0.1, bgcolor="rgba(15, 23, 42, 0.7)", font=dict(color="#f8fafc")),
        height=350,
    )
    return fig


def plot_pr_curve(precision, recall):
    """
    Creates a clean Precision-Recall curve.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name="Precision-Recall",
            line=dict(color="#10b981", width=2.5),
        )
    )
    fig.update_layout(
        **CLINICAL_LAYOUT,
        title=dict(text="Precision-Recall Curve", font=dict(size=14, color="#f8fafc")),
        xaxis=dict(title="Recall (Sensitivity)", gridcolor="#1e293b", zeroline=False, color="#94a3b8"),
        yaxis=dict(title="Precision (PPV)", gridcolor="#1e293b", zeroline=False, color="#94a3b8"),
        height=350,
    )
    return fig


def plot_confusion_matrix(cm, labels=["Non-Diabetic", "Diabetic"]):
    """
    Creates a clinical confusion matrix with clear annotations.
    """
    z = cm
    x = labels
    y = labels

    annotations = []
    labels_desc = [
        ["True Negative (TN)", "False Positive (FP)"],
        ["False Negative (FN)", "True Positive (TP)"]
    ]
    for i in range(2):
        for j in range(2):
            val = z[i][j]
            text = f"<b>{val}</b><br><span style='font-size:10px; color:#94a3b8;'>{labels_desc[i][j]}</span>"
            annotations.append(
                dict(
                    x=x[j],
                    y=y[i],
                    text=text,
                    showarrow=False,
                    font=dict(color="#38bdf8" if (i==0 and j==0) or (i==1 and j==1) else "#f8fafc", size=15),
                )
            )

    fig = px.imshow(
        z,
        x=x,
        y=y,
        color_continuous_scale=[[0, "#0f172a"], [0.5, "#1e293b"], [1, "#1e3a8a"]],
        labels=dict(x="Predicted Diagnosis", y="Actual Ground Truth", color="Count"),
    )
    fig.update_layout(
        **CLINICAL_LAYOUT,
        title=dict(text="Confusion Matrix (Evaluation Cohort: n=154)", font=dict(size=14, color="#f8fafc")),
        coloraxis_showscale=False,
        annotations=annotations,
        height=350,
        xaxis=dict(color="#94a3b8"),
        yaxis=dict(color="#94a3b8"),
    )
    return fig


def plot_shap_factors(increasing_factors: list, reducing_factors: list):
    """
    Plots a clean, restrained horizontal bar chart showing risk drivers.
    """
    records = []
    for f in increasing_factors:
        records.append({
            "Feature": f.get("display_name", f["feature"]),
            "SHAP Value": f["shap_value"],
            "Clinical Impact": "Increases Risk (+)"
        })
    for f in reducing_factors:
        records.append({
            "Feature": f.get("display_name", f["feature"]),
            "SHAP Value": f["shap_value"],
            "Clinical Impact": "Decreases Risk (-)"
        })

    df = pd.DataFrame(records)
    if df.empty:
        return go.Figure()

    df = df.sort_values(by="SHAP Value")

    fig = px.bar(
        df,
        x="SHAP Value",
        y="Feature",
        color="Clinical Impact",
        orientation="h",
        color_discrete_map={
            "Increases Risk (+)": "#ef4444",
            "Decreases Risk (-)": "#10b981"
        },
    )
    fig.update_layout(
        **CLINICAL_LAYOUT,
        title=dict(text="SHAP Feature Attribution (Log-Odds Impact)", font=dict(size=14, color="#f8fafc")),
        xaxis=dict(title="Attribution to Predicted Log-Odds", gridcolor="#1e293b", zeroline=True, zerolinecolor="#475569", color="#94a3b8"),
        yaxis=dict(title="", color="#f8fafc"),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#f8fafc")),
    )
    return fig
