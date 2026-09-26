# PROJECT_STATE.md
> Intelligent Healthcare Disease Risk Prediction & Clinical Decision Support System
> This file is the single source of truth for project context across sessions/agents. Read this FIRST before doing anything. Update it LAST before ending every session.

---

## 0. Onboarding Instructions (for any new AI agent picking this up)

1. Read this entire file before touching any code.
2. Check the **Current Status** section to see exactly which phase is active and what the next task is.
3. Check **Key Decisions Log** — do not re-litigate or silently change any decision listed there.
4. Check **Known Issues / Do NOT Repeat** — these are dead ends already explored.
5. Before making changes, briefly summarize your understanding of current state back to the user to confirm alignment.
6. At the end of the session, update Section 3 (Current Status), Section 5 (Decisions Log if any new ones), Section 6 (Known Issues if any new ones), and mark off completed roadmap items in Section 2.

---

## 1. Project Overview

- **Type:** Full-stack supervised ML system (classification)
- **Domain:** Healthcare — Disease Risk Prediction & Clinical Decision Support
- **Chosen disease focus:** Diabetes (Pima Indians Diabetes dataset)
- **Dataset source:** https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv (no header row — columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome). Saved locally as `data/diabetes_data.csv` with headers added.
- **Known data quirk:** Glucose, BloodPressure, SkinThickness, Insulin, and BMI use `0` to represent missing values (not NaN) — must be treated as missing during preprocessing, not taken at face value.
- **Goal:** Predict disease risk from clinical/lifestyle parameters, output prediction + probability + risk level + explainability (SHAP), via FastAPI backend + Streamlit frontend.
- **Reference brief:** Topic 2 — Supervised Learning (Intelligent Healthcare Disease Risk Prediction & CDSS)
- **Reference training repo:** https://github.com/kpateltnsif/TNSIF_AIML_CodeBase (general AIML modules — used for underlying concepts/syntax reference, not project-specific scaffolding)

---

## 2. Build Roadmap

Ordered phases. Each phase should be completed (or at least left in a working, non-broken state) before moving to the next. Mark `[x]` when done, `[~]` when in progress, `[ ]` when not started.

### Phase 0 — Setup
- [x] Repo/folder structure created (see Section 7)
- [x] `requirements.txt` created and installable
- [x] Dataset selected and downloaded into `data/`
- [x] Chosen disease + dataset documented above in Section 1

### Phase 1 — Data & EDA
- [x] `01_EDA.ipynb`: load dataset, inspect shape/types/nulls/duplicates
- [x] Univariate + bivariate analysis (distributions, class balance, correlations)
- [x] Outlier detection documented
- [x] EDA findings summarized (what features look predictive, what's noisy)

### Phase 2 — Preprocessing & Feature Engineering
- [x] Missing value imputation strategy chosen + implemented (`training/preprocessing.py`)
- [x] Duplicate removal
- [x] Outlier handling
- [x] Categorical encoding (OneHotEncoder / ColumnTransformer)
- [x] Feature scaling (StandardScaler / RobustScaler — decision logged in Section 5)
- [x] Engineered features added: BMI Category, Age Group, Risk Factor Count
- [x] Class imbalance handling (SMOTE or alternative) — applied to train set only
- [x] Train/test split done (stratified)
- [x] Preprocessing pipeline saved as reusable object

### Phase 3 — Model Training
- [x] Baseline: Logistic Regression
- [x] Decision Tree
- [x] Random Forest
- [x] Gradient Boosting
- [x] XGBoost
- [x] SVM
- [ ] (Optional) CatBoost
- [x] Hyperparameter tuning on top candidates
- [x] Model comparison table (Accuracy, Precision, Recall, F1, ROC-AUC, Specificity, Sensitivity)
- [x] Best model selected + rationale logged in Section 5
- [x] False positive vs false negative trade-off discussion written up (for viva)

### Phase 4 — Model Artifacts
- [x] `disease_model.pkl` exported
- [x] `scaler.pkl` exported
- [x] `encoder.pkl` exported
- [x] `feature_columns.pkl` exported
- [x] (or) single combined `disease_pipeline.pkl`
- [x] Verified: model loads and predicts without retraining

### Phase 5 — Backend (FastAPI)
- [x] `GET /health`
- [x] `GET /model-info`
- [x] `POST /predict` (single patient)
- [x] `POST /batch-predict` (CSV upload)
- [x] Input validation via `schemas.py` (pydantic)
- [x] `predictor.py` loads artifacts once at startup, not per-request
- [x] Manual test of all endpoints (curl/Postman) passing

### Phase 6 — Frontend (Streamlit)
- [x] Page 1 — Project Dashboard (record counts, class distribution, model used, performance)
- [x] Page 2 — Individual Risk Prediction (form + result card)
- [x] Page 3 — Explainable Prediction (SHAP factors up/down)
- [x] Page 4 — Population Analytics (filterable charts)
- [x] Page 5 — Model Evaluation (confusion matrix, ROC, PR curve, feature importance, SHAP summary)
- [x] Frontend wired to backend (not calling model directly, with resilient local fallback)

### Phase 7 — Explainability (SHAP)
- [x] SHAP explainer set up for chosen model type
- [x] Per-prediction explanation working (increasing/decreasing factors)
- [x] SHAP summary plot for global feature importance

### Phase 8 — Documentation & Wrap-up
- [x] README with architecture, dataset description, methodology, results, screenshots
- [x] Limitations section written
- [x] Future scope section written
- [x] Final model comparison + chosen model justification documented

---

## 3. Current Status

**Last updated:** 2026-09-25
**Active phase:** Complete (All Phases 0 through 8 Completed)
**Exact next task:** System is fully functional end-to-end. Ready for clinical demo, user testing, or deployment (run `uvicorn backend.app:app --port 8000` and `streamlit run frontend/streamlit_app.py`).
**What's working right now:** Full stack operational: (1) Dataset downloaded and verified; (2) Complete EDA notebook `01_EDA.ipynb`; (3) Preprocessing and feature engineering pipeline with SMOTE; (4) Benchmarking across 6 supervised algorithms, tuning, and evaluation in `02_Model_Training.ipynb` and `training/train.py`; (5) Exported champion model and transformers in `models/`; (6) FastAPI REST API with endpoints `/health`, `/model-info`, `/predict`, `/batch-predict`; (7) Local & global SHAP explainability in `utils/explainability.py`; (8) Interactive 5-page Streamlit clinical frontend with batch CSV prediction; (9) Complete `README.md` system documentation.
**What's broken / half-done:** None. All deliverables verified.

---

## 4. Environment & How to Run

- Python version: 3.14.4
- Key package versions:
  - numpy: 2.4.4
  - pandas: 3.0.2
  - scikit-learn: 1.8.0
  - scipy: 1.17.1
  - matplotlib: 3.10.9
  - seaborn: 0.13.2
  - plotly: 7.1.0
  - joblib: 1.5.3
  - fastapi: 0.141.1
  - uvicorn: 0.53.0
  - streamlit: 1.64.0
  - shap: 0.52.0
  - imbalanced-learn: 0.14.2
  - xgboost: 3.4.1
- Install: `pip install -r requirements.txt`
- Run backend: `uvicorn backend.app:app --reload`
- Run frontend: `streamlit run frontend/streamlit_app.py`
- Run notebooks: Jupyter / Google Colab

---

## 5. Key Decisions Log

> Append new entries at the bottom, don't delete old ones. Each decision should include the "why" so it isn't silently reversed later.

| Date | Decision | Reason |
|------|----------|--------|
| 2026-09-25 | Chose diabetes dataset (Pima Indians) over heart disease | Cleaner public dataset, well-documented clinical features, standard benchmark |
| 2026-09-25 | Verified Python 3.14.4 compatibility for core ML stack (all 14 packages including SHAP 0.52.0 & XGBoost 3.4.1) | Clean installation achieved without needing fallback or alternative explainers |
| 2026-09-25 | Omitted CatBoost from initial requirements | Core stack installed cleanly; per project brief CatBoost is optional and will be evaluated only if tree-based models need an extra candidate |
| 2026-09-25 | Retained raw zero values during initial EDA | Zeros in Glucose, BP, SkinThickness, Insulin, and BMI represent missing values; they are flagged in EDA and deliberately preserved until Phase 2 preprocessing |
| 2026-09-25 | Selected Median Imputation for physiological missing values | Severe right-skewness and extreme outliers in Insulin and SkinThickness make mean imputation statistically biased |
| 2026-09-25 | Selected RobustScaler for feature scaling | Preserves relative outlier distances using median and IQR without compressing feature variances as severely as standard z-score scaling |
| 2026-09-25 | Applied SMOTE strictly to training data split | Oversampling the minority class on the entire dataset or test set causes severe optimistic data leakage; test set remains pristine |
| 2026-09-25 | Serialized pure scikit-learn transformers (`imputer.pkl`, `encoder.pkl`, `scaler.pkl`, `feature_columns.pkl`) | Standard native serialization avoids custom class unpickling path collisions across Python 3.14 execution contexts |
| 2026-09-25 | Selected Logistic Regression as Champion Model (with Random Forest runner-up) | Delivered peak test ROC-AUC (0.8285), tied for highest Sensitivity/Recall (74.07%), and minimized False Negatives (14/54) |
| 2026-09-25 | Prioritized Sensitivity/Recall over raw Accuracy | In healthcare screening, a False Negative (missed diabetic patient) carries severe health risks, whereas a False Positive only incurs benign confirmatory lab testing |
| 2026-09-25 | Singleton RiskPredictor lifecycle in FastAPI | Model artifacts and background SHAP explainer loaded once at application lifespan startup to eliminate per-request disk I/O latency |
| 2026-09-25 | Dual-mode batch inference (`/batch-predict`) | Supports both JSON response and direct downloadable CSV with Content-Disposition attachment |
| 2026-09-25 | Integrated local SHAP explainability directly into `/predict` response | Returns top factors increasing vs reducing risk alongside probability so every prediction is immediately explainable |


---

## 6. Known Issues / Do NOT Repeat

> Dead ends already explored — record them so a new agent doesn't waste a session rediscovering them.

- _[e.g. "XGBoost + SMOTE without stratified CV caused optimistic F1 — don't re-add without stratification"]_

---

## 7. Project Structure (target)

```
health-risk-prediction/
│
├── data/
│   └── dataset.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_Model_Training.ipynb
│
├── training/
│   ├── train.py
│   ├── preprocessing.py
│   ├── evaluate.py
│   └── model_selection.py
│
├── models/
│   ├── disease_model.pkl
│   ├── scaler.pkl
│   ├── encoder.pkl
│   └── feature_columns.pkl
│
├── backend/
│   ├── app.py
│   ├── predictor.py
│   ├── schemas.py
│   └── preprocessing.py
│
├── frontend/
│   └── streamlit_app.py
│
├── utils/
│   ├── visualization.py
│   ├── explainability.py
│   └── validation.py
│
├── outputs/
│   └── predictions.csv
│
├── requirements.txt
├── README.md
├── PROJECT_STATE.md   ← this file
└── .gitignore
```

**Status vs target:**
- `data/`: Exists. Contains raw dataset (`dataset.csv`), download script (`download_data.py`), and formatted dataset (`diabetes_data.csv`).
- `notebooks/`: Exists. Contains `01_EDA.ipynb` (complete EDA) and `02_Model_Training.ipynb` (training, tuning, benchmarking).
- `training/`: Exists. Contains `__init__.py`, `preprocessing.py`, `evaluate.py`, `model_selection.py`, `train.py`.
- `models/`: Exists. Contains `disease_model.pkl`, `model_info.json`, `imputer.pkl`, `encoder.pkl`, `scaler.pkl`, `feature_columns.pkl`.
- `backend/`: Exists. Contains `__init__.py`, `schemas.py`, `predictor.py`, `preprocessing.py`, `app.py`.
- `frontend/`: Exists. Contains `streamlit_app.py` (5-page interactive clinical dashboard + batch prediction).
- `utils/`: Exists. Contains `__init__.py`, `validation.py`, `explainability.py`, `visualization.py`.
- `outputs/`: Exists with `model_comparison_leaderboard.csv`.
- Root files: `requirements.txt`, `.gitignore`, `PROJECT_STATE.md`, `README.md` all exist.

---

## 8. Session Handoff Checklist (do this before ending every session)

- [x] Section 3 (Current Status) updated with today's progress and the exact next task
- [x] Section 2 (Roadmap) checkboxes updated
- [x] Section 5 (Decisions Log) — any new decisions added
- [x] Section 6 (Known Issues) — any new dead ends added
- [x] File saved in project root so it's visible to the next login/session

