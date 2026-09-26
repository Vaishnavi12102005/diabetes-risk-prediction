"""
backend/app.py - FastAPI Clinical Decision Support REST API
Part of the Healthcare Disease Risk Prediction & Clinical Decision Support System.

Exposes:
- GET  /            -> Service welcome and API route directory
- GET  /health      -> Health status and model load verification
- GET  /model-info  -> Model metadata, algorithm name, parameters, and benchmark metrics
- POST /predict     -> Single patient risk prediction with SHAP explainability
- POST /batch-predict -> Vectorized batch prediction from uploaded CSV
"""

import os
import sys
import io
import pandas as pd
from datetime import datetime
from contextlib import asynccontextmanager

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    PatientInput,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from backend.predictor import predictor_instance
from utils.validation import validate_batch_dataframe


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML artifacts into memory
    print("Initializing Clinical Decision Support System backend...")
    if not predictor_instance.is_ready:
        predictor_instance.load()
    print("Backend ready to receive prediction requests.")
    yield
    # Shutdown: Clean up if needed
    print("Shutting down CDSS backend.")


app = FastAPI(
    title="Clinical Disease Risk Prediction & Decision Support System API",
    description="Full-stack supervised machine learning system predicting diabetes disease risk with SHAP explainability.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Streamlit frontend and local dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def root():
    return {
        "service": "Healthcare Disease Risk Prediction API",
        "disease_focus": "Type 2 Diabetes (Pima Indians Clinical Model)",
        "docs_url": "/docs",
        "endpoints": {
            "health": "GET /health",
            "model_info": "GET /model-info",
            "predict": "POST /predict",
            "batch_predict": "POST /batch-predict",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """
    Checks operational health of the REST API and validates that the model is loaded in memory.
    """
    return HealthResponse(
        status="HEALTHY" if predictor_instance.is_ready else "DEGRADED",
        model_loaded=predictor_instance.is_ready,
        model_name=predictor_instance.model_info.get("model_name", "Unknown"),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Governance"])
def get_model_info():
    """
    Returns governance metadata, hyperparameters, and cross-validated test metrics.
    """
    if not predictor_instance.is_ready:
        raise HTTPException(status_code=503, detail="Predictor service is initializing.")

    info = predictor_instance.model_info
    return ModelInfoResponse(
        model_name=info.get("model_name", "Logistic Regression"),
        algorithm=info.get("algorithm", "LogisticRegression"),
        trained_date=info.get("trained_date", "2026-09-25"),
        training_samples=info.get("training_samples", 800),
        test_samples=info.get("test_samples", 154),
        features=info.get("features", []),
        metrics=info.get("metrics", {}),
        parameters=info.get("parameters", {}),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_patient(patient: PatientInput):
    """
    Predicts disease risk and probability for an individual patient,
    returning a clinical recommendation and local SHAP explanations.
    """
    try:
        patient_dict = patient.model_dump(by_alias=False)
        result = predictor_instance.predict_single(patient_dict)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/batch-predict", tags=["Inference"])
async def predict_batch_csv(file: UploadFile = File(...), return_csv: bool = False):
    """
    Accepts a CSV file of patient records, performs batch risk prediction,
    and returns structured risk stratification summaries or a downloadable CSV.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .csv file.")

    try:
        content = await file.read()
        df_uploaded = pd.read_csv(io.BytesIO(content))

        # Validate required columns
        is_valid, missing, df_standardized = validate_batch_dataframe(df_uploaded)
        if not is_valid:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required clinical columns in uploaded CSV: {missing}",
            )

        results_df, summary = predictor_instance.predict_batch(df_standardized)

        if return_csv:
            # Return downloadable CSV
            output_csv = results_df.to_csv(index=False)
            return Response(
                content=output_csv,
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=predictions.csv"},
            )

        # Return JSON response
        records = results_df.to_dict(orient="records")
        return BatchPredictionResponse(
            total_records=summary["total_records"],
            high_risk_count=summary["high_risk_count"],
            low_risk_count=summary["low_risk_count"],
            average_risk_probability=summary["average_risk_probability"],
            predictions=records,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend.app:app", host="0.0.0.0", port=port, reload=False)
