"""
backend/schemas.py - Pydantic Request & Response Data Models
Part of the FastAPI backend for Healthcare Disease Risk Prediction.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class PatientInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    age: int = Field(..., ge=1, le=125, description="Patient age in years", examples=[45])
    bmi: float = Field(..., ge=0.0, le=100.0, description="Body Mass Index in kg/m²", examples=[29.5])
    glucose: float = Field(..., ge=0.0, le=600.0, description="Fasting plasma glucose in mg/dL", examples=[145.0])
    blood_pressure: float = Field(
        ...,
        alias="bloodPressure",
        ge=0.0,
        le=300.0,
        description="Diastolic blood pressure in mm Hg",
        examples=[88.0],
    )
    insulin: float = Field(
        0.0,
        ge=0.0,
        le=1200.0,
        description="2-Hour serum insulin in µU/ml (0 if unrecorded)",
        examples=[120.0],
    )
    skin_thickness: float = Field(
        0.0,
        alias="skinThickness",
        ge=0.0,
        le=120.0,
        description="Triceps skin fold thickness in mm (0 if unrecorded)",
        examples=[25.0],
    )
    pregnancies: int = Field(
        0, ge=0, le=30, description="Number of pregnancies (0 for male or nulliparous)", examples=[2]
    )
    diabetes_pedigree_function: float = Field(
        0.47,
        alias="diabetesPedigreeFunction",
        ge=0.0,
        le=3.0,
        description="Diabetes pedigree genetic predisposition score",
        examples=[0.55],
    )


class SHAPFactor(BaseModel):
    feature: str
    display_name: str
    shap_value: float
    impact_magnitude: float
    direction: str


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="'HIGH_RISK' or 'LOW_RISK'", examples=["HIGH_RISK"])
    probability: float = Field(..., description="Estimated disease risk probability [0.0 - 1.0]", examples=[0.81])
    risk_level: str = Field(..., description="Categorical risk tier: HIGH, MODERATE, or LOW", examples=["HIGH"])
    risk_percentage: float = Field(..., description="Probability expressed as percentage", examples=[81.0])
    confidence_description: str
    clinical_recommendation: str
    warnings: List[str] = Field(default_factory=list)
    increasing_factors: List[SHAPFactor] = Field(default_factory=list)
    reducing_factors: List[SHAPFactor] = Field(default_factory=list)


class BatchPredictionRecord(BaseModel):
    patient_id: str
    prediction: str
    probability: float
    risk_level: str


class BatchPredictionResponse(BaseModel):
    total_records: int
    high_risk_count: int
    low_risk_count: int
    average_risk_probability: float
    predictions: List[BatchPredictionRecord]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_name: str
    algorithm: str
    trained_date: str
    training_samples: int
    test_samples: int
    features: List[str]
    metrics: Dict[str, Any]
    parameters: Dict[str, Any]
