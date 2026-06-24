from __future__ import annotations

import json
import math
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, model_validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"
DEFAULT_MODEL_PATH = MODEL_DIR / "model.joblib"
DEFAULT_MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"
DEFAULT_PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"

MODEL_PATH_ENV = "MODEL_PATH"
MODEL_METADATA_PATH_ENV = "MODEL_METADATA_PATH"
PREPROCESSOR_PATH_ENV = "PREPROCESSOR_PATH"

MODEL_SUFFIXES = {".joblib", ".pkl", ".pickle"}
KNOWN_MODEL_FILENAMES = [
    "model.joblib",
    "classifier.joblib",
    "credit_score_model.joblib",
    "best_model.joblib",
    "lightgbm_model.joblib",
    "xgboost_model.joblib",
    "catboost_model.joblib",
    "model.pkl",
    "classifier.pkl",
]

SCORE_LABELS = {
    0: "Poor",
    1: "Standard",
    2: "Good",
    "0": "Poor",
    "1": "Standard",
    "2": "Good",
}
PROBABILITY_FIELDS_BY_LABEL = {
    "Poor": "poor",
    "Standard": "standard",
    "Good": "good",
}

CreditScoreLabel = Literal["Poor", "Standard", "Good"]
ServiceStatus = Literal[
    "ok",
    "missing_model",
    "model_error",
    "missing_preprocessor",
    "preprocessor_error",
]
RawFeatureValue = str | int | float | None


def configured_path(env_name: str, default: Path) -> Path:
    raw_path = os.getenv(env_name)
    if not raw_path:
        return default

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_model_metadata(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, f"Model metadata not found: {path}"

    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return {}, str(exc)


MODEL_METADATA_PATH = configured_path(MODEL_METADATA_PATH_ENV, DEFAULT_MODEL_METADATA_PATH)
MODEL_METADATA, MODEL_METADATA_ERROR = read_model_metadata(MODEL_METADATA_PATH)
MODEL_FEATURE_NAMES = [str(feature) for feature in MODEL_METADATA.get("feature_names", [])]
MODEL_LABEL_CLASSES = [str(label) for label in MODEL_METADATA.get("label_classes", [])]


class RawCreditScoreFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ID: RawFeatureValue = Field(default=None, description="Optional technical row id.")
    Customer_ID: RawFeatureValue = Field(default=None, description="Optional customer id used for profile imputation.")
    Month: RawFeatureValue = Field(..., description="Observation month, for example January.")
    Name: RawFeatureValue = Field(default=None, description="Optional customer name; dropped by preprocessing.")
    Age: RawFeatureValue = Field(..., description="Customer age.")
    SSN: RawFeatureValue = Field(default=None, description="Optional social security number; dropped by preprocessing.")
    Occupation: RawFeatureValue = Field(..., description="Customer occupation.")
    Annual_Income: RawFeatureValue = Field(..., description="Annual income.")
    Monthly_Inhand_Salary: RawFeatureValue = Field(..., description="Monthly net salary.")
    Num_Bank_Accounts: RawFeatureValue = Field(..., description="Number of bank accounts.")
    Num_Credit_Card: RawFeatureValue = Field(..., description="Number of credit cards.")
    Interest_Rate: RawFeatureValue = Field(..., description="Interest rate.")
    Num_of_Loan: RawFeatureValue = Field(..., description="Number of loans.")
    Type_of_Loan: RawFeatureValue = Field(..., description="Loan types as a comma-separated raw string.")
    Delay_from_due_date: RawFeatureValue = Field(..., description="Payment delay in days.")
    Num_of_Delayed_Payment: RawFeatureValue = Field(..., description="Number of delayed payments.")
    Changed_Credit_Limit: RawFeatureValue = Field(..., description="Changed credit limit.")
    Num_Credit_Inquiries: RawFeatureValue = Field(..., description="Number of credit inquiries.")
    Credit_Mix: RawFeatureValue = Field(..., description="Raw credit mix value.")
    Outstanding_Debt: RawFeatureValue = Field(..., description="Outstanding debt.")
    Credit_Utilization_Ratio: RawFeatureValue = Field(..., description="Credit utilization ratio.")
    Credit_History_Age: RawFeatureValue = Field(..., description="Credit history age, for example 22 Years and 1 Months.")
    Payment_of_Min_Amount: RawFeatureValue = Field(..., description="Whether the customer pays the minimum amount.")
    Total_EMI_per_month: RawFeatureValue = Field(..., description="Total EMI per month.")
    Amount_invested_monthly: RawFeatureValue = Field(..., description="Monthly invested amount.")
    Payment_Behaviour: RawFeatureValue = Field(..., description="Raw payment behaviour category.")
    Monthly_Balance: RawFeatureValue = Field(..., description="Monthly balance.")
    Credit_Score: RawFeatureValue = Field(default=None, description="Optional target value; ignored during inference.")


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: RawCreditScoreFeatures | None = Field(
        default=None,
        description="Single raw feature row. The API applies the saved preprocessor before prediction.",
    )
    records: list[RawCreditScoreFeatures] | None = Field(
        default=None,
        description="Batch of raw feature rows. The API applies the saved preprocessor before prediction.",
    )
    return_probabilities: bool = Field(
        default=True,
        description="Return predict_proba output when the model supports it.",
    )

    @model_validator(mode="after")
    def validate_payload(self) -> "PredictRequest":
        has_features = self.features is not None
        has_records = self.records is not None
        if has_features == has_records:
            raise ValueError("Provide exactly one of 'features' or 'records'.")

        if self.records is not None and not self.records:
            raise ValueError("'records' must contain at least one item.")

        return self

    def as_records(self) -> list[dict[str, Any]]:
        if self.features is not None:
            return [self.features.model_dump()]
        return [record.model_dump() for record in self.records or []]


class CreditScoreProbabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    poor: float | None = Field(default=None, description="Probability of class 0: Poor.")
    standard: float | None = Field(default=None, description="Probability of class 1: Standard.")
    good: float | None = Field(default=None, description="Probability of class 2: Good.")


class PredictionItem(BaseModel):
    index: int
    prediction_code: int = Field(description="Numeric model class: 0=Poor, 1=Standard, 2=Good.")
    prediction_label: CreditScoreLabel = Field(description="Human-readable credit score label.")
    probabilities: CreditScoreProbabilities | None = None


class PredictResponse(BaseModel):
    model_path: str
    preprocessor_path: str
    model_name: str | None = None
    feature_count: int
    predictions: list[PredictionItem]


class HealthResponse(BaseModel):
    status: ServiceStatus
    model_loaded: bool
    model_path: str
    model_error: str | None = None
    model_metadata_path: str
    model_metadata_loaded: bool
    model_metadata_error: str | None = None
    feature_count: int
    label_classes: list[str]
    preprocessor_path: str
    preprocessor_loaded: bool
    preprocessor_error: str | None = None


@dataclass
class RuntimeArtifacts:
    model: Any | None = None
    model_path: Path = DEFAULT_MODEL_PATH
    model_error: str | None = None
    preprocessor: Any | None = None
    preprocessor_path: Path = DEFAULT_PREPROCESSOR_PATH
    preprocessor_error: str | None = None

    def load(self) -> None:
        self.load_model()
        self.load_preprocessor(required=True)

    def load_model(self) -> None:
        self.model_path = resolve_model_path()
        self.model = None
        self.model_error = None

        try:
            self.model = load_model(self.model_path)
        except Exception as exc:  # pragma: no cover - exact loader errors are environment dependent.
            self.model_error = str(exc)

    def load_preprocessor(self, *, required: bool) -> None:
        self.preprocessor_path = configured_path(PREPROCESSOR_PATH_ENV, DEFAULT_PREPROCESSOR_PATH)
        self.preprocessor = None
        self.preprocessor_error = None

        if not self.preprocessor_path.exists():
            if required:
                self.preprocessor_error = f"Preprocessor artifact not found: {self.preprocessor_path}"
            return

        try:
            self.preprocessor = joblib.load(self.preprocessor_path)
        except Exception as exc:  # pragma: no cover - exact loader errors are environment dependent.
            self.preprocessor_error = str(exc)


artifacts = RuntimeArtifacts()


def resolve_model_path() -> Path:
    explicit_path = os.getenv(MODEL_PATH_ENV)
    if explicit_path:
        return configured_path(MODEL_PATH_ENV, DEFAULT_MODEL_PATH)

    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH

    for filename in KNOWN_MODEL_FILENAMES:
        candidate = MODEL_DIR / filename
        if candidate.exists():
            return candidate

    if MODEL_DIR.exists():
        for candidate in sorted(MODEL_DIR.iterdir()):
            name = candidate.name.lower()
            if (
                candidate.is_file()
                and candidate.suffix.lower() in MODEL_SUFFIXES
                and "preprocessor" not in name
            ):
                return candidate

    return DEFAULT_MODEL_PATH


def load_model(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}. Set {MODEL_PATH_ENV} or place a model at {DEFAULT_MODEL_PATH}."
        )

    model = joblib.load(path)
    if not callable(getattr(model, "predict", None)):
        raise TypeError(f"Loaded artifact does not expose predict(): {path}")
    return model


def service_status() -> ServiceStatus:
    if artifacts.model is None:
        if not artifacts.model_path.exists():
            return "missing_model"
        return "model_error"

    if artifacts.preprocessor is None:
        if not artifacts.preprocessor_path.exists():
            return "missing_preprocessor"
        return "preprocessor_error"

    return "ok"


def get_model_or_503() -> Any:
    if artifacts.model is None:
        artifacts.load_model()

    if artifacts.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "Model is not available.",
                "model_path": str(artifacts.model_path),
                "error": artifacts.model_error,
            },
        )

    return artifacts.model


def get_preprocessor_or_503() -> Any:
    if artifacts.preprocessor is None:
        artifacts.load_preprocessor(required=True)

    if artifacts.preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "Preprocessor is not available.",
                "preprocessor_path": str(artifacts.preprocessor_path),
                "error": artifacts.preprocessor_error,
            },
        )

    return artifacts.preprocessor


def dataframe_from_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Request payload produced an empty dataframe.",
        )
    return frame


def prepare_features(frame: pd.DataFrame, model: Any) -> pd.DataFrame:
    preprocessor = get_preprocessor_or_503()
    try:
        transformed = preprocessor.transform(frame)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Preprocessing failed for the provided raw features.",
                "error": str(exc),
            },
        ) from exc

    if not isinstance(transformed, pd.DataFrame):
        transformed = pd.DataFrame(transformed)

    return align_features_for_model(transformed, model)


def expected_model_features(model: Any) -> list[str]:
    if MODEL_FEATURE_NAMES:
        return MODEL_FEATURE_NAMES

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        return [str(feature) for feature in list(feature_names)]
    return []


def align_features_for_model(frame: pd.DataFrame, model: Any) -> pd.DataFrame:
    expected_features = expected_model_features(model)
    if expected_features:
        missing_features = [feature for feature in expected_features if feature not in frame.columns]
        if missing_features:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "message": "Preprocessor output does not match model features.",
                    "missing_features": missing_features,
                },
            )
        return frame[expected_features]

    expected_count = getattr(model, "n_features_in_", None)
    if expected_count is not None and frame.shape[1] != int(expected_count):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Preprocessor output feature count does not match model.",
                "preprocessor_feature_count": int(frame.shape[1]),
                "model_feature_count": int(expected_count),
            },
        )

    return frame


def normalize_prediction(value: Any) -> tuple[int, CreditScoreLabel]:
    prediction = int(to_jsonable(value))
    label = SCORE_LABELS.get(prediction)
    if label is None:
        label = SCORE_LABELS.get(str(prediction))
    if label is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unknown model class returned: {prediction}",
        )
    return prediction, label


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def finite_float_or_none(value: Any) -> float | None:
    number = float(value)
    if math.isfinite(number):
        return number
    return None


def probability_keys(model: Any, width: int) -> list[str]:
    raw_classes = getattr(model, "classes_", None)
    if raw_classes is None:
        raw_classes = list(range(width))

    keys = []
    for raw_class in list(raw_classes):
        class_value = to_jsonable(raw_class)
        keys.append(SCORE_LABELS.get(class_value) or SCORE_LABELS.get(str(class_value)) or str(class_value))
    return keys


def probabilities_from_row(keys: list[str], row: np.ndarray) -> CreditScoreProbabilities:
    values: dict[str, float | None] = {
        "poor": None,
        "standard": None,
        "good": None,
    }
    for key, value in zip(keys, row, strict=False):
        field_name = PROBABILITY_FIELDS_BY_LABEL.get(key)
        if field_name is not None:
            values[field_name] = finite_float_or_none(value)

    return CreditScoreProbabilities(**values)


def predict_probabilities(model: Any, frame: pd.DataFrame, expected_rows: int) -> list[CreditScoreProbabilities] | None:
    predict_proba = getattr(model, "predict_proba", None)
    if not callable(predict_proba):
        return None

    raw_probabilities = predict_proba(frame)
    probabilities = np.asarray(raw_probabilities)
    if probabilities.ndim != 2 or probabilities.shape[0] != expected_rows:
        return None

    keys = probability_keys(model, probabilities.shape[1])
    return [probabilities_from_row(keys, row) for row in probabilities]


@asynccontextmanager
async def lifespan(_: FastAPI):
    artifacts.load()
    yield


app = FastAPI(
    title="Credit Score Prediction API",
    version="1.0.0",
    description="FastAPI service for raw credit score data preprocessing and prediction.",
    lifespan=lifespan,
)


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "service": "credit-score-prediction-api",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status=service_status(),
        model_loaded=artifacts.model is not None,
        model_path=str(artifacts.model_path),
        model_error=artifacts.model_error,
        model_metadata_path=str(MODEL_METADATA_PATH),
        model_metadata_loaded=bool(MODEL_METADATA),
        model_metadata_error=MODEL_METADATA_ERROR,
        feature_count=len(MODEL_FEATURE_NAMES),
        label_classes=MODEL_LABEL_CLASSES,
        preprocessor_path=str(artifacts.preprocessor_path),
        preprocessor_loaded=artifacts.preprocessor is not None,
        preprocessor_error=artifacts.preprocessor_error,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    model = get_model_or_503()
    raw_frame = dataframe_from_records(request.as_records())
    model_frame = prepare_features(raw_frame, model)

    try:
        raw_predictions = np.atleast_1d(np.asarray(model.predict(model_frame)))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Model prediction failed after preprocessing.",
                "error": str(exc),
            },
        ) from exc

    if raw_predictions.ndim > 1 and raw_predictions.shape[1] == 1:
        raw_predictions = raw_predictions.ravel()

    probabilities = None
    if request.return_probabilities:
        try:
            probabilities = predict_probabilities(model, model_frame, len(raw_predictions))
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "message": "Model probability prediction failed after preprocessing.",
                    "error": str(exc),
                },
            ) from exc

    items = []
    for index, raw_prediction in enumerate(raw_predictions):
        prediction_code, prediction_label = normalize_prediction(raw_prediction)
        items.append(
            PredictionItem(
                index=index,
                prediction_code=prediction_code,
                prediction_label=prediction_label,
                probabilities=probabilities[index] if probabilities is not None else None,
            )
        )

    return PredictResponse(
        model_path=str(artifacts.model_path),
        preprocessor_path=str(artifacts.preprocessor_path),
        model_name=MODEL_METADATA.get("model"),
        feature_count=model_frame.shape[1],
        predictions=items,
    )
