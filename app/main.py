from __future__ import annotations

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
DEFAULT_PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"

MODEL_PATH_ENV = "MODEL_PATH"
PREPROCESSOR_PATH_ENV = "PREPROCESSOR_PATH"
APPLY_PREPROCESSOR_ENV = "APPLY_PREPROCESSOR"

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


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, Any] | None = Field(
        default=None,
        description="Single processed feature row.",
    )
    records: list[dict[str, Any]] | None = Field(
        default=None,
        description="Batch of processed feature rows.",
    )
    apply_preprocessing: bool | None = Field(
        default=None,
        description="Use saved raw-data preprocessor before prediction. Defaults to APPLY_PREPROCESSOR.",
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
            return [self.features]
        return list(self.records or [])


class PredictionItem(BaseModel):
    index: int
    prediction: Any
    prediction_label: str | None = None
    probabilities: dict[str, float | None] | None = None


class PredictResponse(BaseModel):
    model_path: str
    used_preprocessing: bool
    predictions: list[PredictionItem]


class HealthResponse(BaseModel):
    status: Literal["ok", "missing_model", "model_error"]
    model_loaded: bool
    model_path: str
    model_error: str | None = None
    preprocessor_path: str
    preprocessor_loaded: bool
    preprocessor_error: str | None = None
    default_apply_preprocessing: bool


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
        self.load_preprocessor(required=False)

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


def configured_path(env_name: str, default: Path) -> Path:
    raw_path = os.getenv(env_name)
    if not raw_path:
        return default

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def env_bool(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


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


def model_status() -> Literal["ok", "missing_model", "model_error"]:
    if artifacts.model is not None:
        return "ok"
    if not artifacts.model_path.exists():
        return "missing_model"
    return "model_error"


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


def prepare_features(frame: pd.DataFrame, apply_preprocessing: bool) -> pd.DataFrame:
    if not apply_preprocessing:
        return frame

    preprocessor = get_preprocessor_or_503()
    try:
        transformed = preprocessor.transform(frame)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Preprocessing failed for the provided features.",
                "error": str(exc),
            },
        ) from exc

    if isinstance(transformed, pd.DataFrame):
        return transformed
    return pd.DataFrame(transformed)


def normalize_prediction(value: Any) -> tuple[Any, str | None]:
    prediction = to_jsonable(value)
    label = SCORE_LABELS.get(prediction)
    if label is None:
        label = SCORE_LABELS.get(str(prediction))
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


def predict_probabilities(model: Any, frame: pd.DataFrame, expected_rows: int) -> list[dict[str, float | None]] | None:
    predict_proba = getattr(model, "predict_proba", None)
    if not callable(predict_proba):
        return None

    raw_probabilities = predict_proba(frame)
    probabilities = np.asarray(raw_probabilities)
    if probabilities.ndim != 2 or probabilities.shape[0] != expected_rows:
        return None

    keys = probability_keys(model, probabilities.shape[1])
    return [
        {
            key: finite_float_or_none(value)
            for key, value in zip(keys, row, strict=False)
        }
        for row in probabilities
    ]


@asynccontextmanager
async def lifespan(_: FastAPI):
    artifacts.load()
    yield


app = FastAPI(
    title="Credit Score Prediction API",
    version="1.0.0",
    description="FastAPI service for running a trained credit score model on processed features.",
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
        status=model_status(),
        model_loaded=artifacts.model is not None,
        model_path=str(artifacts.model_path),
        model_error=artifacts.model_error,
        preprocessor_path=str(artifacts.preprocessor_path),
        preprocessor_loaded=artifacts.preprocessor is not None,
        preprocessor_error=artifacts.preprocessor_error,
        default_apply_preprocessing=env_bool(APPLY_PREPROCESSOR_ENV),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    model = get_model_or_503()
    raw_frame = dataframe_from_records(request.as_records())
    use_preprocessing = (
        request.apply_preprocessing
        if request.apply_preprocessing is not None
        else env_bool(APPLY_PREPROCESSOR_ENV)
    )
    model_frame = prepare_features(raw_frame, use_preprocessing)

    try:
        raw_predictions = np.atleast_1d(np.asarray(model.predict(model_frame)))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Model prediction failed for the provided features.",
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
                    "message": "Model probability prediction failed for the provided features.",
                    "error": str(exc),
                },
            ) from exc

    items = []
    for index, raw_prediction in enumerate(raw_predictions):
        prediction, prediction_label = normalize_prediction(raw_prediction)
        items.append(
            PredictionItem(
                index=index,
                prediction=prediction,
                prediction_label=prediction_label,
                probabilities=probabilities[index] if probabilities is not None else None,
            )
        )

    return PredictResponse(
        model_path=str(artifacts.model_path),
        used_preprocessing=use_preprocessing,
        predictions=items,
    )
