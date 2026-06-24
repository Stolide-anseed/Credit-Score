from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder


DEFAULT_FEATURES_PATH = Path("data") / "X_train_prepared.csv"
DEFAULT_TARGET_PATH = Path("data") / "y_train.csv"
DEFAULT_PARAMS_PATH = Path("Metrics") / "lightgbm_optuna_best_params.json"
DEFAULT_MODEL_OUTPUT_PATH = Path("model") / "model.joblib"
DEFAULT_METADATA_OUTPUT_PATH = Path("model") / "model_metadata.json"
DEFAULT_SPLIT_METADATA_PATH = Path("data") / "preprocessing_split_metadata.json"
DEFAULT_PREPROCESSOR_METADATA_PATH = Path("model") / "preprocessor_metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train final LightGBM model on prepared features using best Optuna "
            "hyperparameters."
        )
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help="CSV with prepared numeric features.",
    )
    parser.add_argument(
        "--target-path",
        type=Path,
        default=DEFAULT_TARGET_PATH,
        help="CSV with target values aligned by source_index.",
    )
    parser.add_argument(
        "--target",
        default="Credit_Score",
        help="Target column name.",
    )
    parser.add_argument(
        "--index-column",
        default="source_index",
        help="Index column used to align features and target.",
    )
    parser.add_argument(
        "--params-path",
        type=Path,
        default=DEFAULT_PARAMS_PATH,
        help="JSON artifact produced by lightgbm_optuna_search.py.",
    )
    parser.add_argument(
        "--model-output-path",
        type=Path,
        default=DEFAULT_MODEL_OUTPUT_PATH,
        help="Where to save the trained model artifact.",
    )
    parser.add_argument(
        "--metadata-output-path",
        type=Path,
        default=DEFAULT_METADATA_OUTPUT_PATH,
        help="Where to save final model metadata.",
    )
    parser.add_argument(
        "--split-metadata-path",
        type=Path,
        default=DEFAULT_SPLIT_METADATA_PATH,
        help="Optional preprocessing/split metadata JSON.",
    )
    parser.add_argument(
        "--preprocessor-metadata-path",
        type=Path,
        default=DEFAULT_PREPROCESSOR_METADATA_PATH,
        help="Optional saved preprocessor metadata JSON.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Override LightGBM n_jobs from the Optuna artifact.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Override LightGBM random_state from the Optuna artifact.",
    )
    return parser.parse_args()


def resolve_project_root() -> Path:
    current = Path.cwd()
    for candidate in [current, *current.parents]:
        if (candidate / "data").exists():
            return candidate
    return current


def resolve_path(project_root: Path, path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def load_json(path: Path, *, required: bool) -> dict[str, Any] | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required JSON file not found: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_aligned_data(
    features_path: Path,
    target_path: Path,
    target: str,
    index_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    features = pd.read_csv(features_path)
    target_data = pd.read_csv(target_path)

    if index_column not in features.columns:
        raise ValueError(f"Feature file does not contain index column: {index_column}")
    if index_column not in target_data.columns:
        raise ValueError(f"Target file does not contain index column: {index_column}")
    if target not in target_data.columns:
        raise ValueError(f"Target file does not contain target column: {target}")

    data = features.merge(
        target_data[[index_column, target]],
        on=index_column,
        how="inner",
        validate="one_to_one",
    )
    if len(data) != len(features) or len(data) != len(target_data):
        raise ValueError(
            "Feature and target files are not aligned by index column: "
            f"features={len(features)}, target={len(target_data)}, merged={len(data)}"
        )

    X = data.drop(columns=[index_column, target])
    y = data[target]

    non_numeric = [
        column for column in X.columns if not pd.api.types.is_numeric_dtype(X[column])
    ]
    if non_numeric:
        preview = ", ".join(non_numeric[:20])
        raise ValueError(f"LightGBM input must be numeric. Non-numeric columns: {preview}")
    if y.isna().any():
        raise ValueError("Target contains missing values.")

    return X.replace([np.inf, -np.inf], np.nan), y


def build_model_params(
    optuna_result: dict[str, Any],
    *,
    n_jobs_override: int | None,
    random_state_override: int | None,
) -> dict[str, Any]:
    params = {
        "objective": "multiclass",
        "boosting_type": "gbdt",
        "class_weight": "balanced",
        "verbosity": -1,
    }
    params.update(optuna_result.get("fixed_params", {}))
    params.update(optuna_result["best_params"])

    if n_jobs_override is not None:
        params["n_jobs"] = n_jobs_override
    if random_state_override is not None:
        params["random_state"] = random_state_override

    params.setdefault("random_state", 42)
    params.setdefault("n_jobs", -1)
    return params


def validate_optuna_labels(
    optuna_result: dict[str, Any],
    label_encoder: LabelEncoder,
) -> None:
    expected_labels = optuna_result.get("label_classes")
    current_labels = [str(label) for label in label_encoder.classes_]
    if expected_labels is not None and list(expected_labels) != current_labels:
        raise ValueError(
            "Current target labels differ from Optuna artifact labels: "
            f"current={current_labels}, optuna={expected_labels}"
        )


def save_metadata(
    metadata_output_path: Path,
    *,
    model_output_path: Path,
    params_path: Path,
    optuna_result: dict[str, Any],
    model_params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    label_encoder: LabelEncoder,
    args: argparse.Namespace,
    split_metadata: dict[str, Any] | None,
    preprocessor_metadata: dict[str, Any] | None,
) -> None:
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "lightgbm.LGBMClassifier",
        "model_output_path": str(model_output_path),
        "params_source_path": str(params_path),
        "optimized_metric": optuna_result.get("optimized_metric"),
        "optuna_best_value": optuna_result.get("best_value"),
        "optuna_best_trial_number": optuna_result.get("best_trial_number"),
        "optuna_best_trial_user_attrs": optuna_result.get("best_trial_user_attrs"),
        "model_params": model_params,
        "target": args.target,
        "index_column": args.index_column,
        "feature_names": list(X.columns),
        "feature_count": int(X.shape[1]),
        "label_classes": [str(label) for label in label_encoder.classes_],
        "priority_classes": optuna_result.get("priority_classes"),
        "score_map": (split_metadata or {}).get("score_map"),
        "thresholds": None,
        "thresholds_note": (
            "Thresholds are not recalculated in final training. Use thresholds "
            "selected on train/OOF validation artifacts before inference."
        ),
        "training_data": {
            "features_path": str(args.features_path),
            "target_path": str(args.target_path),
            "rows": int(len(X)),
            "class_counts": {
                str(label): int(count)
                for label, count in y.value_counts().sort_index().items()
            },
        },
        "preprocessing_split_metadata": split_metadata,
        "preprocessor_metadata": preprocessor_metadata,
        "dependencies": {
            "lightgbm": lightgbm.__version__,
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    metadata_output_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    project_root = resolve_project_root()

    args.features_path = resolve_path(project_root, args.features_path)
    args.target_path = resolve_path(project_root, args.target_path)
    args.params_path = resolve_path(project_root, args.params_path)
    args.model_output_path = resolve_path(project_root, args.model_output_path)
    args.metadata_output_path = resolve_path(project_root, args.metadata_output_path)
    args.split_metadata_path = resolve_path(project_root, args.split_metadata_path)
    args.preprocessor_metadata_path = resolve_path(
        project_root,
        args.preprocessor_metadata_path,
    )

    optuna_result = load_json(args.params_path, required=True)
    if optuna_result is None:
        raise RuntimeError("Unreachable: required Optuna result was not loaded.")

    X, y = load_aligned_data(
        features_path=args.features_path,
        target_path=args.target_path,
        target=args.target,
        index_column=args.index_column,
    )

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    validate_optuna_labels(optuna_result, label_encoder)

    model_params = build_model_params(
        optuna_result,
        n_jobs_override=args.n_jobs,
        random_state_override=args.random_state,
    )
    model = LGBMClassifier(**model_params)

    print("Training final LightGBM model...")
    print(f"Rows: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Params source: {args.params_path}")
    model.fit(X, y_encoded)

    args.model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_output_path)

    split_metadata = load_json(args.split_metadata_path, required=False)
    preprocessor_metadata = load_json(args.preprocessor_metadata_path, required=False)
    save_metadata(
        args.metadata_output_path,
        model_output_path=args.model_output_path,
        params_path=args.params_path,
        optuna_result=optuna_result,
        model_params=model_params,
        X=X,
        y=y,
        label_encoder=label_encoder,
        args=args,
        split_metadata=split_metadata,
        preprocessor_metadata=preprocessor_metadata,
    )

    sample_proba = model.predict_proba(X.iloc[:5])
    print("Training complete.")
    print("Model saved to:", args.model_output_path)
    print("Metadata saved to:", args.metadata_output_path)
    print("Classes:", [str(label) for label in label_encoder.classes_])
    print("First 5 probability rows shape:", sample_proba.shape)


if __name__ == "__main__":
    main()
