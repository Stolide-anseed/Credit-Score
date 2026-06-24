from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

try:
    import optuna
except ImportError:  # pragma: no cover - handled at runtime with a clear message.
    optuna = None


RANDOM_STATE = 42
DEFAULT_PRIORITY_CLASSES = ("1", "2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search LightGBM hyperparameters with Optuna using StratifiedKFold "
            "and mean Average Precision for priority classes."
        )
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("data") / "X_train_prepared.csv",
        help="CSV with prepared numeric features.",
    )
    parser.add_argument(
        "--target-path",
        type=Path,
        default=Path("data") / "y_train.csv",
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
        "--priority-classes",
        default=",".join(DEFAULT_PRIORITY_CLASSES),
        help="Comma-separated target labels whose Average Precision is optimized.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of StratifiedKFold splits.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional Optuna timeout in seconds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help="Random seed.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="LightGBM n_jobs.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("Metrics") / "lightgbm_optuna_best_params.json",
        help="Where to save best params and search summary.",
    )
    return parser.parse_args()


def resolve_project_root() -> Path:
    current = Path.cwd()
    for candidate in [current, *current.parents]:
        if (candidate / "data").exists():
            return candidate
    return current


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
    return X, y


def parse_priority_classes(raw_value: str) -> list[str]:
    priority_classes = [value.strip() for value in raw_value.split(",") if value.strip()]
    if not priority_classes:
        raise ValueError("At least one priority class must be provided.")
    return priority_classes


def priority_class_indices(
    label_encoder: LabelEncoder,
    priority_classes: Sequence[str],
) -> list[int]:
    indices = []
    known_labels = [str(label) for label in label_encoder.classes_]
    for priority_class in priority_classes:
        for class_index, label in enumerate(label_encoder.classes_):
            if priority_class == str(label):
                indices.append(class_index)
                break
        else:
            raise ValueError(
                f"Priority class {priority_class!r} not found in target labels: {known_labels}"
            )
    return indices


def validate_class_counts(y_encoded: np.ndarray, n_splits: int) -> None:
    class_counts = pd.Series(y_encoded).value_counts().sort_index()
    too_small = class_counts[class_counts < n_splits]
    if not too_small.empty:
        raise ValueError(
            "Each class must have at least n_splits rows for StratifiedKFold. "
            f"Too small classes: {too_small.to_dict()}"
        )


def suggest_lgbm_params(
    trial: Any,
    seed: int,
    n_jobs: int,
) -> dict[str, Any]:
    max_depth = trial.suggest_int("max_depth", 4, 12)
    max_num_leaves = min(256, 2**max_depth)

    return {
        "objective": "multiclass",
        "boosting_type": "gbdt",
        "class_weight": "balanced",
        "n_estimators": trial.suggest_int("n_estimators", 250, 1200, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, max_num_leaves, log=True),
        "max_depth": max_depth,
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "random_state": seed,
        "n_jobs": n_jobs,
        "verbosity": -1,
    }


def aligned_predict_proba(
    model: LGBMClassifier,
    X_valid: pd.DataFrame,
    n_classes: int,
) -> np.ndarray:
    raw_proba = model.predict_proba(X_valid)
    aligned_proba = np.zeros((len(X_valid), n_classes), dtype=float)
    for probability_column, class_index in enumerate(model.classes_):
        aligned_proba[:, int(class_index)] = raw_proba[:, probability_column]
    return aligned_proba


def priority_average_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
    priority_indices: Sequence[int],
) -> tuple[float, dict[str, float]]:
    per_class_scores = {}
    for class_index in priority_indices:
        y_binary = (y_true == class_index).astype(int)
        score = average_precision_score(y_binary, y_score[:, class_index])
        per_class_scores[str(class_index)] = float(score)
    return float(np.mean(list(per_class_scores.values()))), per_class_scores


def make_objective(
    X: pd.DataFrame,
    y_encoded: np.ndarray,
    priority_indices: Sequence[int],
    n_splits: int,
    seed: int,
    n_jobs: int,
):
    n_classes = len(np.unique(y_encoded))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def objective(trial: Any) -> float:
        params = suggest_lgbm_params(trial, seed=seed, n_jobs=n_jobs)
        y_score_oof = np.zeros((len(y_encoded), n_classes), dtype=float)

        for fold_index, (train_idx, valid_idx) in enumerate(cv.split(X, y_encoded)):
            model = LGBMClassifier(**params)
            model.fit(X.iloc[train_idx], y_encoded[train_idx])
            y_score_oof[valid_idx] = aligned_predict_proba(
                model,
                X.iloc[valid_idx],
                n_classes=n_classes,
            )

            fold_score, _ = priority_average_precision(
                y_encoded[valid_idx],
                y_score_oof[valid_idx],
                priority_indices,
            )
            trial.report(fold_score, step=fold_index)
            if trial.should_prune():
                raise optuna.TrialPruned()

        score, per_class_scores = priority_average_precision(
            y_encoded,
            y_score_oof,
            priority_indices,
        )
        for class_index, class_score in per_class_scores.items():
            trial.set_user_attr(f"average_precision_class_{class_index}", class_score)
        trial.set_user_attr("average_precision_priority_mean", score)
        return score

    return objective


def save_result(
    output_path: Path,
    study: Any,
    args: argparse.Namespace,
    X: pd.DataFrame,
    y: pd.Series,
    label_encoder: LabelEncoder,
    priority_indices: Sequence[int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_trial = study.best_trial
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "lightgbm.LGBMClassifier",
        "optimized_metric": "mean_average_precision_priority_classes",
        "best_value": float(study.best_value),
        "best_params": best_trial.params,
        "fixed_params": {
            "objective": "multiclass",
            "boosting_type": "gbdt",
            "class_weight": "balanced",
            "random_state": args.seed,
            "n_jobs": args.n_jobs,
            "verbosity": -1,
        },
        "best_trial_number": best_trial.number,
        "best_trial_user_attrs": best_trial.user_attrs,
        "priority_classes": [
            str(label_encoder.classes_[class_index]) for class_index in priority_indices
        ],
        "label_classes": [str(label) for label in label_encoder.classes_],
        "n_trials": len(study.trials),
        "n_splits": args.n_splits,
        "data": {
            "features_path": str(args.features_path),
            "target_path": str(args.target_path),
            "target": args.target,
            "rows": int(len(X)),
            "features": int(X.shape[1]),
            "class_counts": {str(label): int(count) for label, count in y.value_counts().sort_index().items()},
        },
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    if optuna is None:
        raise SystemExit(
            "Optuna is not installed. Install dependencies first, for example: "
            "pip install -r requrements.txt"
        )

    project_root = resolve_project_root()
    args.features_path = (project_root / args.features_path).resolve()
    args.target_path = (project_root / args.target_path).resolve()
    args.output_path = (project_root / args.output_path).resolve()

    X, y = load_aligned_data(
        features_path=args.features_path,
        target_path=args.target_path,
        target=args.target,
        index_column=args.index_column,
    )

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    validate_class_counts(y_encoded, args.n_splits)

    priority_indices = priority_class_indices(
        label_encoder,
        parse_priority_classes(args.priority_classes),
    )

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="lightgbm_average_precision_search",
    )
    study.optimize(
        make_objective(
            X=X,
            y_encoded=y_encoded,
            priority_indices=priority_indices,
            n_splits=args.n_splits,
            seed=args.seed,
            n_jobs=args.n_jobs,
        ),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    save_result(args.output_path, study, args, X, y, label_encoder, priority_indices)

    print("Best value:", round(float(study.best_value), 6))
    print("Best trial:", study.best_trial.number)
    print("Best params:")
    print(json.dumps(study.best_trial.params, ensure_ascii=False, indent=2))
    print("Per-priority-class Average Precision:")
    print(json.dumps(study.best_trial.user_attrs, ensure_ascii=False, indent=2))
    print("Saved to:", args.output_path)


if __name__ == "__main__":
    main()
