from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


DEFAULT_MODEL_PATH = Path("model") / "model.joblib"
DEFAULT_METADATA_PATH = Path("model") / "model_metadata.json"
DEFAULT_OUTPUT_DIR = Path("Metrics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build LightGBM feature selection artifacts from trained model "
            "gain/split importances."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained LightGBM joblib artifact.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path to final model metadata JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where feature selection reports will be written.",
    )
    parser.add_argument(
        "--cumulative-gain-threshold",
        type=float,
        default=0.95,
        help="Select the smallest feature set that reaches this cumulative gain share.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top features to show in the Markdown report.",
    )
    return parser.parse_args()


def resolve_project_root() -> Path:
    current = Path.cwd()
    for candidate in [current, *current.parents]:
        if (candidate / "model").exists() and (candidate / "Metrics").exists():
            return candidate
    return current


def resolve_path(project_root: Path, path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_booster(model: Any) -> Any:
    booster = getattr(model, "booster_", None)
    if booster is None:
        booster = getattr(model, "booster", None)
    if callable(booster):
        booster = booster()
    if booster is None:
        raise TypeError("Loaded model does not expose a LightGBM booster.")
    return booster


def feature_family(feature_name: str) -> str:
    if feature_name.startswith("Loan_Type__"):
        return "loan_type_ohe"
    if feature_name.startswith("Month_"):
        return "month_ohe"
    if feature_name.startswith("Occupation_"):
        return "occupation_ohe"
    if feature_name.startswith("Credit_Mix_"):
        return "credit_mix_ohe"
    if feature_name.startswith("Payment_of_Min_Amount_"):
        return "payment_min_amount_ohe"
    if feature_name.startswith("Payment_Behaviour_"):
        return "payment_behaviour_ohe"
    if feature_name.startswith("Payment_Spend_Level_"):
        return "payment_spend_level_ohe"
    if feature_name.startswith("Payment_Value_Size_"):
        return "payment_value_size_ohe"
    if feature_name.startswith("is_missing_") or feature_name == "missing_marker_count":
        return "missing_indicators"
    if feature_name.startswith("is_anomaly_") or feature_name == "anomaly_count":
        return "anomaly_indicators"

    engineered_tokens = (
        "_to_",
        "_per_",
        "_ratio",
        "pressure",
        "available_income",
        "free_cash_flow",
        "history_year",
        "diversity",
        "ordinal",
        "flag",
    )
    if any(token in feature_name for token in engineered_tokens):
        return "engineered_numeric"
    return "base_numeric"


def build_importance_frame(
    model: Any,
    metadata: dict[str, Any],
    cumulative_gain_threshold: float,
) -> pd.DataFrame:
    booster = get_booster(model)
    booster_feature_names = list(booster.feature_name())
    metadata_feature_names = metadata.get("feature_names") or []
    feature_names = metadata_feature_names or booster_feature_names

    if len(feature_names) != len(booster_feature_names):
        raise ValueError(
            "Feature count in model metadata differs from LightGBM booster feature count: "
            f"metadata={len(feature_names)}, booster={len(booster_feature_names)}"
        )

    gain_importance = np.asarray(
        booster.feature_importance(importance_type="gain"),
        dtype=float,
    )
    split_importance = np.asarray(
        booster.feature_importance(importance_type="split"),
        dtype=float,
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "lightgbm_feature_name": booster_feature_names,
            "feature_family": [feature_family(feature) for feature in feature_names],
            "gain_importance": gain_importance,
            "split_importance": split_importance,
        }
    )
    total_gain = float(importance_df["gain_importance"].sum())
    total_split = float(importance_df["split_importance"].sum())

    if total_gain <= 0:
        raise ValueError("Total gain importance is zero; feature selection is not possible.")

    importance_df["gain_share"] = importance_df["gain_importance"] / total_gain
    importance_df["split_share"] = np.where(
        total_split > 0,
        importance_df["split_importance"] / total_split,
        0.0,
    )
    importance_df = importance_df.sort_values(
        by=["gain_importance", "split_importance", "feature"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    importance_df["rank_by_gain"] = np.arange(1, len(importance_df) + 1)
    importance_df["cumulative_gain_share"] = importance_df["gain_share"].cumsum()

    selected_count = int(
        np.searchsorted(
            importance_df["cumulative_gain_share"].to_numpy(),
            cumulative_gain_threshold,
            side="left",
        )
        + 1
    )
    selected_count = min(selected_count, len(importance_df))
    importance_df["selected_by_cumulative_gain"] = (
        importance_df["rank_by_gain"] <= selected_count
    )
    return importance_df


def build_family_summary(importance_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        importance_df.groupby("feature_family", as_index=False)
        .agg(
            feature_count=("feature", "count"),
            selected_count=("selected_by_cumulative_gain", "sum"),
            gain_importance=("gain_importance", "sum"),
            gain_share=("gain_share", "sum"),
            split_importance=("split_importance", "sum"),
            split_share=("split_share", "sum"),
        )
        .sort_values(
            by=["gain_importance", "split_importance", "feature_family"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
    summary["selected_count"] = summary["selected_count"].astype(int)
    return summary


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame.loc[:, columns].copy()

    def format_cell(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return f"{value:.6f}"
        if isinstance(value, np.floating):
            return f"{float(value):.6f}"
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(format_cell(value) for value in row) + " |"
        for row in view.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def write_markdown_report(
    report_path: Path,
    *,
    importance_df: pd.DataFrame,
    family_summary: pd.DataFrame,
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    selected_df = importance_df[importance_df["selected_by_cumulative_gain"]]
    top_features = importance_df.head(args.top_n)

    lines = [
        "# LightGBM feature selection",
        "",
        f"Создано UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Контекст",
        "",
        f"- Модель: `{args.model_path}`",
        f"- Metadata: `{args.metadata_path}`",
        f"- Метрика Optuna: `{metadata.get('optimized_metric')}`",
        f"- Лучшее значение Optuna: `{metadata.get('optuna_best_value')}`",
        f"- Порог cumulative gain: `{args.cumulative_gain_threshold}`",
        f"- Всего признаков: `{len(importance_df)}`",
        f"- Выбрано признаков: `{len(selected_df)}`",
        f"- Покрытие gain выбранными признаками: `{selected_df['gain_share'].sum():.6f}`",
        "",
        "## Как читать отчёт",
        "",
        "`gain_importance` показывает вклад признака в снижение ошибки в деревьях LightGBM. "
        "`split_importance` показывает, сколько раз признак использовался в разбиениях. "
        "Для feature selection используется минимальный набор признаков, который покрывает "
        "заданную долю cumulative gain.",
        "",
        "Важно: это model-based selection, а не доказательство причинности. "
        "Коррелирующие признаки могут делить важность между собой.",
        "",
        f"## Top-{args.top_n} признаков по gain",
        "",
        markdown_table(
            top_features,
            [
                "rank_by_gain",
                "feature",
                "feature_family",
                "gain_share",
                "cumulative_gain_share",
                "split_share",
                "selected_by_cumulative_gain",
            ],
        ),
        "",
        "## Важность по группам признаков",
        "",
        markdown_table(
            family_summary,
            [
                "feature_family",
                "feature_count",
                "selected_count",
                "gain_share",
                "split_share",
            ],
        ),
        "",
        "## Файлы",
        "",
        "- `lightgbm_feature_importance.csv` — все признаки с gain/split importance.",
        "- `lightgbm_selected_features.csv` — выбранный набор признаков.",
        "- `lightgbm_feature_family_importance.csv` — важность по группам признаков.",
        "- `lightgbm_feature_selection_summary.json` — машинно-читаемое резюме.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    output_dir: Path,
    *,
    importance_df: pd.DataFrame,
    family_summary: pd.DataFrame,
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    importance_path = output_dir / "lightgbm_feature_importance.csv"
    selected_path = output_dir / "lightgbm_selected_features.csv"
    family_path = output_dir / "lightgbm_feature_family_importance.csv"
    summary_path = output_dir / "lightgbm_feature_selection_summary.json"
    report_path = output_dir / "lightgbm_feature_selection_report.md"

    selected_df = importance_df[importance_df["selected_by_cumulative_gain"]]
    importance_df.to_csv(importance_path, index=False)
    selected_df.to_csv(selected_path, index=False)
    family_summary.to_csv(family_path, index=False)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(args.model_path),
        "metadata_path": str(args.metadata_path),
        "importance_source": "lightgbm_booster_gain_and_split",
        "selection_rule": "smallest_feature_set_reaching_cumulative_gain_threshold",
        "cumulative_gain_threshold": args.cumulative_gain_threshold,
        "feature_count": int(len(importance_df)),
        "selected_feature_count": int(len(selected_df)),
        "selected_gain_share": float(selected_df["gain_share"].sum()),
        "optimized_metric": metadata.get("optimized_metric"),
        "optuna_best_value": metadata.get("optuna_best_value"),
        "top_20_features": importance_df.head(20)["feature"].tolist(),
        "selected_features": selected_df["feature"].tolist(),
        "output_files": {
            "feature_importance": str(importance_path),
            "selected_features": str(selected_path),
            "feature_family_importance": str(family_path),
            "report": str(report_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_markdown_report(
        report_path,
        importance_df=importance_df,
        family_summary=family_summary,
        metadata=metadata,
        args=args,
    )

    print("Feature selection artifacts saved:")
    print("Feature importance:", importance_path)
    print("Selected features:", selected_path)
    print("Family importance:", family_path)
    print("Summary:", summary_path)
    print("Report:", report_path)
    print("Selected features:", len(selected_df), "of", len(importance_df))
    print("Selected gain share:", round(float(selected_df["gain_share"].sum()), 6))


def main() -> None:
    args = parse_args()
    if not 0 < args.cumulative_gain_threshold <= 1:
        raise ValueError("--cumulative-gain-threshold must be in the interval (0, 1].")

    project_root = resolve_project_root()
    args.model_path = resolve_path(project_root, args.model_path)
    args.metadata_path = resolve_path(project_root, args.metadata_path)
    args.output_dir = resolve_path(project_root, args.output_dir)

    model = joblib.load(args.model_path)
    metadata = load_metadata(args.metadata_path)
    importance_df = build_importance_frame(
        model,
        metadata,
        cumulative_gain_threshold=args.cumulative_gain_threshold,
    )
    family_summary = build_family_summary(importance_df)

    write_outputs(
        args.output_dir,
        importance_df=importance_df,
        family_summary=family_summary,
        metadata=metadata,
        args=args,
    )


if __name__ == "__main__":
    main()
