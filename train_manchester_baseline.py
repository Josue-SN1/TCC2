"""Compare XGBoost and Random Forest on the Manchester risk-score proxy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROCESSED_DIR = Path(__file__).resolve().parent / "Manchester_processed"
GLUCOSE_FEATURES = [
    "glucose_mean",
    "glucose_std",
    "glucose_min",
    "glucose_max",
    "glucose_missing_fraction",
]
SLEEP_COLUMNS = [
    "sleep_step_count",
    "sleep_heart_rate",
    "sleep_stress_level_value",
]
SLEEP_SUMMARY_COLUMNS = [
    "sleep_summary_duration_s",
    "sleep_summary_deep_s",
    "sleep_summary_light_s",
    "sleep_summary_rem_s",
    "sleep_summary_awake_s",
    "sleep_summary_unmeasurable_s",
    "sleep_summary_sessions",
]
NUTRITION_COLUMNS = ["nutrition_carbs_g", "nutrition_prot_g", "nutrition_fat_g", "nutrition_fibre_g"]
INSULIN_BASAL_COLUMNS = ["basal_basal_dose"]
INSULIN_BOLUS_COLUMNS = ["bolus_bolus_dose"]


def has_nutrition_observations(frame: pd.DataFrame) -> bool:
    columns = [column for column in NUTRITION_COLUMNS if column in frame]
    return bool(columns) and frame[columns].notna().any().any()


def risk_score(values: pd.Series, missing_fraction: float) -> int:
    values = values.dropna()
    high_fraction = float((values >= 250).mean()) if not values.empty else 0.0
    maximum = float(values.max()) if not values.empty else 0.0
    score = sum(
        (high_fraction >= 0.05, high_fraction >= 0.15, maximum >= 300)
    )
    return sum(
        (score, missing_fraction >= 0.10, missing_fraction >= 0.25)
    )


def make_windows(frame: pd.DataFrame, days: int) -> pd.DataFrame:
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame.sort_values("timestamp").set_index("timestamp")
    windows = ((frame.index - frame.index.min()).total_seconds() // (days * 86400)).astype(
        int
    )
    frame["window_id"] = windows
    rows: list[dict[str, Any]] = []
    for window_id in range(int(frame["window_id"].max())):
        current = frame[frame["window_id"] == window_id]
        target = frame[frame["window_id"] == window_id + 1]
        current_glucose = current["glucose_mg_dl"].dropna()
        target_glucose = target["glucose_mg_dl"].dropna()
        if len(current_glucose) < 10 or len(target_glucose) < 10:
            continue
        rows.append(
            {
                "window_id": window_id,
                "glucose_mean": float(current_glucose.mean()),
                "glucose_std": float(current_glucose.std()),
                "glucose_min": float(current_glucose.min()),
                "glucose_max": float(current_glucose.max()),
                "glucose_missing_fraction": float(
                    current["glucose_mg_dl"].isna().mean()
                ),
                "risk_score": risk_score(
                    target_glucose,
                    float(target["glucose_mg_dl"].isna().mean()),
                ),
            }
        )
        for column in SLEEP_COLUMNS:
            if column in current:
                rows[-1][f"{column}_mean"] = float(
                    pd.to_numeric(current[column], errors="coerce").mean()
                )
        for column in SLEEP_SUMMARY_COLUMNS:
            if column in current:
                daily = pd.to_numeric(current[column], errors="coerce").groupby(
                    current.index.normalize()
                ).first()
                rows[-1][f"{column}_mean"] = float(daily.mean())
        for column in NUTRITION_COLUMNS:
            if column in current:
                total = pd.to_numeric(current[column], errors="coerce").sum(min_count=1)
                rows[-1][f"{column}_total"] = float(total) if pd.notna(total) else 0.0
        for column in INSULIN_BASAL_COLUMNS + INSULIN_BOLUS_COLUMNS:
            if column in current:
                total = pd.to_numeric(current[column], errors="coerce").sum(min_count=1)
                rows[-1][f"{column}_total"] = float(total) if pd.notna(total) else 0.0
    return pd.DataFrame(rows)


def regression_metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "r2": float(r2_score(y_true, prediction)),
    }


def classification_metrics(
    y_true: pd.Series, prediction: np.ndarray, score_prediction: np.ndarray
) -> dict[str, Any]:
    recalls = recall_score(
        y_true,
        prediction,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )
    present_classes = np.unique(y_true)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(np.mean(recalls[present_classes])),
        "precision": float(
            precision_score(y_true, prediction, labels=[0, 1], zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, prediction, labels=[0, 1], zero_division=0)
        ),
        "f1": float(f1_score(y_true, prediction, labels=[0, 1], zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
        "roc_auc": (
            float(roc_auc_score(y_true, score_prediction))
            if y_true.nunique() == 2
            else None
        ),
    }


def plot_results(
    output_dir: Path,
    regression: dict[str, dict[str, float]],
    classification: dict[str, dict[str, Any]],
    test: pd.DataFrame,
    binary_predictions: dict[str, np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    models = list(regression)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, metric in zip(axes, ["mae", "rmse", "r2"]):
        axis.bar(models, [regression[model][metric] for model in models])
        axis.set_title(metric.upper())
        axis.grid(axis="y", alpha=0.3)
    fig.suptitle("Regressão do proxy ordinal")
    fig.tight_layout()
    fig.savefig(output_dir / "regression_metrics.png", dpi=150)
    plt.close(fig)

    metric_names = ["precision", "recall", "f1", "balanced_accuracy"]
    fig, axis = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metric_names))
    width = 0.8 / len(models)
    for index, model in enumerate(models):
        axis.bar(
            x + index * width,
            [classification[model][metric] for metric in metric_names],
            width,
            label=model,
        )
    axis.set_xticks(x + width * (len(models) - 1) / 2, metric_names)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Classificação derivada do proxy")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "model_metrics.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    axes = np.atleast_1d(axes)
    for axis, model in zip(axes, models):
        ConfusionMatrixDisplay.from_predictions(
            test["risk_label"],
            binary_predictions[model],
            labels=[0, 1],
            ax=axis,
            colorbar=False,
        )
        axis.set_title(model)
    fig.suptitle("Matrizes de confusão no teste temporal")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--positive-threshold", type=int, default=1)
    parser.add_argument(
        "--include-sleep",
        action="store_true",
        help="Include aggregated sleep time-series features in the model.",
    )
    parser.add_argument(
        "--include-sleep-summary",
        action="store_true",
        help="Include daily sleep-episode summary features in the model.",
    )
    parser.add_argument(
        "--include-nutrition",
        action="store_true",
        help="Include nutrition totals in the model.",
    )
    parser.add_argument(
        "--include-basal",
        action="store_true",
        help="Include total basal insulin dose in the window.",
    )
    parser.add_argument(
        "--include-bolus",
        action="store_true",
        help="Include total bolus insulin dose in the window.",
    )
    parser.add_argument(
        "--nutrition-observed-only",
        action="store_true",
        help="When using nutrition, keep only participants with nutrition records.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not 0.5 <= args.train_fraction < 1:
        raise ValueError("--train-fraction must be at least 0.5 and below 1.")
    if args.positive_threshold not in {1, 2, 3, 4, 5}:
        raise ValueError("--positive-threshold must be between 1 and 5.")

    splits: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    excluded_nutrition_participants: list[str] = []
    for path in sorted(args.processed_dir.glob("participant_*.csv")):
        participant = path.stem.removeprefix("participant_")
        source = pd.read_csv(path)
        if args.include_nutrition and args.nutrition_observed_only and not has_nutrition_observations(source):
            excluded_nutrition_participants.append(participant)
            continue
        windows = make_windows(source, args.days)
        split = int(len(windows) * args.train_fraction)
        splits[participant] = (windows.iloc[:split], windows.iloc[split:])

    train_frames = [train for train, _ in splits.values() if not train.empty]
    test_frames = [
        test.assign(participant_id=participant)
        for participant, (_, test) in splits.items()
        if not test.empty
    ]
    if not train_frames or not test_frames:
        raise RuntimeError("No valid chronological train/test windows were generated.")
    train = pd.concat(train_frames, ignore_index=True)
    test = pd.concat(test_frames, ignore_index=True)
    train["risk_label"] = (train["risk_score"] >= args.positive_threshold).astype(int)
    test["risk_label"] = (test["risk_score"] >= args.positive_threshold).astype(int)

    from xgboost import XGBRegressor

    models = {
        "XGBoost": XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=42,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=2,
        ),
        "Ridge": make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
    }
    features = GLUCOSE_FEATURES
    if args.include_sleep:
        features += [
            f"{column}_mean" for column in SLEEP_COLUMNS if f"{column}_mean" in train
        ]
    if args.include_sleep_summary:
        features += [
            f"{column}_mean"
            for column in SLEEP_SUMMARY_COLUMNS
            if f"{column}_mean" in train
        ]
    if args.include_nutrition:
        features += [
            f"{column}_total"
            for column in NUTRITION_COLUMNS
            if f"{column}_total" in train
        ]
    if args.include_basal:
        features += [
            f"{column}_total"
            for column in INSULIN_BASAL_COLUMNS
            if f"{column}_total" in train
        ]
    if args.include_bolus:
        features += [
            f"{column}_total"
            for column in INSULIN_BOLUS_COLUMNS
            if f"{column}_total" in train
        ]
    predictions: dict[str, np.ndarray] = {}
    binary_predictions: dict[str, np.ndarray] = {}
    regression: dict[str, dict[str, float]] = {}
    classification: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        model.fit(train[features].fillna(0), train["risk_score"])
        score_prediction = np.clip(
            model.predict(test[features].fillna(0)), 0, 5
        )
        predictions[name] = score_prediction
        binary_predictions[name] = (
            score_prediction >= args.positive_threshold
        ).astype(int)
        regression[name] = regression_metrics(test["risk_score"], score_prediction)
        classification[name] = classification_metrics(
            test["risk_label"], binary_predictions[name], score_prediction
        )

    output = args.output or args.processed_dir / f"baseline_report_{args.days}d.json"
    report: dict[str, Any] = {
        "target": "next-window metabolic-risk proxy ordinal score (0-5)",
        "classification_target": "risk_score converted to binary for secondary metrics",
        "positive_definition": f"risk_score >= {args.positive_threshold}",
        "models": ["XGBoost", "Random Forest", "Ridge"],
        "features": features,
        "include_sleep": args.include_sleep,
        "include_sleep_summary": args.include_sleep_summary,
        "include_nutrition": args.include_nutrition,
        "include_basal": args.include_basal,
        "include_bolus": args.include_bolus,
        "nutrition_observed_only": args.nutrition_observed_only,
        "excluded_nutrition_participants": excluded_nutrition_participants,
        "window_days": args.days,
        "split": f"chronological {args.train_fraction:.0%}/{1 - args.train_fraction:.0%} per participant",
        "train_windows": int(len(train)),
        "test_windows": int(len(test)),
        "train_score_distribution": train["risk_score"].value_counts().sort_index().astype(int).to_dict(),
        "test_score_distribution": test["risk_score"].value_counts().sort_index().astype(int).to_dict(),
        "train_class_distribution": train["risk_label"].value_counts().sort_index().astype(int).to_dict(),
        "test_class_distribution": test["risk_label"].value_counts().sort_index().astype(int).to_dict(),
        "regression": regression,
        "classification": classification,
        "participants": {},
        "plots": [
            str(output.parent / "regression_metrics.png"),
            str(output.parent / "model_metrics.png"),
            str(output.parent / "confusion_matrices.png"),
        ],
    }
    for participant, (participant_train, participant_test) in splits.items():
        result: dict[str, Any] = {
            "train_windows": int(len(participant_train)),
            "test_windows": int(len(participant_test)),
        }
        if participant_test.empty:
            result["status"] = "not_evaluable"
        else:
            result["status"] = "evaluated"
            mask = test["participant_id"] == participant
            participant_labels = (
                participant_test["risk_score"] >= args.positive_threshold
            ).astype(int)
            result["regression"] = {
                name: regression_metrics(
                    participant_test["risk_score"], predictions[name][mask.to_numpy()]
                )
                for name in models
            }
            result["classification"] = {
                name: classification_metrics(
                    participant_labels,
                    binary_predictions[name][mask.to_numpy()],
                    predictions[name][mask.to_numpy()],
                )
                for name in models
            }
        report["participants"][participant] = result

    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_results(output.parent, regression, classification, test, binary_predictions)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report written to {output}")


if __name__ == "__main__":
    main()
