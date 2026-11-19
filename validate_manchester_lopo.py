"""Evaluate Manchester models with leave-one-participant-out validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_manchester_baseline import (
    GLUCOSE_FEATURES,
    PROCESSED_DIR,
    SLEEP_COLUMNS,
    SLEEP_SUMMARY_COLUMNS,
    NUTRITION_COLUMNS,
    INSULIN_BASAL_COLUMNS,
    INSULIN_BOLUS_COLUMNS,
    classification_metrics,
    has_nutrition_observations,
    make_windows,
)


def regression_metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "r2": float(r2_score(y_true, prediction))
        if y_true.nunique() > 1
        else None,
    }


def make_model_set() -> dict[str, Any]:
    from xgboost import XGBRegressor

    return {
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--positive-threshold", type=int, default=1)
    parser.add_argument("--include-sleep", action="store_true")
    parser.add_argument("--include-sleep-summary", action="store_true")
    parser.add_argument("--include-nutrition", action="store_true")
    parser.add_argument("--include-basal", action="store_true")
    parser.add_argument("--include-bolus", action="store_true")
    parser.add_argument(
        "--nutrition-observed-only",
        action="store_true",
        help="When using nutrition, keep only participants with nutrition records.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.positive_threshold not in {1, 2, 3, 4, 5}:
        raise ValueError("--positive-threshold must be between 1 and 5.")

    windows_by_participant: dict[str, pd.DataFrame] = {}
    excluded_nutrition_participants: list[str] = []
    for path in sorted(args.processed_dir.glob("participant_*.csv")):
        participant = path.stem.removeprefix("participant_")
        source = pd.read_csv(path)
        if args.include_nutrition and args.nutrition_observed_only and not has_nutrition_observations(source):
            excluded_nutrition_participants.append(participant)
            continue
        windows = make_windows(source, args.days)
        if not windows.empty:
            windows_by_participant[participant] = windows

    if len(windows_by_participant) < 2:
        raise RuntimeError("LOPO validation requires at least two participants.")

    available_columns = set().union(
        *(frame.columns for frame in windows_by_participant.values())
    )
    features = list(GLUCOSE_FEATURES)
    if args.include_sleep:
        features.extend(
            f"{column}_mean"
            for column in SLEEP_COLUMNS
            if f"{column}_mean" in available_columns
        )
    if args.include_sleep_summary:
        features.extend(
            f"{column}_mean"
            for column in SLEEP_SUMMARY_COLUMNS
            if f"{column}_mean" in available_columns
        )
    if args.include_nutrition:
        features.extend(
            f"{column}_total"
            for column in NUTRITION_COLUMNS
            if f"{column}_total" in available_columns
        )
    if args.include_basal:
        features.extend(
            f"{column}_total"
            for column in INSULIN_BASAL_COLUMNS
            if f"{column}_total" in available_columns
        )
    if args.include_bolus:
        features.extend(
            f"{column}_total"
            for column in INSULIN_BOLUS_COLUMNS
            if f"{column}_total" in available_columns
        )

    model_names = list(make_model_set())
    predictions: dict[str, list[np.ndarray]] = {
        name: [] for name in model_names
    }
    truths: list[np.ndarray] = []
    participant_results: dict[str, Any] = {}

    for held_out, test in windows_by_participant.items():
        train = pd.concat(
            [
                frame
                for participant, frame in windows_by_participant.items()
                if participant != held_out
            ],
            ignore_index=True,
        )
        test = test.copy()
        y_true = test["risk_score"]
        truths.append(y_true.to_numpy())
        participant_results[held_out] = {
            "test_windows": int(len(test)),
            "models": {},
        }

        for name, model in make_model_set().items():
            train_features = train.reindex(columns=features, fill_value=0).fillna(0)
            test_features = test.reindex(columns=features, fill_value=0).fillna(0)
            model.fit(train_features, train["risk_score"])
            prediction = np.clip(model.predict(test_features), 0, 5)
            predictions[name].append(prediction)
            labels = (y_true >= args.positive_threshold).astype(int)
            binary_prediction = (prediction >= args.positive_threshold).astype(int)
            participant_results[held_out]["models"][name] = {
                "regression": regression_metrics(y_true, prediction),
                "classification": classification_metrics(
                    labels, binary_prediction, prediction
                ),
            }

    all_truth = np.concatenate(truths)
    report: dict[str, Any] = {
        "validation": "leave-one-participant-out",
        "target": "next-window metabolic-risk proxy ordinal score (0-5)",
        "classification_target": "risk_score converted to binary for secondary metrics",
        "positive_definition": f"risk_score >= {args.positive_threshold}",
        "window_days": args.days,
        "include_sleep": args.include_sleep,
        "include_sleep_summary": args.include_sleep_summary,
        "include_nutrition": args.include_nutrition,
        "include_basal": args.include_basal,
        "include_bolus": args.include_bolus,
        "nutrition_observed_only": args.nutrition_observed_only,
        "excluded_nutrition_participants": excluded_nutrition_participants,
        "features": features,
        "participants": len(windows_by_participant),
        "test_windows": int(len(all_truth)),
        "models": {},
        "per_participant": participant_results,
    }

    for name, model_predictions in predictions.items():
        all_prediction = np.concatenate(model_predictions)
        labels = (all_truth >= args.positive_threshold).astype(int)
        binary_prediction = (all_prediction >= args.positive_threshold).astype(int)
        report["models"][name] = {
            "pooled_regression": regression_metrics(
                pd.Series(all_truth), all_prediction
            ),
            "pooled_classification": classification_metrics(
                pd.Series(labels), binary_prediction, all_prediction
            ),
            "macro_mae": float(
                np.mean(
                    [
                        participant_results[participant]["models"][name][
                            "regression"
                        ]["mae"]
                        for participant in participant_results
                    ]
                )
            ),
            "macro_rmse": float(
                np.mean(
                    [
                        participant_results[participant]["models"][name][
                            "regression"
                        ]["rmse"]
                        for participant in participant_results
                    ]
                )
            ),
        }

    output = args.output or args.processed_dir / (
        f"lopo_report_{args.days}d"
        f"{'_sleep' if args.include_sleep else '_glucose'}.json"
    )
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report written to {output}")


if __name__ == "__main__":
    main()
