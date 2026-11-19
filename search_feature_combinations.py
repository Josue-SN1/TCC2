"""Search modality combinations with participant-isolated validation."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_manchester_baseline import (
    GLUCOSE_FEATURES,
    NUTRITION_COLUMNS,
    PROCESSED_DIR,
    SLEEP_COLUMNS,
    SLEEP_SUMMARY_COLUMNS,
    classification_metrics,
    has_nutrition_observations,
    make_windows,
)
from validate_manchester_lopo import make_model_set, regression_metrics


GROUP_NAMES = ("sleep_ts", "sleep_summary", "nutrition")


def feature_groups(available_columns: set[str]) -> dict[str, list[str]]:
    groups = {
        "glucose": list(GLUCOSE_FEATURES),
        "sleep_ts": [
            f"{column}_mean"
            for column in SLEEP_COLUMNS
            if f"{column}_mean" in available_columns
        ],
        "sleep_summary": [
            f"{column}_mean"
            for column in SLEEP_SUMMARY_COLUMNS
            if f"{column}_mean" in available_columns
        ],
        "nutrition": [
            f"{column}_total"
            for column in NUTRITION_COLUMNS
            if f"{column}_total" in available_columns
        ],
    }
    return groups


def combinations() -> list[tuple[str, ...]]:
    return [
        ("glucose",) + optional
        for size in range(len(GROUP_NAMES) + 1)
        for optional in itertools.combinations(GROUP_NAMES, size)
    ]


def evaluate(
    windows_by_participant: dict[str, pd.DataFrame],
    groups: dict[str, list[str]],
    selected_groups: tuple[str, ...],
    positive_threshold: int,
) -> dict[str, Any]:
    features = [
        feature
        for group in selected_groups
        for feature in groups[group]
    ]
    predictions: dict[str, list[np.ndarray]] = {
        name: [] for name in make_model_set()
    }
    truths: list[np.ndarray] = []
    participant_metrics: dict[str, dict[str, Any]] = {}

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
        participant_metrics[held_out] = {}
        train_features = train.reindex(columns=features, fill_value=0).fillna(0)
        test_features = test.reindex(columns=features, fill_value=0).fillna(0)

        for name, model in make_model_set().items():
            model.fit(train_features, train["risk_score"])
            prediction = np.clip(model.predict(test_features), 0, 5)
            predictions[name].append(prediction)
            labels = (y_true >= positive_threshold).astype(int)
            participant_metrics[held_out][name] = {
                "regression": regression_metrics(y_true, prediction),
                "classification": classification_metrics(
                    labels,
                    (prediction >= positive_threshold).astype(int),
                    prediction,
                ),
            }

    all_truth = np.concatenate(truths)
    result: dict[str, Any] = {
        "groups": list(selected_groups),
        "features": features,
        "participants": len(windows_by_participant),
        "test_windows": int(len(all_truth)),
        "models": {},
    }
    labels = (all_truth >= positive_threshold).astype(int)
    for name, model_predictions in predictions.items():
        all_prediction = np.concatenate(model_predictions)
        result["models"][name] = {
            "pooled_regression": regression_metrics(
                pd.Series(all_truth), all_prediction
            ),
            "pooled_classification": classification_metrics(
                pd.Series(labels),
                (all_prediction >= positive_threshold).astype(int),
                all_prediction,
            ),
            "macro_mae": float(
                np.mean(
                    [
                        participant_metrics[participant][name]["regression"]["mae"]
                        for participant in participant_metrics
                    ]
                )
            ),
            "macro_rmse": float(
                np.mean(
                    [
                        participant_metrics[participant][name]["regression"]["rmse"]
                        for participant in participant_metrics
                    ]
                )
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--positive-threshold", type=int, default=1)
    parser.add_argument("--nutrition-observed-only", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.positive_threshold not in {1, 2, 3, 4, 5}:
        raise ValueError("--positive-threshold must be between 1 and 5.")

    windows_by_participant: dict[str, pd.DataFrame] = {}
    excluded: list[str] = []
    for path in sorted(args.processed_dir.glob("participant_*.csv")):
        participant = path.stem.removeprefix("participant_")
        source = pd.read_csv(path)
        if (
            args.nutrition_observed_only
            and not has_nutrition_observations(source)
        ):
            excluded.append(participant)
            continue
        windows = make_windows(source, args.days)
        if not windows.empty:
            windows_by_participant[participant] = windows
    if len(windows_by_participant) < 2:
        raise RuntimeError("LOPO search requires at least two participants.")

    available_columns = set().union(
        *(frame.columns for frame in windows_by_participant.values())
    )
    groups = feature_groups(available_columns)
    results = [
        evaluate(windows_by_participant, groups, selected, args.positive_threshold)
        for selected in combinations()
    ]
    rows = []
    for result in results:
        for model, metrics in result["models"].items():
            rows.append(
                {
                    "configuration": "+".join(result["groups"]),
                    "model": model,
                    "mae": metrics["pooled_regression"]["mae"],
                    "rmse": metrics["pooled_regression"]["rmse"],
                    "r2": metrics["pooled_regression"]["r2"],
                    "macro_mae": metrics["macro_mae"],
                    "macro_rmse": metrics["macro_rmse"],
                }
            )
    ranking = pd.DataFrame(rows).sort_values(
        ["mae", "rmse"], ascending=[True, True]
    )
    output = args.output or args.processed_dir / (
        f"feature_combination_lopo_{args.days}d.json"
    )
    report = {
        "validation": "leave-one-participant-out",
        "search": "all modality-group combinations with glucose mandatory",
        "window_days": args.days,
        "positive_definition": f"risk_score >= {args.positive_threshold}",
        "nutrition_observed_only": args.nutrition_observed_only,
        "excluded_nutrition_participants": excluded,
        "groups": groups,
        "results": results,
        "ranking_by_pooled_mae": ranking.to_dict(orient="records"),
    }
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ranking.to_csv(output.with_suffix(".csv"), index=False)
    print(ranking.to_string(index=False))
    print(f"Report written to {output}")


if __name__ == "__main__":
    main()
