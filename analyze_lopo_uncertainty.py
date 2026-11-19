"""Analyze LOPO uncertainty and errors for the fixed Manchester baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from train_manchester_baseline import (
    GLUCOSE_FEATURES,
    PROCESSED_DIR,
    make_windows,
)


MODEL_NAMES = ("XGBoost", "Random Forest", "Ridge")


def make_models() -> dict[str, Any]:
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
            min_samples_leaf=2,
            random_state=42,
        ),
        "Ridge": make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
    }


def regression_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    result = {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
    }
    result["r2"] = (
        float(r2_score(truth, prediction)) if np.unique(truth).size > 1 else None
    )
    return result


def bootstrap_ci(
    samples: list[float],
    seed: int,
) -> dict[str, float]:
    values = np.asarray(samples, dtype=float)
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(2000, len(values)), replace=True)
    boot_means = boot.mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "median": float(np.median(boot_means)),
        "ci95_low": float(np.percentile(boot_means, 2.5)),
        "ci95_high": float(np.percentile(boot_means, 97.5)),
    }


def interval(values: np.ndarray) -> tuple[float, float]:
    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def markdown_table(
    headers: list[str],
    rows: list[list[Any]],
) -> str:
    def format_cell(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    content = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    content.extend(
        "| " + " | ".join(format_cell(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(content)


def save_uncertainty_figure(
    report: dict[str, Any],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)

    for axis_index, metric in enumerate(("mae", "r2")):
        axis = axes[axis_index]
        positions = np.arange(len(MODEL_NAMES))
        estimates = [
            report["model_uncertainty"][model]["pooled"][metric]["estimate"]
            for model in MODEL_NAMES
        ]
        lows = [
            report["model_uncertainty"][model]["pooled"][metric]["ci95_low"]
            for model in MODEL_NAMES
        ]
        highs = [
            report["model_uncertainty"][model]["pooled"][metric]["ci95_high"]
            for model in MODEL_NAMES
        ]
        errors = np.array([
            np.asarray(estimates) - np.asarray(lows),
            np.asarray(highs) - np.asarray(estimates),
        ])
        axis.errorbar(
            positions,
            estimates,
            yerr=errors,
            fmt="o",
            capsize=5,
            linewidth=1.5,
        )
        axis.set_xticks(positions, MODEL_NAMES)
        axis.set_title(metric.upper())
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)

    axes[0].set_ylabel("MAE — menor é melhor")
    axes[1].set_ylabel("R² — maior é melhor")
    figure.suptitle("Incerteza LOPO de 7 dias — bootstrap por participante")
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_error_figure(
    report: dict[str, Any],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.0), constrained_layout=True)

    participant_rows = report["participant_error"]
    participants = [row["participant"] for row in participant_rows]
    x = np.arange(len(participants))
    width = 0.24
    for index, model in enumerate(MODEL_NAMES):
        values = [row["models"][model]["mae"] for row in participant_rows]
        axes[0].bar(
            x + (index - 1) * width,
            values,
            width,
            label=model,
        )
    axes[0].set_xticks(x, participants, rotation=45)
    axes[0].set_ylabel("MAE")
    axes[0].set_title("MAE por participante")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_axisbelow(True)
    axes[0].legend(fontsize=8)

    classes = [str(value) for value in range(6)]
    x = np.arange(len(classes))
    for index, model in enumerate(MODEL_NAMES):
        values = [
            report["error_by_true_class"][str_class]["models"][model]["mae"]
            for str_class in classes
            if str_class in report["error_by_true_class"]
        ]
        axes[1].bar(
            x[: len(values)] + (index - 1) * width,
            values,
            width,
            label=model,
        )
    axes[1].set_xticks(x[: len(values)], classes[: len(values)])
    axes[1].set_xlabel("Classe verdadeira do risk_score")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("MAE por classe verdadeira")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_axisbelow(True)

    figure.suptitle("Análise de erros no LOPO de 7 dias")
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "lopo_uncertainty_7d.json",
    )
    args = parser.parse_args()

    participant_truth: dict[str, np.ndarray] = {}
    participant_prediction: dict[str, dict[str, np.ndarray]] = {}
    participant_models: dict[str, dict[str, Any]] = {}

    windows_by_participant: dict[str, pd.DataFrame] = {}
    for path in sorted(args.processed_dir.glob("participant_*.csv")):
        participant = path.stem.removeprefix("participant_")
        frame = pd.read_csv(path)
        windows = make_windows(frame, args.days)
        if not windows.empty:
            windows_by_participant[participant] = windows

    if len(windows_by_participant) < 2:
        raise RuntimeError("LOPO uncertainty requires at least two participants.")

    for held_out, test in windows_by_participant.items():
        train = pd.concat(
            [
                frame
                for participant, frame in windows_by_participant.items()
                if participant != held_out
            ],
            ignore_index=True,
        )
        y_true = test["risk_score"].to_numpy(dtype=float)
        participant_truth[held_out] = y_true
        participant_prediction[held_out] = {}
        participant_models[held_out] = {}

        for model_name, model in make_models().items():
            train_features = train.reindex(
                columns=GLUCOSE_FEATURES,
                fill_value=0,
            ).fillna(0)
            test_features = test.reindex(
                columns=GLUCOSE_FEATURES,
                fill_value=0,
            ).fillna(0)
            model.fit(train_features, train["risk_score"])
            prediction = np.clip(model.predict(test_features), 0, 5)
            participant_prediction[held_out][model_name] = prediction
            participant_models[held_out][model_name] = regression_metrics(
                y_true,
                prediction,
            )

    participants = sorted(participant_truth)
    participant_error: list[dict[str, Any]] = []
    for participant in participants:
        truth = participant_truth[participant]
        row: dict[str, Any] = {
            "participant": participant,
            "test_windows": int(len(truth)),
            "models": {},
        }
        for model_name in MODEL_NAMES:
            prediction = participant_prediction[participant][model_name]
            metrics = regression_metrics(truth, prediction)
            true_binary = (truth >= 1).astype(int)
            pred_binary = (prediction >= 1).astype(int)
            row["models"][model_name] = {
                **metrics,
                "precision": float(
                    precision_score(
                        true_binary,
                        pred_binary,
                        zero_division=0,
                    )
                ),
                "recall": float(
                    recall_score(
                        true_binary,
                        pred_binary,
                        zero_division=0,
                    )
                ),
            }
        participant_error.append(row)

    error_by_true_class: dict[str, dict[str, Any]] = {}
    for risk_class in range(6):
        str_class = str(risk_class)
        row: dict[str, Any] = {"n": 0, "models": {}}
        for model_name in MODEL_NAMES:
            selected_truth: list[np.ndarray] = []
            selected_prediction: list[np.ndarray] = []
            for participant in participants:
                truth = participant_truth[participant]
                prediction = participant_prediction[participant][model_name]
                mask = truth == risk_class
                if mask.any():
                    selected_truth.append(truth[mask])
                    selected_prediction.append(prediction[mask])
            if selected_truth:
                truth_values = np.concatenate(selected_truth)
                prediction_values = np.concatenate(selected_prediction)
                row["n"] = int(len(truth_values))
                row["models"][model_name] = {
                    "n": int(len(truth_values)),
                    "mae": float(
                        mean_absolute_error(truth_values, prediction_values)
                    ),
                    "rmse": float(
                        np.sqrt(
                            mean_squared_error(truth_values, prediction_values)
                        )
                    ),
                    "bias": float(
                        np.mean(prediction_values - truth_values)
                    ),
                    "prediction_mean": float(np.mean(prediction_values)),
                }
        if row["n"] > 0:
            error_by_true_class[str_class] = row

    rng = np.random.default_rng(args.seed)
    bootstrap_pooled: dict[str, dict[str, list[float]]] = {
        model_name: {"mae": [], "rmse": [], "r2": []}
        for model_name in MODEL_NAMES
    }
    bootstrap_macro_mae: dict[str, list[float]] = {
        model_name: [] for model_name in MODEL_NAMES
    }
    bootstrap_differences: dict[str, list[float]] = {
        "ridge_minus_xgboost_mae": [],
        "ridge_minus_rf_mae": [],
    }

    for _ in range(args.bootstrap_samples):
        selected = rng.choice(
            participants,
            size=len(participants),
            replace=True,
        )
        for model_name in MODEL_NAMES:
            truth_parts = [
                participant_truth[participant]
                for participant in selected
            ]
            prediction_parts = [
                participant_prediction[participant][model_name]
                for participant in selected
            ]
            truth_values = np.concatenate(truth_parts)
            prediction_values = np.concatenate(prediction_parts)
            metrics = regression_metrics(truth_values, prediction_values)
            bootstrap_pooled[model_name]["mae"].append(metrics["mae"])
            bootstrap_pooled[model_name]["rmse"].append(metrics["rmse"])
            if metrics["r2"] is not None:
                bootstrap_pooled[model_name]["r2"].append(metrics["r2"])

            macro_maes = [
                participant_models[participant][model_name]["mae"]
                for participant in selected
            ]
            bootstrap_macro_mae[model_name].append(float(np.mean(macro_maes)))

        ridge_mae = bootstrap_pooled["Ridge"]["mae"][-1]
        xgboost_mae = bootstrap_pooled["XGBoost"]["mae"][-1]
        random_forest_mae = bootstrap_pooled["Random Forest"]["mae"][-1]
        bootstrap_differences["ridge_minus_xgboost_mae"].append(
            ridge_mae - xgboost_mae
        )
        bootstrap_differences["ridge_minus_rf_mae"].append(
            ridge_mae - random_forest_mae
        )

    model_uncertainty: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        macro_mae_exact = float(
            np.mean(
                [
                    participant_models[participant][model_name]["mae"]
                    for participant in participants
                ]
            )
        )
        macro_rmse_exact = float(
            np.mean(
                [
                    participant_models[participant][model_name]["rmse"]
                    for participant in participants
                ]
            )
        )
        macro_boot = np.asarray(
            bootstrap_macro_mae[model_name],
            dtype=float,
        )
        macro_ci_low, macro_ci_high = interval(macro_boot)
        pooled_truth = np.concatenate(
            [participant_truth[participant] for participant in participants]
        )
        pooled_prediction = np.concatenate(
            [
                participant_prediction[participant][model_name]
                for participant in participants
            ]
        )
        pooled_point = regression_metrics(pooled_truth, pooled_prediction)
        macro_mae_result = {
            "estimate": macro_mae_exact,
            "median": float(np.median(macro_boot)),
            "ci95_low": macro_ci_low,
            "ci95_high": macro_ci_high,
        }

        model_uncertainty[model_name] = {
            "pooled": {},
            "macro_mae": macro_mae_result,
            "macro_rmse": {"estimate": macro_rmse_exact},
        }
        for metric in ("mae", "rmse", "r2"):
            values = np.asarray(
                bootstrap_pooled[model_name][metric],
                dtype=float,
            )
            ci_low, ci_high = interval(values)
            model_uncertainty[model_name]["pooled"][metric] = {
                "estimate": pooled_point[metric],
                "median": float(np.median(values)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }

    difference_uncertainty: dict[str, Any] = {}
    for difference_name, values in bootstrap_differences.items():
        array = np.asarray(values, dtype=float)
        ci_low, ci_high = interval(array)
        difference_uncertainty[difference_name] = {
            "estimate": float(np.mean(array)),
            "median": float(np.median(array)),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
        }

    report: dict[str, Any] = {
        "validation": "leave-one-participant-out",
        "window_days": args.days,
        "positive_definition": "risk_score >= 1",
        "participants": len(participants),
        "test_windows": sum(
            len(participant_truth[participant])
            for participant in participants
        ),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_unit": "participant",
        "seed": args.seed,
        "model_uncertainty": model_uncertainty,
        "paired_differences_mae": difference_uncertainty,
        "participant_error": participant_error,
        "error_by_true_class": error_by_true_class,
    }

    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    uncertainty_headers = ["Modelo", "MAE", "IC 95% MAE", "R²", "IC 95% R²"]
    uncertainty_rows: list[list[Any]] = []
    for model_name in MODEL_NAMES:
        mae = model_uncertainty[model_name]["pooled"]["mae"]
        r2 = model_uncertainty[model_name]["pooled"]["r2"]
        uncertainty_rows.append([
            model_name,
            mae["estimate"],
            f"[{mae['ci95_low']:.3f}; {mae['ci95_high']:.3f}]",
            r2["estimate"],
            f"[{r2['ci95_low']:.3f}; {r2['ci95_high']:.3f}]",
        ])

    markdown = [
        "# Incerteza e análise de erros — LOPO 7 dias",
        "",
        "## Método",
        "",
        f"- Participantes: {report['participants']}",
        f"- Janelas LOPO: {report['test_windows']}",
        f"- Reamostras bootstrap por participante: {args.bootstrap_samples}",
        f"- Semente: {args.seed}",
        "- Intervalos de 95% obtidos pelos percentis 2,5% e 97,5%",
        "- As diferenças pareadas usam as mesmas reamostras participantes",
        "",
        "## Incerteza dos modelos",
        "",
        markdown_table(uncertainty_headers, uncertainty_rows),
        "",
        "## Diferenças pareadas de MAE",
        "",
    ]
    difference_headers = ["Comparação", "Diferença", "IC 95%", "Interpretação"]
    difference_rows: list[list[Any]] = []
    for name, result in difference_uncertainty.items():
        crosses_zero = result["ci95_low"] <= 0 <= result["ci95_high"]
        interpretation = (
            "direção incerta" if crosses_zero else "direção consistente"
        )
        label = (
            "Ridge menos XGBoost"
            if name == "ridge_minus_xgboost_mae"
            else "Ridge menos Random Forest"
        )
        difference_rows.append([
            label,
            result["estimate"],
            f"[{result['ci95_low']:.3f}; {result['ci95_high']:.3f}]",
            interpretation,
        ])
    markdown.append(
        markdown_table(difference_headers, difference_rows)
    )

    markdown.extend(["", "## MAE por participante", ""])
    participant_headers = ["Participante", "Janelas"]
    for model_name in MODEL_NAMES:
        participant_headers.extend([f"{model_name} MAE", f"{model_name} R²"])
    participant_rows: list[list[Any]] = []
    for row in sorted(
        participant_error,
        key=lambda item: item["models"]["Ridge"]["mae"],
        reverse=True,
    ):
        values: list[Any] = [row["participant"], row["test_windows"]]
        for model_name in MODEL_NAMES:
            r2_value = row["models"][model_name]["r2"]
            values.extend([
                row["models"][model_name]["mae"],
                "n.e." if r2_value is None else r2_value,
            ])
        participant_rows.append(values)
    markdown.append(markdown_table(participant_headers, participant_rows))

    markdown.extend(["", "## Erro por classe verdadeira", ""])
    class_headers = ["Classe", "N"]
    for model_name in MODEL_NAMES:
        class_headers.extend([f"{model_name} MAE", f"{model_name} bias"])
    class_rows: list[list[Any]] = []
    for str_class, row in error_by_true_class.items():
        values: list[Any] = [str_class, row["n"]]
        for model_name in MODEL_NAMES:
            values.extend([
                row["models"][model_name]["mae"],
                row["models"][model_name]["bias"],
            ])
        class_rows.append(values)
    markdown.append(markdown_table(class_headers, class_rows))

    markdown.extend([
        "",
        "## Limitações",
        "",
        "- O bootstrap por participante preserva a estrutura de amostra por "
        "participante, mas não cria novos participantes.",
        "- As janelas dentro de um participante permanecem correlacionadas.",
        "- Os resultados não substituem validação externa.",
        "- As métricas continuam avaliando o proxy, não diagnóstico clínico de DKA.",
        "",
    ])

    markdown_output = args.output.with_suffix(".md")
    markdown_output.write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )

    root = args.output.parent
    save_uncertainty_figure(
        report,
        root / "lopo_uncertainty_7d.png",
    )
    save_error_figure(
        report,
        root / "lopo_participant_error_7d.png",
    )

    print(json.dumps(report["paired_differences_mae"], indent=2))
    print(f"JSON: {args.output}")
    print(f"Markdown: {markdown_output}")


if __name__ == "__main__":
    main()
