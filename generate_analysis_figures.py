"""Generate human-readable figures for the final TCC2 volume."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "Manchester_processed"
FIGURE_DIR = ROOT / "figures"


def load(name: str) -> dict:
    return json.loads((REPORT_DIR / name).read_text(encoding="utf-8"))


def save_lopo_comparison() -> Path:
    configurations = [
        ("Glicose", "reproduced_lopo_7d_glucose_ridge.json"),
        ("Resumo diário\nde sono", "reproduced_lopo_7d_sleep_summary_ridge.json"),
        ("Telemetria pontual\ndo relógio", "reproduced_lopo_7d_sleep_step_stress_clean.json"),
    ]
    combination_report = load("feature_combination_lopo_7d.json")
    metrics = ["mae", "rmse", "r2"]
    values = {
        metric: {"XGBoost": [], "Random Forest": [], "Ridge": []}
        for metric in metrics
    }
    for _, filename in configurations:
        if filename == "reproduced_lopo_7d_sleep_step_stress_clean.json":
            report = next(
                result
                for result in combination_report["results"]
                if result["groups"] == ["glucose", "sleep_ts"]
            )
            report = {"models": report["models"]}
        else:
            report = load(filename)
        for metric in metrics:
            for model in values[metric]:
                values[metric][model].append(
                    report["models"][model]["pooled_regression"][metric]
                )

    FIGURE_DIR.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(13, 5.2), constrained_layout=True)
    x = np.arange(len(configurations))
    width = 0.78
    models = ["XGBoost", "Random Forest", "Ridge"]
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    labels = [
        "Glicose",
        "Glicose +\nsono diário",
        "Glicose +\ntelemetria do relógio",
    ]
    for axis, metric in zip(axes, metrics):
        for model_index, (model, color) in enumerate(zip(models, colors)):
            offset = (model_index - 1) * width / 3
            bars = axis.bar(
                x + offset,
                values[metric][model],
                width / 3,
                label=model,
                color=color,
            )
            axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
        axis.set_title(metric.upper())
        axis.set_xticks(x, labels)
        axis.tick_params(axis="x", labelsize=9)
        for tick in axis.get_xticklabels():
            tick.set_horizontalalignment("center")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.set_ylim(bottom=0)
    axes[0].set_ylabel("Valor da métrica")
    axes[0].legend(fontsize=8, loc="upper left")
    figure.suptitle(
        "Validação LOPO de 7 dias: configurações multimodais (228 janelas)",
        fontsize=14,
    )
    output = FIGURE_DIR / "lopo_7d_comparison.png"
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def save_horizon_comparison() -> Path:
    horizons = [3, 7, 14]
    reports = {
        "Glicose": [load(f"reproduced_lopo_{days}d_glucose_ridge.json") for days in horizons],
        "Sono diário": [
            load(f"reproduced_lopo_{days}d_sleep_summary_ridge.json") for days in horizons
        ],
    }
    FIGURE_DIR.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.0), constrained_layout=True)
    model_colors = {
        "XGBoost": "#4C78A8",
        "Random Forest": "#F58518",
        "Ridge": "#54A24B",
    }
    window_counts = {3: 536, 7: 228, 14: 109}
    for model in ["XGBoost", "Random Forest", "Ridge"]:
        for label, model_reports in reports.items():
            marker = "o" if label == "Glicose" else "s"
            linestyle = "-" if label == "Glicose" else "--"
            axes[0].plot(
                horizons,
                [report["models"][model]["pooled_regression"]["mae"] for report in model_reports],
                marker=marker,
                linestyle=linestyle,
                color=model_colors[model],
                label=f"{label} — {model}",
            )
            axes[1].plot(
                horizons,
                [report["models"][model]["pooled_regression"]["r2"] for report in model_reports],
                marker=marker,
                linestyle=linestyle,
                color=model_colors[model],
                label=f"{label} — {model}",
            )
    for axis, title in zip(axes, ["MAE por horizonte", "$R^2$ por horizonte"]):
        axis.set_title(title)
        axis.set_xlabel("Janela (dias)")
        axis.set_xticks(horizons)
        axis.grid(alpha=0.25)
        axis.set_axisbelow(True)
    for days in horizons:
        axes[0].annotate(
            f"{window_counts[days]}",
            (days, axes[0].get_ylim()[0]),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[0].set_ylabel("MAE (menor é melhor)")
    axes[1].set_ylabel("$R^2$ (maior é melhor)")
    axes[1].legend(fontsize=7, loc="best")
    figure.suptitle("Generalização LOPO nos horizontes temporais", fontsize=13)
    output = FIGURE_DIR / "lopo_horizons_comparison.png"
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def save_proxy_distribution() -> Path:
    report = load("reproduced_baseline_7d_glucose.json")
    classes = [str(value) for value in range(6)]
    train = [report["train_score_distribution"][value] for value in classes]
    test = [report["test_score_distribution"][value] for value in classes]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    x = np.arange(6)
    width = 0.38

    train_bars = axes[0].bar(x - width / 2, train, width, label="Treino", color="#4C78A8")
    test_bars = axes[0].bar(x + width / 2, test, width, label="Teste", color="#F58518")
    axes[0].bar_label(train_bars, padding=2, fontsize=8)
    axes[0].bar_label(test_bars, padding=2, fontsize=8)
    axes[0].set_title("Distribuição ordinal do proxy")
    axes[0].set_xlabel("Classe do risk_score")
    axes[0].set_xticks(x, classes)
    axes[0].set_ylabel("Número de janelas")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_axisbelow(True)
    axes[0].legend()

    train_binary = report["train_class_distribution"]
    test_binary = report["test_class_distribution"]
    train_values = [train_binary["0"], train_binary["1"]]
    test_values = [test_binary["0"], test_binary["1"]]
    binary_x = np.arange(2)
    binary_train = axes[1].bar(
        binary_x - width / 2,
        [100 * value / sum(train_values) for value in train_values],
        width,
        label="Treino",
        color="#4C78A8",
    )
    binary_test = axes[1].bar(
        binary_x + width / 2,
        [100 * value / sum(test_values) for value in test_values],
        width,
        label="Teste",
        color="#F58518",
    )
    axes[1].bar_label(binary_train, fmt="%.1f%%", padding=2, fontsize=8)
    axes[1].bar_label(binary_test, fmt="%.1f%%", padding=2, fontsize=8)
    axes[1].set_title("Limiar binário risk_score >= 1")
    axes[1].set_xticks(binary_x, ["Negativa (0)", "Positiva (1–5)"])
    axes[1].set_ylim(0, 100)
    axes[1].set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_axisbelow(True)

    figure.suptitle("Desbalanceamento do proxy na janela de 7 dias", fontsize=14)
    FIGURE_DIR.mkdir(exist_ok=True)
    output = FIGURE_DIR / "proxy_distribution_7d.png"
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def save_pipeline() -> Path:
    FIGURE_DIR.mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(18, 5.0))
    axis.axis("off")
    steps = [
        ("Dados brutos", "Glicose, sono,\ninsulina e nutrição"),
        ("Pré-processamento", "timestamps e unidades\ngrade de 5 min"),
        ("Auditoria", "cronologia e gaps\ncobertura modal"),
        ("Janelas por participante", "t → t+1\n3, 7 e 14 dias"),
        ("Modelos globais", "XGBoost\nRandom Forest\nRidge"),
        ("Validação temporal", "70% inicial\n30% final"),
        ("Generalização", "LOPO\num participante fora"),
    ]
    positions = np.linspace(0.08, 0.92, len(steps))
    for index, ((title, detail), x) in enumerate(zip(steps, positions)):
        axis.text(
            x,
            0.53,
            f"{title}\n\n{detail}",
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.7", "facecolor": "#e8f1f8", "edgecolor": "#39739d"},
        )
        if index < len(steps) - 1:
            axis.annotate(
                "",
                xy=(positions[index + 1] - 0.07, 0.53),
                xytext=(x + 0.07, 0.53),
                arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#39739d"},
            )
    axis.set_title("Pipeline reprodutível da análise Manchester", fontsize=13, pad=12)
    output = FIGURE_DIR / "analysis_pipeline.png"
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


if __name__ == "__main__":
    for path in (
        save_pipeline(),
        save_lopo_comparison(),
        save_horizon_comparison(),
        save_proxy_distribution(),
    ):
        print(path)
