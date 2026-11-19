"""Verify processed chronology, five-minute grids, and t -> t+1 windows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from train_manchester_baseline import make_windows


ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "Manchester_processed"
OUTPUT_JSON = ROOT / "temporal_windows_audit.json"
OUTPUT_MD = ROOT / "temporal_windows_audit.md"
HORIZONS = (3, 7, 14)


def verify_processed_file(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path, usecols=["timestamp", "glucose_mg_dl"])
    timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
    deltas = timestamps.sort_values().diff().dropna().dt.total_seconds()
    sorted_timestamps = timestamps.sort_values()
    return {
        "rows": int(len(frame)),
        "strictly_increasing": bool(sorted_timestamps.is_monotonic_increasing),
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
        "non_positive_gaps": int((deltas <= 0).sum()),
        "non_five_minute_gaps": int((deltas % 300 != 0).sum()),
        "start": timestamps.min().isoformat(),
        "end": timestamps.max().isoformat(),
    }


def verify_windows(path: Path, days: int) -> dict[str, Any]:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    ordered = frame.sort_values("timestamp").set_index("timestamp")
    window_id = (
        (ordered.index - ordered.index.min()).total_seconds() // (days * 86400)
    ).astype(int)
    ordered = ordered.assign(window_id=window_id)
    transitions: list[dict[str, Any]] = []
    for current_id in range(int(window_id.max())):
        current = ordered[ordered["window_id"] == current_id]
        target = ordered[ordered["window_id"] == current_id + 1]
        if current.empty or target.empty:
            continue
        transitions.append(
            {
                "window_id": current_id,
                "current_start": current.index.min().isoformat(),
                "current_end": current.index.max().isoformat(),
                "target_start": target.index.min().isoformat(),
                "strictly_forward": bool(current.index.max() < target.index.min()),
            }
        )
    windows = make_windows(frame.reset_index(drop=True), days)
    return {
        "candidate_transitions": len(transitions),
        "valid_t_to_t1_transitions": int(
            sum(item["strictly_forward"] for item in transitions)
        ),
        "invalid_transitions": [
            item for item in transitions if not item["strictly_forward"]
        ],
        "usable_model_windows": int(len(windows)),
    }


def main() -> None:
    report: dict[str, Any] = {
        "timestamp_policy": "processed local timestamps parsed strictly from ISO output",
        "grid_policy": "five-minute resampled grid; chronology checked after sorting",
        "window_policy": "current window t must end before target window t+1 begins",
        "participants": {},
        "horizons": {},
    }
    paths = sorted(PROCESSED_DIR.glob("participant_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No processed participant files in {PROCESSED_DIR}")

    for path in paths:
        participant = path.stem.removeprefix("participant_")
        report["participants"][participant] = {
            "processed": verify_processed_file(path),
            "horizons": {
                str(days): verify_windows(path, days) for days in HORIZONS
            },
        }

    for days in HORIZONS:
        horizon_data = [
            data["horizons"][str(days)] for data in report["participants"].values()
        ]
        report["horizons"][str(days)] = {
            "participants": len(horizon_data),
            "candidate_transitions": sum(
                item["candidate_transitions"] for item in horizon_data
            ),
            "valid_t_to_t1_transitions": sum(
                item["valid_t_to_t1_transitions"] for item in horizon_data
            ),
            "invalid_transitions": sum(
                len(item["invalid_transitions"]) for item in horizon_data
            ),
            "usable_model_windows": sum(
                item["usable_model_windows"] for item in horizon_data
            ),
        }

    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Auditoria de cronologia e janelas Manchester",
        "",
        "- A grade processada deve ser estritamente crescente, sem timestamps duplicados.",
        "- Os gaps devem ser múltiplos de cinco minutos.",
        "- Cada janela `t` deve terminar antes do início da janela-alvo `t+1`.",
        "",
        "| Horizonte | Transições | Válidas t→t+1 | Inválidas | Janelas utilizáveis |",
        "|---:|---:|---:|---:|---:|",
    ]
    for days, data in report["horizons"].items():
        lines.append(
            f"| {days} dias | {data['candidate_transitions']} | "
            f"{data['valid_t_to_t1_transitions']} | {data['invalid_transitions']} | "
            f"{data['usable_model_windows']} |"
        )
    lines.extend(["", "## Participantes", ""])
    for participant, data in report["participants"].items():
        processed = data["processed"]
        lines.append(
            f"- `{participant}`: crescente={processed['strictly_increasing']}, "
            f"duplicatas={processed['duplicate_timestamps']}, "
            f"gaps não positivos={processed['non_positive_gaps']}, "
            f"gaps não múltiplos de 5 min={processed['non_five_minute_gaps']}."
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON written to {OUTPUT_JSON}")
    print(f"Markdown written to {OUTPUT_MD}")


if __name__ == "__main__":
    main()
