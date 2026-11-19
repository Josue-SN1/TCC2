"""Audit temporal alignment and coverage across Manchester modalities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "ManchesterCSCoordinatedDiabetesStudy-main"
OUTPUT_JSON = ROOT / "manchester_alignment_audit.json"
OUTPUT_MD = ROOT / "manchester_alignment_audit.md"
MAX_TIMESTAMP = pd.Timestamp("2024-12-31 23:59:59")

MODALITIES = {
    "activity": ("Activity Data", "UoMActivity{participant}.csv", "activity_ts"),
    "sleep_ts": ("Sleep Data", "UoMsleep{participant}.csv", "sleep_ts"),
    "nutrition": ("Nutrition Data", "UoMNutrition{participant}.csv", "meal_ts"),
    "basal": (
        "Insulin Data/Basal Data",
        "UoMBasal{participant}.csv",
        "basal_ts",
    ),
    "bolus": ("Insulin Data/Bolus Data", "UoMBolus{participant}.csv", "bolus_ts"),
}


def parse_timestamps(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, format="mixed", dayfirst=True, errors="coerce")


def read_series(path: Path, timestamp_column: str) -> dict[str, Any]:
    frame = pd.read_csv(path, usecols=[timestamp_column])
    timestamps = parse_timestamps(frame[timestamp_column])
    valid = timestamps.dropna()
    bins = valid.dt.floor("5min")
    return {
        "file": str(path),
        "rows": int(len(frame)),
        "invalid_timestamps": int(timestamps.isna().sum()),
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
        "duplicate_5min_bins": int(bins.duplicated().sum()),
        "start": valid.min().isoformat() if not valid.empty else None,
        "end": valid.max().isoformat() if not valid.empty else None,
        "bins": bins,
    }


def audit_summary(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path, usecols=["calendar_date", "start_date_ts"])
    calendar = parse_timestamps(frame["calendar_date"]).dt.normalize()
    start = parse_timestamps(frame["start_date_ts"])
    delta_days = (calendar - start.dt.normalize()).dt.days.dropna()
    return {
        "file": str(path),
        "rows": int(len(frame)),
        "invalid_calendar_dates": int(calendar.isna().sum()),
        "invalid_start_timestamps": int(start.isna().sum()),
        "calendar_start_day_offsets": delta_days.value_counts().sort_index().astype(int).to_dict(),
        "calendar_start_mismatch_rows": int((delta_days != 0).sum()),
    }


def main() -> None:
    glucose_paths = sorted((DATASET_DIR / "Glucose Data").glob("UoMGlucose*.csv"))
    report: dict[str, Any] = {
        "timestamp_policy": "format=mixed, dayfirst=True, local timestamps without timezone",
        "binning": "timestamps floored to five-minute bins for overlap checks",
        "max_timestamp": str(MAX_TIMESTAMP),
        "participants": {},
    }

    for glucose_path in glucose_paths:
        participant = "".join(char for char in glucose_path.stem if char.isdigit())
        glucose = read_series(glucose_path, "bg_ts")
        glucose_bins = set(glucose.pop("bins"))
        modalities: dict[str, Any] = {}
        for name, (folder, pattern, column) in MODALITIES.items():
            path = DATASET_DIR / folder / pattern.format(participant=participant)
            if not path.exists():
                modalities[name] = {"status": "missing", "file": str(path)}
                continue
            result = read_series(path, column)
            bins = set(result.pop("bins"))
            overlap = bins & glucose_bins
            result.update(
                {
                    "status": "available",
                    "common_5min_bins": len(overlap),
                    "glucose_5min_bins": len(glucose_bins),
                    "modality_5min_bins": len(bins),
                    "overlap_fraction_of_glucose": (
                        len(overlap) / len(glucose_bins) if glucose_bins else 0.0
                    ),
                }
            )
            modalities[name] = result

        summary_path = DATASET_DIR / "Sleep Data" / f"UoM{participant}sleeptime.csv"
        summary = (
            audit_summary(summary_path)
            if summary_path.exists()
            else {"status": "missing", "file": str(summary_path)}
        )
        report["participants"][participant] = {
            "glucose": glucose,
            "modalities": modalities,
            "sleep_summary": summary,
        }

    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    lines = [
        "# Auditoria de alinhamento temporal Manchester",
        "",
        "- Parsing: `format=mixed`, `dayfirst=True`.",
        "- Checagem de séries: bins de cinco minutos, sem imputação.",
        "- Resumos `sleeptime`: comparação entre `calendar_date` e `start_date_ts`.",
        "",
        "| Participante | Modalidade | Bins comuns | Cobertura da glicose | Duplicatas |",
        "|---|---|---:|---:|---:|",
    ]
    for participant, data in report["participants"].items():
        for name, modality in data["modalities"].items():
            if modality.get("status") != "available":
                lines.append(f"| {participant} | {name} | - | ausente | - |")
                continue
            lines.append(
                f"| {participant} | {name} | {modality['common_5min_bins']} | "
                f"{modality['overlap_fraction_of_glucose']:.1%} | "
                f"{modality['duplicate_5min_bins']} |"
            )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON written to {OUTPUT_JSON}")
    print(f"Markdown written to {OUTPUT_MD}")


if __name__ == "__main__":
    main()
