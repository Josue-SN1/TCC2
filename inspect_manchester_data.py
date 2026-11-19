"""Create a reproducible inventory of the Manchester multimodal dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


DATASET_DIR = Path(__file__).resolve().parent / "ManchesterCSCoordinatedDiabetesStudy-main"
MODALITIES = {
    "glucose": ("Glucose Data", "UoMGlucose*.csv", "bg_ts"),
    "activity": ("Activity Data", "UoMActivity*.csv", "activity_ts"),
    "sleep": ("Sleep Data", "UoMsleep[0-9]*.csv", "sleep_ts"),
    "nutrition": ("Nutrition Data", "UoMNutrition*.csv", "meal_ts"),
    "basal": ("Insulin Data/Basal Data", "UoMBasal*.csv", "basal_ts"),
    "bolus": ("Insulin Data/Bolus Data", "UoMBolus*.csv", "bolus_ts"),
}


def participant_from_name(path: Path) -> str:
    return path.stem.removeprefix("UoM").removeprefix("Activity").removeprefix("Glucose")


def inspect_file(path: Path, timestamp_column: str) -> dict[str, Any]:
    frame = pd.read_csv(path)
    timestamps = pd.to_datetime(
        frame[timestamp_column], format="mixed", dayfirst=True, errors="coerce"
    )
    invalid = int(timestamps.isna().sum())
    dates: dict[str, Any] = {
        "rows": int(len(frame)),
        "participant": participant_from_name(path),
        "columns": frame.columns.tolist(),
        "invalid_timestamps": invalid,
        "start": timestamps.min().isoformat() if timestamps.notna().any() else None,
        "end": timestamps.max().isoformat() if timestamps.notna().any() else None,
    }
    if timestamps.notna().any():
        dates["future_dates_after_2024_12_31"] = int((timestamps > pd.Timestamp("2024-12-31")).sum())
    return dates


def main() -> None:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")
    report: dict[str, Any] = {
        "dataset": "T1D-UOM / ManchesterCSCoordinatedDiabetesStudy",
        "dataset_root": str(DATASET_DIR),
        "units": {
            "glucose": "mmol/L; convert explicitly before applying mg/dL thresholds",
            "activity": "Garmin activity and motion features",
            "sleep": "Garmin sleep, heart rate, and stress features",
        },
        "modalities": {},
    }
    for name, (folder, pattern, timestamp_column) in MODALITIES.items():
        files = sorted((DATASET_DIR / folder).glob(pattern))
        report["modalities"][name] = {
            "files": len(files),
            "participants": sorted({participant_from_name(path) for path in files}),
            "records": {path.name: inspect_file(path, timestamp_column) for path in files},
        }
    destination = Path(__file__).resolve().parent / "manchester_inventory.json"
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nInventory written to {destination}")


if __name__ == "__main__":
    main()
