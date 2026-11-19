"""Normalize Manchester multimodal data into one 5-minute file per participant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


DATASET_DIR = Path(__file__).resolve().parent / "ManchesterCSCoordinatedDiabetesStudy-main"
OUTPUT_DIR = Path(__file__).resolve().parent / "Manchester_processed"
GLUCOSE_TO_MGDL = 18.0
FREQUENCY = "5min"
MAX_DATASET_TIMESTAMP = pd.Timestamp("2024-12-31 23:59:59")


def participant_id(path: Path) -> str:
    digits = "".join(char for char in path.stem if char.isdigit())
    return digits


def parse_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="mixed", dayfirst=True, errors="coerce")


def read_time_series(path: Path, timestamp_column: str) -> tuple[pd.DataFrame, int]:
    frame = pd.read_csv(path)
    frame["timestamp"] = parse_timestamp(frame.pop(timestamp_column))
    invalid = int(frame["timestamp"].isna().sum())
    if invalid:
        print(f"WARNING: dropping {invalid} invalid timestamp(s) from {path.name}")
        frame = frame.dropna(subset=["timestamp"]).copy()
    suspicious = int((frame["timestamp"] > MAX_DATASET_TIMESTAMP).sum())
    if suspicious:
        print(
            f"WARNING: dropping {suspicious} timestamp(s) after "
            f"{MAX_DATASET_TIMESTAMP.date()} from {path.name}"
        )
        frame = frame[frame["timestamp"] <= MAX_DATASET_TIMESTAMP].copy()
    return frame.set_index("timestamp").sort_index(), suspicious


def aggregate_numeric(
    frame: pd.DataFrame,
    prefix: str,
    aggregations: dict[str, str],
) -> pd.DataFrame:
    available = {column: operation for column, operation in aggregations.items() if column in frame}
    if not available:
        return pd.DataFrame(index=frame.resample(FREQUENCY).asfreq().index)
    result = frame[list(available)].apply(pd.to_numeric, errors="coerce").resample(FREQUENCY).agg(available)
    return result.rename(columns={column: f"{prefix}_{column}" for column in available})


def clean_sleep_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    numeric_columns = [
        "step_count",
        "heart_rate",
        "stress_level_value",
        "resting_heart_rate",
    ]
    for column in numeric_columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "heart_rate" in frame:
        frame.loc[frame["heart_rate"] <= 0, "heart_rate"] = pd.NA
    if "resting_heart_rate" in frame:
        frame.loc[frame["resting_heart_rate"] <= 0, "resting_heart_rate"] = pd.NA
    if "stress_level_value" in frame:
        frame.loc[frame["stress_level_value"] < 0, "stress_level_value"] = pd.NA
    return frame


def read_sleep_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["summary_date"] = parse_timestamp(frame["calendar_date"]).dt.normalize()
    frame["start_date_ts"] = parse_timestamp(frame["start_date_ts"])
    frame = frame.dropna(subset=["summary_date"])
    frame = frame[frame["summary_date"] <= MAX_DATASET_TIMESTAMP.normalize()]
    columns = {
        "duration_in_sec": "sleep_summary_duration_s",
        "deep_sleep_s": "sleep_summary_deep_s",
        "light_sleep_s": "sleep_summary_light_s",
        "rem_sleep_s": "sleep_summary_rem_s",
        "awake_s": "sleep_summary_awake_s",
        "unmeasurable_sleep_s": "sleep_summary_unmeasurable_s",
    }
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    summary = frame.groupby("summary_date")[list(columns)].sum()
    summary = summary.rename(columns=columns)
    summary["sleep_summary_sessions"] = frame.groupby("summary_date").size()
    return summary


def merge_modalities(participant: str, glucose_path: Path) -> pd.DataFrame:
    glucose, _ = read_time_series(glucose_path, "bg_ts")
    glucose = glucose.rename(columns={"value": "glucose_mmol_l"})
    glucose["glucose_mg_dl"] = glucose["glucose_mmol_l"] * GLUCOSE_TO_MGDL
    result = glucose[["glucose_mmol_l", "glucose_mg_dl"]].resample(FREQUENCY).mean()

    sources: Iterable[tuple[str, Path, str, dict[str, str]]] = (
        (
            "activity",
            DATASET_DIR / "Activity Data" / f"UoMActivity{participant}.csv",
            "activity_ts",
            {
                "active_Kcal": "sum",
                "step_count": "sum",
                "distance_m": "sum",
                "duration_s": "sum",
                "active_time_s": "sum",
                "met": "mean",
                "motion_intensity_mean": "mean",
                "motion_intensity_max": "max",
            },
        ),
        (
            "sleep",
            DATASET_DIR / "Sleep Data" / f"UoMsleep{participant}.csv",
            "sleep_ts",
            {
                "step_count": "sum",
                "heart_rate": "mean",
                "current_activity_type_intensity": "mean",
                "stress_level_value": "mean",
                "sleep_level": "mean",
                "resting_heart_rate": "mean",
            },
        ),
        (
            "basal",
            DATASET_DIR / "Insulin Data" / "Basal Data" / f"UoMBasal{participant}.csv",
            "basal_ts",
            {"basal_dose": "sum"},
        ),
        (
            "bolus",
            DATASET_DIR / "Insulin Data" / "Bolus Data" / f"UoMBolus{participant}.csv",
            "bolus_ts",
            {"bolus_dose": "sum"},
        ),
        (
            "nutrition",
            DATASET_DIR / "Nutrition Data" / f"UoMNutrition{participant}.csv",
            "meal_ts",
            {"carbs_g": "sum", "prot_g": "sum", "fat_g": "sum", "fibre_g": "sum"},
        ),
    )
    for prefix, path, timestamp_column, aggregations in sources:
        if not path.exists():
            continue
        frame, _ = read_time_series(path, timestamp_column)
        if prefix == "sleep":
            frame = clean_sleep_measurements(frame)
        result = result.join(aggregate_numeric(frame, prefix, aggregations), how="left")

    summary_path = DATASET_DIR / "Sleep Data" / f"UoM{participant}sleeptime.csv"
    if summary_path.exists():
        summary = read_sleep_summary(summary_path)
        aligned_summary = summary.reindex(result.index.normalize())
        aligned_summary.index = result.index
        result = result.join(aligned_summary, how="left")

    result = result.reset_index(names="timestamp")
    result["participant_id"] = participant
    result["timestamp"] = result["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    ordered = ["participant_id", "timestamp"] + [
        column for column in result.columns if column not in {"participant_id", "timestamp"}
    ]
    return result[ordered]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    glucose_files = sorted((DATASET_DIR / "Glucose Data").glob("UoMGlucose*.csv"))
    if not glucose_files:
        raise FileNotFoundError("No Manchester glucose files found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "source": str(DATASET_DIR),
        "output": str(args.output_dir),
        "timestamp_format": "YYYY-MM-DDTHH:MM:SS, local dataset time without timezone",
        "frequency": FREQUENCY,
        "glucose_conversion": "glucose_mg_dl = glucose_mmol_l * 18.0",
        "participants": {},
    }
    for path in glucose_files:
        participant = participant_id(path)
        output = args.output_dir / f"participant_{participant}.csv"
        processed = merge_modalities(participant, path)
        processed.to_csv(output, index=False)
        manifest["participants"][participant] = {
            "source": str(path),
            "output": str(output),
            "rows": int(len(processed)),
            "columns": processed.columns.tolist(),
            "start": processed["timestamp"].min(),
            "end": processed["timestamp"].max(),
        }
        print(f"{participant}: {len(processed)} rows -> {output.name}")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
