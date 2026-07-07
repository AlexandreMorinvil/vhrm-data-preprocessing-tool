from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import SignalLoader

log = logging.getLogger(__name__)


class ZephyrSensorLoader(SignalLoader):
    display_name = "Zephyr Sensor"

    def can_load(self, path: str) -> bool:
        try:
            p = Path(path)
            if p.suffix.lower() != ".csv":
                return False
            header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
            columns = {str(c).strip() for c in header}
            zephyr_markers = {"BR", "SkinTemp", "Posture", "Activity", "HRConfidence"}
            return "Time" in columns and "HR" in columns and bool(columns & zephyr_markers)
        except Exception:
            return False

    def load(self, path: str) -> pd.DataFrame:
        samples = pd.read_csv(path, encoding="utf-8-sig")
        if "Time" not in samples.columns:
            raise ValueError(f"Cannot find Time column in {path}")
        if "HR" not in samples.columns:
            raise ValueError(f"Cannot find HR column in {path}")

        timestamps = pd.to_datetime(
            samples["Time"],
            dayfirst=True,
            errors="coerce",
            utc=True,
        )
        values = pd.to_numeric(samples["HR"], errors="coerce")

        df = pd.DataFrame({
            "timestamp_utc": timestamps,
            "value": values,
            "sensor_id": Path(path).stem,
        })
        df = df.dropna(subset=["timestamp_utc", "value"]).reset_index(drop=True)
        log.info("Loaded %d Zephyr HR samples from %s", len(df), path)
        return df
