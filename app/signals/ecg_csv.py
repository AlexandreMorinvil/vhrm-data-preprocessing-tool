from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import SignalLoader

log = logging.getLogger(__name__)


class EcgWaveformLoader(SignalLoader):
    display_name = "ECG waveform"
    sensor_type = "ECG"

    def can_load(self, path: str) -> bool:
        try:
            if Path(path).suffix.lower() != ".csv":
                return False
            columns = pd.read_csv(path, nrows=0, encoding="utf-8").columns
            return {"Time", "EcgWaveform"}.issubset(columns)
        except Exception:
            return False

    def load(self, path: str) -> pd.DataFrame:
        samples = pd.read_csv(path, usecols=["Time", "EcgWaveform"], encoding="utf-8")
        timestamps = pd.to_datetime(samples["Time"], dayfirst=True, utc=True, errors="coerce")
        values = pd.to_numeric(samples["EcgWaveform"], errors="coerce")
        df = pd.DataFrame({
            "timestamp_utc": timestamps,
            "value": values,
            "sensor_id": Path(path).stem,
        }).dropna(subset=["timestamp_utc", "value"])
        log.info("Loaded %d ECG samples from %s", len(df), path)
        return df.reset_index(drop=True)