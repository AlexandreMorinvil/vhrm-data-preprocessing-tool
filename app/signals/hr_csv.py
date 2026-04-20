from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import SignalLoader

log = logging.getLogger(__name__)


class HRCsvLoader(SignalLoader):
    def can_load(self, path: str) -> bool:
        try:
            p = Path(path)
            if p.suffix.lower() != ".csv":
                return False
            with open(path, "r", encoding="utf-8") as f:
                header0 = f.readline()
                _values = f.readline()
                header2 = f.readline()
            if "Date" not in header0 or "Start time" not in header0:
                return False
            if "Time" not in header2 or "HR" not in header2:
                return False
            return True
        except Exception:
            return False

    def load(self, path: str) -> pd.DataFrame:
        meta = pd.read_csv(path, nrows=1, header=0, encoding="utf-8")
        date_str = str(meta["Date"].iloc[0]).strip()
        start_str = str(meta["Start time"].iloc[0]).strip()

        base_dt = None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
            try:
                base_dt = datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
        if base_dt is None:
            raise ValueError(f"Cannot parse date: {date_str}")

        parts = start_str.split(":")
        base_dt = base_dt.replace(
            hour=int(parts[0]),
            minute=int(parts[1]),
            second=int(parts[2]) if len(parts) > 2 else 0,
        )

        samples = pd.read_csv(path, skiprows=2, header=0, encoding="utf-8")

        hr_col = None
        for c in samples.columns:
            if "HR" in c.upper() and "BPM" in c.upper():
                hr_col = c
                break
        if hr_col is None:
            for c in samples.columns:
                if "HR" in c.upper():
                    hr_col = c
                    break
        if hr_col is None:
            raise ValueError(f"Cannot find HR column in {path}")

        time_col = "Time"
        if time_col not in samples.columns:
            raise ValueError(f"Cannot find Time column in {path}")

        def _parse_elapsed(t):
            p = str(t).strip().split(":")
            h, m, s = int(p[0]), int(p[1]), int(p[2])
            return timedelta(hours=h, minutes=m, seconds=s)

        samples = samples.dropna(subset=[hr_col])
        timestamps = samples[time_col].apply(lambda t: base_dt + _parse_elapsed(t))
        values = pd.to_numeric(samples[hr_col], errors="coerce")

        sensor_id = Path(path).stem
        df = pd.DataFrame({
            "timestamp_utc": pd.to_datetime(timestamps, utc=True),
            "value": values,
            "sensor_id": sensor_id,
        })
        df = df.dropna(subset=["value"]).reset_index(drop=True)
        log.info("Loaded %d HR samples from %s", len(df), path)
        return df
