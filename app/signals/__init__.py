from __future__ import annotations

import abc
import importlib
import logging
import pkgutil
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

SIGNAL_CORE_COLUMNS = {"timestamp_utc", "value", "sensor_id"}
SIGNAL_AUX_PREFIX = "aux__"
SYNCED_PRIMARY_COLUMNS = ("heart_rate_bpm", "ecg_waveform")


class SignalLoader(abc.ABC):
    display_name: str = "Signal Loader"
    sensor_type: str = "Signal"

    @abc.abstractmethod
    def can_load(self, path: str) -> bool: ...

    @abc.abstractmethod
    def load(self, path: str) -> pd.DataFrame: ...


_LOADERS: list[SignalLoader] = []
_DISCOVERED = False


def _discover_loaders() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    package_dir = Path(__file__).resolve().parent
    for finder, name, ispkg in pkgutil.iter_modules([str(package_dir)]):
        if name.startswith("_"):
            continue
        module = importlib.import_module(f".{name}", package=__package__)
        for attr in dir(module):
            obj = getattr(module, attr)
            if (
                isinstance(obj, type)
                and issubclass(obj, SignalLoader)
                and obj is not SignalLoader
            ):
                instance = obj()
                if instance not in _LOADERS:
                    _LOADERS.append(instance)
                    log.info("Registered signal loader: %s", obj.__name__)


def get_loaders() -> list[SignalLoader]:
    _discover_loaders()
    return list(_LOADERS)


def get_signal_loader(path: str) -> Optional[SignalLoader]:
    _discover_loaders()
    for loader in _LOADERS:
        if loader.can_load(path):
            return loader
    return None


def signal_file_type_name(path: str) -> str:
    loader = get_signal_loader(path)
    if loader is None:
        return "Unknown sensor"
    return getattr(loader, "display_name", type(loader).__name__)


def _sensor_type(loader: SignalLoader) -> str:
    name = getattr(loader, "sensor_type", "") or getattr(loader, "display_name", "Signal")
    return str(name).replace(" ", "_")


def _with_sensor_id(df: pd.DataFrame, sensor_id: str) -> pd.DataFrame:
    out = df.copy()
    out["sensor_id"] = sensor_id
    return out


def load_signal(path: str) -> Optional[pd.DataFrame]:
    loader = get_signal_loader(path)
    if loader is not None:
        log.info("Loading %s with %s", path, type(loader).__name__)
        return _with_sensor_id(loader.load(path), f"{_sensor_type(loader)}_1")
    log.warning("No loader found for %s", path)
    return None


def load_signal_files(paths: list[str]) -> tuple[list[pd.DataFrame], dict[str, str], list[str]]:
    """Load signal files and assign type-based sensor IDs.

    Files are named ``Polar_1``, ``Polar_2`` or ``Zephyr_1``, ``Zephyr_2`` in
    selection order, including the ``_1`` suffix when only one sensor of a type
    is loaded. Returns ``(dataframes, display_type_by_path, failed_paths)``.
    """
    entries: list[tuple[str, SignalLoader, pd.DataFrame]] = []
    display_type_by_path: dict[str, str] = {}
    failed_paths: list[str] = []

    for path in paths:
        loader = get_signal_loader(path)
        if loader is None:
            display_type_by_path[path] = "Unknown sensor"
            failed_paths.append(path)
            log.warning("No loader found for %s", path)
            continue
        display_type_by_path[path] = getattr(loader, "display_name", type(loader).__name__)
        log.info("Loading %s with %s", path, type(loader).__name__)
        entries.append((path, loader, loader.load(path)))

    type_seen: Counter[str] = Counter()
    dfs: list[pd.DataFrame] = []
    for _path, loader, df in entries:
        sensor_type = _sensor_type(loader)
        type_seen[sensor_type] += 1
        sensor_id = f"{sensor_type}_{type_seen[sensor_type]}"
        dfs.append(_with_sensor_id(df, sensor_id))

    return dfs, display_type_by_path, failed_paths


def is_aux_signal_column(column: str) -> bool:
    return str(column).startswith(SIGNAL_AUX_PREFIX)


def read_synced_signal_csv(path: str | Path) -> pd.DataFrame:
    """Read an exported signal CSV and normalize mixed-precision UTC timestamps."""
    df = pd.read_csv(path)
    if "timestamp_utc" not in df.columns:
        raise ValueError(f"Synchronized signal CSV has no timestamp_utc column: {path}")
    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        format="mixed",
        utc=True,
        errors="raise",
    )
    return df


def _sensor_name_from_path(path: str | Path) -> str:
    parts = Path(path).stem.split("_")
    return "_".join(part.upper() if part.lower() == "ecg" else part.capitalize() for part in parts)


def read_synced_signal_csvs(paths: list[str] | list[Path]) -> Optional[pd.DataFrame]:
    """Load separated sensor CSVs, while accepting the legacy combined format."""
    if not paths:
        return None

    frames = [read_synced_signal_csv(path) for path in paths]
    if len(frames) == 1 and not any(
        column in frames[0].columns for column in SYNCED_PRIMARY_COLUMNS
    ):
        return frames[0]

    combined: Optional[pd.DataFrame] = None
    for path, frame in zip(paths, frames):
        primary = next(
            (column for column in SYNCED_PRIMARY_COLUMNS if column in frame.columns),
            None,
        )
        if primary is None:
            raise ValueError(f"Separated signal CSV has no primary value column: {path}")
        sensor = _sensor_name_from_path(path)
        renamed = frame.rename(columns={primary: sensor}).copy()
        renamed = renamed.rename(columns={
            column: f"{SIGNAL_AUX_PREFIX}{sensor}__{column}"
            for column in renamed.columns
            if column not in {"timestamp_utc", sensor}
        })
        combined = renamed if combined is None else combined.merge(
            renamed, on="timestamp_utc", how="outer"
        )

    return combined.sort_values("timestamp_utc").reset_index(drop=True) if combined is not None else None


def signal_wide_to_device_frames(
    frame: pd.DataFrame,
    primary_column: str,
) -> dict[str, pd.DataFrame]:
    """Convert the app's wide synchronized frame to one native-rate frame per sensor."""
    if "timestamp_utc" not in frame.columns:
        raise ValueError("Signal data missing required column: timestamp_utc")

    sensors = [
        str(column) for column in frame.columns
        if column != "timestamp_utc"
        and column != "averaged"
        and not is_aux_signal_column(str(column))
    ]
    all_auxiliary = [
        str(column) for column in frame.columns if is_aux_signal_column(str(column))
    ]
    outputs: dict[str, pd.DataFrame] = {}
    for sensor in sensors:
        prefix = f"{SIGNAL_AUX_PREFIX}{sensor}__"
        own_auxiliary = [column for column in all_auxiliary if column.startswith(prefix)]
        if own_auxiliary:
            native_rows = frame[own_auxiliary].notna().any(axis=1)
        elif all_auxiliary:
            native_rows = frame[sensor].notna() & ~frame[all_auxiliary].notna().any(axis=1)
        else:
            native_rows = frame[sensor].notna()
        if not native_rows.any():
            native_rows = frame[sensor].notna()

        device = frame.loc[native_rows, ["timestamp_utc", sensor, *own_auxiliary]].copy()
        device = device.rename(columns={sensor: primary_column})
        device = device.rename(columns={
            column: column.removeprefix(prefix) for column in own_auxiliary
        })
        outputs[sensor] = device.reset_index(drop=True)
    return outputs


def write_synced_signal_csvs(
    frame: pd.DataFrame,
    output_directory: str | Path,
    primary_column: str,
) -> list[str]:
    """Write one synchronized CSV per sensor directly in the output directory."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for sensor, device in signal_wide_to_device_frames(frame, primary_column).items():
        numbered_sensor = sensor if re.search(r"_\d+$", sensor) else f"{sensor}_1"
        path = output / f"{numbered_sensor.lower()}.csv"
        device.to_csv(path, index=False)
        paths.append(str(path))
    return paths


def signal_long_to_wide(df: pd.DataFrame, include_average: bool = False) -> pd.DataFrame:
    """Pivot loader output to the wide CSV format used by the app.

    The canonical HR/value series is pivoted into one column per sensor_id.
    Loader-specific columns that share the same timestamps are preserved as
    ``aux__{sensor_id}__{column}`` so they can be exported without being plotted
    as HR traces.
    """
    missing = SIGNAL_CORE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Signal data missing required column(s): {sorted(missing)}")

    wide = df.pivot_table(
        index="timestamp_utc", columns="sensor_id", values="value", aggfunc="first"
    )
    wide.columns.name = None
    wide = wide.sort_index().ffill().bfill()

    hr_columns = list(wide.columns)
    if include_average and hr_columns:
        wide["averaged"] = wide[hr_columns].mean(axis=1)

    aux_cols = [c for c in df.columns if c not in SIGNAL_CORE_COLUMNS]
    if aux_cols:
        aux_frames = []
        for sensor_id, sub in df.groupby("sensor_id", sort=False):
            sensor_aux_cols = [c for c in aux_cols if sub[c].notna().any()]
            if not sensor_aux_cols:
                continue
            aux = sub[["timestamp_utc", *sensor_aux_cols]].copy()
            aux = aux.drop_duplicates(subset=["timestamp_utc"], keep="first")
            aux = aux.set_index("timestamp_utc").sort_index()
            aux = aux.rename(columns={
                col: f"{SIGNAL_AUX_PREFIX}{sensor_id}__{col}" for col in sensor_aux_cols
            })
            aux_frames.append(aux)
        for aux in aux_frames:
            wide = wide.join(aux, how="left")

    return wide.reset_index()
