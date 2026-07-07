from __future__ import annotations

import abc
import importlib
import logging
import pkgutil
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

SIGNAL_CORE_COLUMNS = {"timestamp_utc", "value", "sensor_id"}
SIGNAL_AUX_PREFIX = "aux__"


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
        return _with_sensor_id(loader.load(path), _sensor_type(loader))
    log.warning("No loader found for %s", path)
    return None


def load_signal_files(paths: list[str]) -> tuple[list[pd.DataFrame], dict[str, str], list[str]]:
    """Load signal files and assign type-based sensor IDs.

    A single file of a type is named ``Polar``/``Zephyr``. Multiple files of the
    same type are named ``Polar_1``, ``Polar_2`` or ``Zephyr_1``, ``Zephyr_2`` in
    selection order. Returns ``(dataframes, display_type_by_path, failed_paths)``.
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

    type_counts = Counter(_sensor_type(loader) for _path, loader, _df in entries)
    type_seen: Counter[str] = Counter()
    dfs: list[pd.DataFrame] = []
    for _path, loader, df in entries:
        sensor_type = _sensor_type(loader)
        if type_counts[sensor_type] == 1:
            sensor_id = sensor_type
        else:
            type_seen[sensor_type] += 1
            sensor_id = f"{sensor_type}_{type_seen[sensor_type]}"
        dfs.append(_with_sensor_id(df, sensor_id))

    return dfs, display_type_by_path, failed_paths


def is_aux_signal_column(column: str) -> bool:
    return str(column).startswith(SIGNAL_AUX_PREFIX)


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
