from __future__ import annotations

import abc
import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


class SignalLoader(abc.ABC):
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


def load_signal(path: str) -> Optional[pd.DataFrame]:
    _discover_loaders()
    for loader in _LOADERS:
        if loader.can_load(path):
            log.info("Loading %s with %s", path, type(loader).__name__)
            return loader.load(path)
    log.warning("No loader found for %s", path)
    return None
