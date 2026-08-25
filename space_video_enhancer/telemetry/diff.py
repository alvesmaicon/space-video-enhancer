"""Flatten nested dataclasses and diff two snapshots -> {"filters.contrast": [50, 62]}."""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float):
        return round(value, 4)
    return value


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        return {prefix.rstrip("."): _plain(obj)}
    flat: dict[str, Any] = {}
    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)
        key = f"{prefix}{field.name}"
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            flat.update(flatten(value, f"{key}."))
        elif isinstance(value, (tuple, list)) and any(dataclasses.is_dataclass(item) for item in value):
            for index, item in enumerate(value):
                flat.update(flatten(item, f"{key}.{index}."))
        else:
            flat[key] = _plain(value)
    return flat


def diff(old: Any, new: Any) -> dict[str, list[Any]]:
    """Changed leaves between two dataclass snapshots as ``{path: [old, new]}``; ``old`` may be ``None``."""
    new_flat = flatten(new)
    if old is None:
        return {key: [None, value] for key, value in new_flat.items()}
    old_flat = flatten(old)
    return {key: [old_flat.get(key), value] for key, value in new_flat.items() if old_flat.get(key) != value}
