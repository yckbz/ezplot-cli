from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from .errors import DataFormatError, DataSelectionError
from .probe import ProbeResult

try:
    import polars as pl
except ModuleNotFoundError:  # pragma: no cover - exercised when polars is absent.
    pl = None


@dataclass(frozen=True)
class ColumnSelection:
    indices: list[int]
    labels: list[str]


def _split_csv_values(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_indices(raw: str) -> list[int]:
    values = []
    for token in _split_csv_values(raw):
        try:
            value = int(token)
        except ValueError as exc:
            raise DataSelectionError(f"Invalid column index {token!r}.") from exc
        if value < 0:
            raise DataSelectionError(f"Column index must be >= 0, got {value}.")
        values.append(value)
    if not values:
        raise DataSelectionError("No column indices were provided.")
    return values


def _match_header_token(headers: list[str], token: str) -> tuple[int, str]:
    if token in headers:
        index = headers.index(token)
        return index, headers[index]

    matches = [(index, name) for index, name in enumerate(headers) if name.startswith(token)]
    if not matches:
        raise DataSelectionError(f"Could not resolve column name {token!r}.")
    if len(matches) > 1:
        joined = ", ".join(name for _, name in matches)
        raise DataSelectionError(f"Column selector {token!r} is ambiguous: {joined}.")
    return matches[0]


def resolve_column_selection(
    headers: list[str] | None,
    *,
    names: str | None,
    indices: str | None,
    axis_name: str,
    column_count: int | None = None,
    multiple: bool = True,
) -> ColumnSelection:
    if bool(names) == bool(indices):
        raise DataSelectionError(f"Exactly one of -{axis_name} or -{axis_name}i must be provided.")

    if names:
        if headers is None:
            raise DataSelectionError(
                f"Column names cannot be used for {axis_name} because the file has no header."
            )
        resolved = [_match_header_token(headers, token) for token in _split_csv_values(names)]
        indices_list = [index for index, _ in resolved]
        labels = [label for _, label in resolved]
    else:
        indices_list = _parse_indices(indices or "")
        if column_count is not None:
            invalid = [value for value in indices_list if value >= column_count]
            if invalid:
                raise DataSelectionError(
                    f"Column index {invalid[0]} is out of range for {column_count} columns."
                )
        labels = [headers[index] if headers else f"col{index}" for index in indices_list]

    if not multiple and len(indices_list) != 1:
        raise DataSelectionError(f"{axis_name} accepts exactly one column.")

    return ColumnSelection(indices=indices_list, labels=labels)


def _split_fields(line: str, delimiter: str) -> list[str]:
    if delimiter == "csv":
        return [part.strip() for part in line.split(",")]
    return [part.strip() for part in re.split(r"\s+", line.strip()) if part.strip()]


def _load_with_polars(path: Path, probe: ProbeResult, indices: list[int]) -> dict[int, np.ndarray]:
    if pl is None:
        raise ModuleNotFoundError("polars")

    frame = pl.read_csv(
        path,
        has_header=probe.inline_header,
        comment_prefix=probe.comment_prefix,
        columns=indices,
    )
    return {
        original_index: frame[:, position].to_numpy().astype(float, copy=False)
        for position, original_index in enumerate(indices)
    }


def _load_with_streaming(path: Path, probe: ProbeResult, indices: list[int]) -> dict[int, np.ndarray]:
    values = {index: [] for index in indices}
    skipped_inline_header = False

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith(probe.comment_prefix):
                continue
            if probe.inline_header and not skipped_inline_header:
                skipped_inline_header = True
                continue

            fields = _split_fields(stripped, probe.delimiter)
            if len(fields) != probe.column_count:
                raise DataFormatError(
                    f"Line {line_number} has {len(fields)} columns; expected {probe.column_count}."
                )

            for index in indices:
                try:
                    values[index].append(float(fields[index]))
                except ValueError as exc:
                    raise DataFormatError(
                        f"Line {line_number}, column {index} is not numeric: {fields[index]!r}."
                    ) from exc

    return {index: np.asarray(column, dtype=float) for index, column in values.items()}


def load_selected_columns(
    path: str | Path, probe: ProbeResult, indices: Iterable[int]
) -> dict[int, np.ndarray]:
    path_obj = Path(path)
    ordered_indices = list(dict.fromkeys(indices))
    if probe.delimiter == "csv":
        try:
            return _load_with_polars(path_obj, probe, ordered_indices)
        except ModuleNotFoundError:
            return _load_with_streaming(path_obj, probe, ordered_indices)
    return _load_with_streaming(path_obj, probe, ordered_indices)


def filter_finite_rows(columns: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int]:
    if not columns:
        return {}, 0
    arrays = list(columns.values())
    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    dropped = int(mask.size - mask.sum())
    if dropped == 0:
        return columns, 0
    return {name: array[mask] for name, array in columns.items()}, dropped
