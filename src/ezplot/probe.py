from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .errors import DataFormatError


@dataclass(frozen=True)
class ProbeResult:
    delimiter: str
    header_names: list[str] | None
    inline_header: bool
    column_count: int
    comment_prefix: str


def _split_fields(line: str, delimiter: str) -> list[str]:
    if delimiter == "csv":
        return [part.strip() for part in line.split(",")]
    return [part.strip() for part in re.split(r"\s+", line.strip()) if part.strip()]


def _split_comment_header(line: str, expected_columns: int) -> list[str]:
    normalized = line.strip()

    if "," in normalized:
        tokens = [part.strip() for part in normalized.split(",") if part.strip()]
    else:
        tokens = [part.strip() for part in re.split(r"\s+", normalized) if part.strip()]

    if not tokens:
        return []

    head = tokens[0].lstrip("!")
    if tokens[0] == "!" and len(tokens) > 1 and tokens[1] == "FIELDS":
        field_tokens = tokens[2:]
        if len(field_tokens) == expected_columns:
            return field_tokens
        return []

    if head == "FIELDS":
        field_tokens = tokens[1:]
        if len(field_tokens) == expected_columns:
            return field_tokens
        return []

    tokens = [part.strip() for part in re.split(r"\s+", line.strip()) if part.strip()]
    if len(tokens) == expected_columns:
        return tokens
    return []


def _tokens_are_numeric(tokens: list[str]) -> bool:
    if not tokens:
        return False
    try:
        for token in tokens:
            float(token)
    except ValueError:
        return False
    return True


def probe_file(path: str | Path, comment_prefix: str = "#") -> ProbeResult:
    file_path = Path(path)
    comment_candidates: list[str] = []
    non_comment_lines: list[str] = []

    try:
        with file_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                if stripped.startswith(comment_prefix):
                    body = stripped[len(comment_prefix) :].strip()
                    if body:
                        comment_candidates.append(body)
                    continue
                non_comment_lines.append(stripped)
                if len(non_comment_lines) >= 2:
                    break
    except FileNotFoundError as exc:
        raise DataFormatError(f"Input file not found: {file_path}.") from exc
    except OSError as exc:
        raise DataFormatError(f"Could not read input file {file_path}: {exc}.") from exc

    if not non_comment_lines:
        raise DataFormatError(f"No data rows found in {file_path}.")

    first_line = non_comment_lines[0]
    delimiter = "csv" if "," in first_line else "whitespace"
    first_tokens = _split_fields(first_line, delimiter)
    if not first_tokens:
        raise DataFormatError(f"Unable to parse the first data row in {file_path}.")

    inline_header = not _tokens_are_numeric(first_tokens)
    if inline_header:
        if len(non_comment_lines) >= 2:
            column_count = len(_split_fields(non_comment_lines[1], delimiter))
        else:
            column_count = len(first_tokens)
        inline_header_names = first_tokens
    else:
        column_count = len(first_tokens)
        inline_header_names = None

    comment_header_names = None
    for candidate in comment_candidates:
        tokens = _split_comment_header(candidate, column_count)
        if len(tokens) == column_count and not _tokens_are_numeric(tokens):
            comment_header_names = tokens
            if candidate.lstrip().startswith("! FIELDS") or candidate.lstrip().startswith("!FIELDS"):
                break

    return ProbeResult(
        delimiter=delimiter,
        header_names=comment_header_names or inline_header_names,
        inline_header=inline_header,
        column_count=column_count,
        comment_prefix=comment_prefix,
    )
