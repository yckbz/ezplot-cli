from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .errors import EzPlotError
from .plots import (
    PlotOptions,
    plot_density_heatmap,
    plot_density_lines,
    plot_heatmap,
    plot_hist,
    plot_line,
    plot_scatter,
)
from .probe import probe_file
from .reader import filter_finite_rows, load_selected_columns, resolve_column_selection


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Compact formatter with defaults and preserved line breaks."""


def _parse_pair(raw: str, option_name: str) -> tuple[float, float]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 2:
        raise EzPlotError(f"{option_name} expects values in the form min,max.")
    return float(parts[0]), float(parts[1])


def _parse_figsize(raw: str) -> tuple[float, float]:
    return _parse_positive_pair(raw, "--figsize")


def _parse_positive_pair(raw: str, option_name: str) -> tuple[float, float]:
    first, second = _parse_pair(raw, option_name)
    if first <= 0 or second <= 0:
        raise EzPlotError(f"{option_name} expects positive values.")
    return first, second


def _parse_positive_float(raw: str, option_name: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise EzPlotError(f"{option_name} expects a positive value.") from exc
    if value <= 0:
        raise EzPlotError(f"{option_name} expects a positive value.")
    return value


def _normalize_argv(argv: list[str] | None) -> list[str]:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    raw_args = _normalize_attached_option_values(raw_args)
    if len(raw_args) == 1:
        candidate = raw_args[0]
        if candidate not in {"-h", "--help", "scatter", "line", "hist", "heatmap", "density"} and not candidate.startswith("-"):
            return ["scatter", "-xi", "0", "-yi", "1", candidate]
    return raw_args


def _normalize_attached_option_values(raw_args: list[str]) -> list[str]:
    attachable_options = {"--xlim", "--ylim", "--figsize", "--bin"}
    normalized: list[str] = []
    index = 0
    while index < len(raw_args):
        token = raw_args[index]
        if token in attachable_options and index + 1 < len(raw_args):
            value = raw_args[index + 1]
            if value.startswith("-") and not value.startswith("--"):
                normalized.append(f"{token}={value}")
                index += 2
                continue
        normalized.append(token)
        index += 1
    return normalized


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ezplot",
        description="Fast CLI plotting for research data files.",
        epilog="Run 'ezplot <command> -h' for command-specific help.",
        formatter_class=HelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, title="commands", metavar="command")

    def add_common_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("input_file", help="Input data file.")
        command.add_argument("-o", "--output", default="graph.png", help="Output PNG path.")
        command.add_argument("--xlabel", help="Override x-axis label.")
        command.add_argument("--ylabel", help="Override y-axis label.")
        command.add_argument("--xlim", help="Set x-axis range as min,max.")
        command.add_argument("--ylim", help="Set y-axis range as min,max.")
        command.add_argument("--dpi", type=int, default=160, help="PNG resolution.")
        command.add_argument("--figsize", default="6,4", help="Figure size in inches as width,height.")
        command.add_argument(
            "--comment",
            default="#",
            help="Skip lines whose first non-space character matches this prefix.",
        )

    scatter = subparsers.add_parser(
        "scatter",
        help="Scatter plot with one x column and one or more y columns.",
        description=(
            "Draw a scatter plot.\n"
            "Choose x with -x or -xi, and choose one or more y columns with -y or -yi."
        ),
        formatter_class=HelpFormatter,
    )
    scatter.add_argument("-x", help="Use this x column name.")
    scatter.add_argument("-xi", help="Use this zero-based x column index.")
    scatter.add_argument("-y", help="Use one or more y column names, comma-separated.")
    scatter.add_argument("-yi", help="Use one or more zero-based y column indices, comma-separated.")
    scatter.add_argument("--size", type=float, default=12.0, help="Marker size.")
    scatter.add_argument("--alpha", type=float, default=0.75, help="Marker opacity.")
    add_common_arguments(scatter)

    line = subparsers.add_parser(
        "line",
        help="Line plot with points, sorted by x before drawing.",
        description=(
            "Draw a point-line plot.\n"
            "Choose x with -x or -xi, and choose one or more y columns with -y or -yi."
        ),
        formatter_class=HelpFormatter,
    )
    line.add_argument("-x", help="Use this x column name.")
    line.add_argument("-xi", help="Use this zero-based x column index.")
    line.add_argument("-y", help="Use one or more y column names, comma-separated.")
    line.add_argument("-yi", help="Use one or more zero-based y column indices, comma-separated.")
    line.add_argument("--linewidth", type=float, default=1.2, help="Line width.")
    line.add_argument("--markersize", type=float, default=3.0, help="Point marker size.")
    line.add_argument("--alpha", type=float, default=0.9, help="Line and marker opacity.")
    add_common_arguments(line)

    hist = subparsers.add_parser(
        "hist",
        help="Histogram for one or more y columns.",
        description="Draw a histogram. Choose one or more columns with -y or -yi.",
        formatter_class=HelpFormatter,
    )
    hist.add_argument("-y", help="Use one or more y column names, comma-separated.")
    hist.add_argument("-yi", help="Use one or more zero-based y column indices, comma-separated.")
    hist.add_argument("--bins", type=int, default=50, help="Number of histogram bins.")
    add_common_arguments(hist)

    density = subparsers.add_parser(
        "density",
        help="Density plot from one x column or x/y pairs.",
        description=(
            "Draw a 1D or 2D density plot.\n"
            "Use -x for 1D. Add -y for 2D."
        ),
        formatter_class=HelpFormatter,
    )
    density.add_argument("-x", help="Use one or more x column names, comma-separated.")
    density.add_argument("-xi", help="Use one or more zero-based x column indices, comma-separated.")
    density.add_argument("-y", help="Use this y column name for 2D density.")
    density.add_argument("-yi", help="Use this zero-based y column index for 2D density.")
    density.add_argument("--bin", help="Use one positive value for 1D or dx,dy for 2D.")
    add_common_arguments(density)

    heatmap = subparsers.add_parser(
        "heatmap",
        help="Heatmap from x, y, z columns.",
        description=(
            "Draw a heatmap from x, y, z columns.\n"
            "Choose each axis with either the name form or the zero-based index form."
        ),
        formatter_class=HelpFormatter,
    )
    heatmap.add_argument("-x", help="Use this x column name.")
    heatmap.add_argument("-xi", help="Use this zero-based x column index.")
    heatmap.add_argument("-y", help="Use this y column name.")
    heatmap.add_argument("-yi", help="Use this zero-based y column index.")
    heatmap.add_argument("-z", help="Use this z column name.")
    heatmap.add_argument("-zi", help="Use this zero-based z column index.")
    heatmap.add_argument("--bin", help="Bin noisy x,y coordinates as dx,dy.")
    heatmap.add_argument("--vmin", type=float, help="Set the minimum heatmap color value.")
    heatmap.add_argument("--vmax", type=float, help="Set the maximum heatmap color value.")
    add_common_arguments(heatmap)

    return parser


def _build_plot_options(args: argparse.Namespace, *, default_xlabel: str, default_ylabel: str) -> PlotOptions:
    return PlotOptions(
        output_path=Path(args.output),
        xlabel=args.xlabel or default_xlabel,
        ylabel=args.ylabel or default_ylabel,
        xlim=_parse_pair(args.xlim, "--xlim") if args.xlim else None,
        ylim=_parse_pair(args.ylim, "--ylim") if args.ylim else None,
        figsize=_parse_figsize(args.figsize),
        dpi=args.dpi,
    )


def _warn_dropped_rows(dropped: int) -> None:
    if dropped:
        print(f"Warning: dropped {dropped} rows with NaN or infinite values.", file=sys.stderr)


def _require_remaining_rows(columns: dict[str, object]) -> None:
    if not columns:
        raise EzPlotError("No finite rows remain after filtering NaN/Inf values.")
    first_column = next(iter(columns.values()))
    if getattr(first_column, "size", 0) == 0:
        raise EzPlotError("No finite rows remain after filtering NaN/Inf values.")


def _handle_scatter_or_line(args: argparse.Namespace, *, mode: str) -> int:
    probe = probe_file(args.input_file, comment_prefix=args.comment)
    if args.x is None and args.xi == "0" and args.y is None and args.yi == "1" and probe.column_count < 2:
        raise EzPlotError("Default scatter mode requires at least two columns (x=0, y=1).")
    x_selection = resolve_column_selection(
        probe.header_names,
        names=args.x,
        indices=args.xi,
        axis_name="x",
        column_count=probe.column_count,
        multiple=False,
    )
    y_selection = resolve_column_selection(
        probe.header_names,
        names=args.y,
        indices=args.yi,
        axis_name="y",
        column_count=probe.column_count,
        multiple=True,
    )
    all_indices = x_selection.indices + y_selection.indices
    raw_columns = load_selected_columns(args.input_file, probe, all_indices)

    keyed_columns = {"x": raw_columns[x_selection.indices[0]]}
    for index, label in zip(y_selection.indices, y_selection.labels, strict=True):
        keyed_columns[label] = raw_columns[index]
    filtered_columns, dropped = filter_finite_rows(keyed_columns)
    _warn_dropped_rows(dropped)
    _require_remaining_rows(filtered_columns)

    x_values = filtered_columns["x"]
    series = {label: filtered_columns[label] for label in y_selection.labels}
    default_xlabel = x_selection.labels[0] if probe.header_names else "X"
    default_ylabel = y_selection.labels[0] if probe.header_names and len(series) == 1 else "Y"
    options = _build_plot_options(args, default_xlabel=default_xlabel, default_ylabel=default_ylabel)

    if mode == "scatter":
        plot_scatter(x_values, series, options=options, size=args.size, alpha=args.alpha)
    else:
        plot_line(
            x_values,
            series,
            options=options,
            linewidth=args.linewidth,
            markersize=args.markersize,
            alpha=args.alpha,
        )
    return 0


def _handle_hist(args: argparse.Namespace) -> int:
    probe = probe_file(args.input_file, comment_prefix=args.comment)
    y_selection = resolve_column_selection(
        probe.header_names,
        names=args.y,
        indices=args.yi,
        axis_name="y",
        column_count=probe.column_count,
        multiple=True,
    )
    raw_columns = load_selected_columns(args.input_file, probe, y_selection.indices)
    keyed_columns = {label: raw_columns[index] for index, label in zip(y_selection.indices, y_selection.labels, strict=True)}
    filtered_columns, dropped = filter_finite_rows(keyed_columns)
    _warn_dropped_rows(dropped)
    _require_remaining_rows(filtered_columns)

    if probe.header_names and len(y_selection.labels) == 1:
        default_xlabel = y_selection.labels[0]
    elif probe.header_names:
        default_xlabel = "Value"
    else:
        default_xlabel = "X"
    options = _build_plot_options(args, default_xlabel=default_xlabel, default_ylabel="Count")
    plot_hist(filtered_columns, options=options, bins=args.bins)
    return 0


def _handle_heatmap(args: argparse.Namespace) -> int:
    probe = probe_file(args.input_file, comment_prefix=args.comment)
    x_selection = resolve_column_selection(
        probe.header_names,
        names=args.x,
        indices=args.xi,
        axis_name="x",
        column_count=probe.column_count,
        multiple=False,
    )
    y_selection = resolve_column_selection(
        probe.header_names,
        names=args.y,
        indices=args.yi,
        axis_name="y",
        column_count=probe.column_count,
        multiple=False,
    )
    z_selection = resolve_column_selection(
        probe.header_names,
        names=args.z,
        indices=args.zi,
        axis_name="z",
        column_count=probe.column_count,
        multiple=False,
    )
    all_indices = x_selection.indices + y_selection.indices + z_selection.indices
    raw_columns = load_selected_columns(args.input_file, probe, all_indices)

    keyed_columns = {
        "x": raw_columns[x_selection.indices[0]],
        "y": raw_columns[y_selection.indices[0]],
        "z": raw_columns[z_selection.indices[0]],
    }
    filtered_columns, dropped = filter_finite_rows(keyed_columns)
    _warn_dropped_rows(dropped)
    _require_remaining_rows(filtered_columns)

    default_xlabel = x_selection.labels[0] if probe.header_names else "X"
    default_ylabel = y_selection.labels[0] if probe.header_names else "Y"
    options = _build_plot_options(args, default_xlabel=default_xlabel, default_ylabel=default_ylabel)
    bin_size = _parse_positive_pair(args.bin, "--bin") if args.bin else None
    plot_heatmap(
        filtered_columns["x"],
        filtered_columns["y"],
        filtered_columns["z"],
        options=options,
        vmin=args.vmin,
        vmax=args.vmax,
        bin_size=bin_size,
    )
    return 0


def _handle_density(args: argparse.Namespace) -> int:
    probe = probe_file(args.input_file, comment_prefix=args.comment)
    is_2d = bool(args.y or args.yi)
    x_selection = resolve_column_selection(
        probe.header_names,
        names=args.x,
        indices=args.xi,
        axis_name="x",
        column_count=probe.column_count,
        multiple=not is_2d,
    )

    if not is_2d:
        raw_columns = load_selected_columns(args.input_file, probe, x_selection.indices)
        keyed_columns = {label: raw_columns[index] for index, label in zip(x_selection.indices, x_selection.labels, strict=True)}
        filtered_columns, dropped = filter_finite_rows(keyed_columns)
        _warn_dropped_rows(dropped)
        _require_remaining_rows(filtered_columns)

        if args.bin:
            if "," in args.bin:
                raise EzPlotError("--bin expects a single positive value for 1D density.")
            bin_size = _parse_positive_float(args.bin, "--bin")
        else:
            bin_size = None

        if probe.header_names and len(x_selection.labels) == 1:
            default_xlabel = x_selection.labels[0]
        elif len(x_selection.labels) > 1:
            default_xlabel = "Value"
        else:
            default_xlabel = "X"
        options = _build_plot_options(args, default_xlabel=default_xlabel, default_ylabel="Density")
        plot_density_lines(filtered_columns, options=options, bin_size=bin_size, bins=100)
        return 0

    y_selection = resolve_column_selection(
        probe.header_names,
        names=args.y,
        indices=args.yi,
        axis_name="y",
        column_count=probe.column_count,
        multiple=False,
    )
    raw_columns = load_selected_columns(args.input_file, probe, x_selection.indices + y_selection.indices)
    keyed_columns = {
        "x": raw_columns[x_selection.indices[0]],
        "y": raw_columns[y_selection.indices[0]],
    }
    filtered_columns, dropped = filter_finite_rows(keyed_columns)
    _warn_dropped_rows(dropped)
    _require_remaining_rows(filtered_columns)

    if args.bin:
        if "," not in args.bin:
            raise EzPlotError("--bin expects two positive values for 2D density.")
        bin_size = _parse_positive_pair(args.bin, "--bin")
    else:
        bin_size = None

    default_xlabel = x_selection.labels[0] if probe.header_names else "X"
    default_ylabel = y_selection.labels[0] if probe.header_names else "Y"
    options = _build_plot_options(args, default_xlabel=default_xlabel, default_ylabel=default_ylabel)
    plot_density_heatmap(filtered_columns["x"], filtered_columns["y"], options=options, bin_size=bin_size, bins=100)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _base_parser()
    args = parser.parse_args(_normalize_argv(argv))
    try:
        if args.command == "scatter":
            return _handle_scatter_or_line(args, mode="scatter")
        if args.command == "line":
            return _handle_scatter_or_line(args, mode="line")
        if args.command == "hist":
            return _handle_hist(args)
        if args.command == "density":
            return _handle_density(args)
        if args.command == "heatmap":
            return _handle_heatmap(args)
        raise EzPlotError(f"Unsupported command {args.command!r}.")
    except EzPlotError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
