from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

from .errors import EzPlotError


@dataclass(frozen=True)
class PlotOptions:
    output_path: Path
    xlabel: str | None
    ylabel: str | None
    xlim: tuple[float, float] | None
    ylim: tuple[float, float] | None
    figsize: tuple[float, float]
    dpi: int


@dataclass(frozen=True)
class HeatmapGrid:
    x_values: np.ndarray
    y_values: np.ndarray
    matrix: np.ndarray


def prepare_line_series(x: np.ndarray, ys: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    sorted_ys = {label: values[order] for label, values in ys.items()}
    return sorted_x, sorted_ys


def build_heatmap_grid(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> HeatmapGrid:
    if x.size == 0 or y.size == 0 or z.size == 0:
        raise EzPlotError("No finite rows remain after filtering NaN/Inf values.")

    order = np.lexsort((y, x))
    x_sorted = x[order]
    y_sorted = y[order]
    z_sorted = z[order]

    agg_x: list[float] = []
    agg_y: list[float] = []
    agg_z: list[float] = []

    current_x = None
    current_y = None
    bucket: list[float] = []

    for x_value, y_value, z_value in zip(x_sorted, y_sorted, z_sorted, strict=True):
        if current_x is None or x_value != current_x or y_value != current_y:
            if bucket:
                agg_x.append(current_x)
                agg_y.append(current_y)
                agg_z.append(float(np.mean(bucket)))
            current_x = float(x_value)
            current_y = float(y_value)
            bucket = [float(z_value)]
        else:
            bucket.append(float(z_value))

    if bucket:
        agg_x.append(current_x)
        agg_y.append(current_y)
        agg_z.append(float(np.mean(bucket)))

    x_values = np.unique(np.asarray(agg_x, dtype=float))
    y_values = np.unique(np.asarray(agg_y, dtype=float))
    matrix = np.full((len(y_values), len(x_values)), np.nan, dtype=float)

    x_index = {value: index for index, value in enumerate(x_values)}
    y_index = {value: index for index, value in enumerate(y_values)}

    for x_value, y_value, z_value in zip(agg_x, agg_y, agg_z, strict=True):
        matrix[y_index[y_value], x_index[x_value]] = z_value

    return HeatmapGrid(x_values=x_values, y_values=y_values, matrix=matrix)


def build_binned_heatmap_grid(x: np.ndarray, y: np.ndarray, z: np.ndarray, *, dx: float, dy: float) -> HeatmapGrid:
    if x.size == 0 or y.size == 0 or z.size == 0:
        raise EzPlotError("No finite rows remain after filtering NaN/Inf values.")
    if dx <= 0 or dy <= 0:
        raise EzPlotError("Heatmap bin sizes must be positive.")

    x_bins = np.floor(x / dx).astype(int)
    y_bins = np.floor(y / dy).astype(int)

    bucket_values: dict[tuple[int, int], list[float]] = {}
    for x_bin, y_bin, z_value in zip(x_bins, y_bins, z, strict=True):
        bucket_values.setdefault((int(x_bin), int(y_bin)), []).append(float(z_value))

    unique_x_bins = np.array(sorted({x_bin for x_bin, _ in bucket_values}), dtype=int)
    unique_y_bins = np.array(sorted({y_bin for _, y_bin in bucket_values}), dtype=int)

    x_index = {value: index for index, value in enumerate(unique_x_bins)}
    y_index = {value: index for index, value in enumerate(unique_y_bins)}

    matrix = np.full((len(unique_y_bins), len(unique_x_bins)), np.nan, dtype=float)
    for (x_bin, y_bin), values in bucket_values.items():
        matrix[y_index[y_bin], x_index[x_bin]] = float(np.mean(values))

    x_values = (unique_x_bins.astype(float) + 0.5) * dx
    y_values = (unique_y_bins.astype(float) + 0.5) * dy
    return HeatmapGrid(x_values=x_values, y_values=y_values, matrix=matrix)


def _resolve_limits(values: np.ndarray, override: tuple[float, float] | None) -> tuple[float, float]:
    if override is not None:
        return override
    if values.size == 0:
        raise EzPlotError("No finite rows remain after filtering NaN/Inf values.")
    min_value = float(np.nanmin(values))
    max_value = float(np.nanmax(values))
    if min_value == max_value:
        return min_value - 0.5, max_value + 0.5
    return min_value, max_value


def _centers_to_edges(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        raise EzPlotError("No finite rows remain after filtering NaN/Inf values.")
    if values.size == 1:
        value = float(values[0])
        return np.array([value - 0.5, value + 0.5], dtype=float)
    midpoints = (values[:-1] + values[1:]) / 2.0
    first = values[0] - (midpoints[0] - values[0])
    last = values[-1] + (values[-1] - midpoints[-1])
    return np.concatenate(([first], midpoints, [last]))


def _apply_axes_style(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    ax.grid(False)


def _finalize_figure(fig: plt.Figure, options: PlotOptions) -> None:
    options.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(options.output_path, dpi=options.dpi)
    plt.close(fig)


def _compute_histogram_bin_edges(series: dict[str, np.ndarray], bins: int) -> np.ndarray:
    if not series or any(values.size == 0 for values in series.values()):
        raise EzPlotError("No finite rows remain after filtering NaN/Inf values.")
    all_values = np.concatenate(list(series.values()))
    return np.histogram_bin_edges(all_values, bins=bins)


def _compute_aligned_bin_edges(values: np.ndarray, *, bin_size: float) -> np.ndarray:
    if values.size == 0:
        raise EzPlotError("No finite rows remain after filtering NaN/Inf values.")
    if bin_size <= 0:
        raise EzPlotError("Bin sizes must be positive.")

    min_value = float(np.min(values))
    max_value = float(np.max(values))
    start = np.floor(min_value / bin_size) * bin_size
    stop = np.ceil(max_value / bin_size) * bin_size
    if np.isclose(start, stop):
        stop = start + bin_size
    n_bins = max(1, int(np.ceil((stop - start) / bin_size)))
    return start + np.arange(n_bins + 1, dtype=float) * bin_size


def build_density_lines(
    series: dict[str, np.ndarray], *, bin_size: float | None, bins: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not series or any(values.size == 0 for values in series.values()):
        raise EzPlotError("No finite rows remain after filtering NaN/Inf values.")

    all_values = np.concatenate(list(series.values()))
    if bin_size is None:
        edges = np.histogram_bin_edges(all_values, bins=bins)
    else:
        edges = _compute_aligned_bin_edges(all_values, bin_size=bin_size)

    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)
    densities = {}
    for label, values in series.items():
        counts, _ = np.histogram(values, bins=edges)
        densities[label] = counts.astype(float) / (values.size * widths)
    return centers, densities


def build_density_heatmap_grid(
    x: np.ndarray, y: np.ndarray, *, bin_size: tuple[float, float] | None, bins: int
) -> HeatmapGrid:
    if x.size == 0 or y.size == 0:
        raise EzPlotError("No finite rows remain after filtering NaN/Inf values.")

    if bin_size is None:
        x_edges = np.histogram_bin_edges(x, bins=bins)
        y_edges = np.histogram_bin_edges(y, bins=bins)
    else:
        x_edges = _compute_aligned_bin_edges(x, bin_size=bin_size[0])
        y_edges = _compute_aligned_bin_edges(y, bin_size=bin_size[1])

    matrix, x_edges, y_edges = np.histogram2d(x, y, bins=[x_edges, y_edges], density=True)
    x_values = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_values = (y_edges[:-1] + y_edges[1:]) / 2.0
    return HeatmapGrid(x_values=x_values, y_values=y_values, matrix=matrix.T)


def plot_scatter(
    x: np.ndarray,
    series: dict[str, np.ndarray],
    *,
    options: PlotOptions,
    size: float,
    alpha: float,
) -> None:
    if x.size == 0 or any(values.size == 0 for values in series.values()):
        raise EzPlotError("No finite rows remain after filtering NaN/Inf values.")
    fig, ax = plt.subplots(figsize=options.figsize)
    colors = plt.get_cmap("tab10").colors
    for index, (label, values) in enumerate(series.items()):
        ax.scatter(x, values, s=size, alpha=alpha, color=colors[index % len(colors)], label=label)

    ax.set_xlabel(options.xlabel or "X")
    ax.set_ylabel(options.ylabel or ("Y" if len(series) > 1 else next(iter(series))))
    ax.set_xlim(*_resolve_limits(x, options.xlim))
    all_y = np.concatenate(list(series.values()))
    ax.set_ylim(*_resolve_limits(all_y, options.ylim))
    if len(series) > 1:
        ax.legend(frameon=False)
    _apply_axes_style(ax)
    _finalize_figure(fig, options)


def plot_line(
    x: np.ndarray,
    series: dict[str, np.ndarray],
    *,
    options: PlotOptions,
    linewidth: float,
    markersize: float,
    alpha: float,
) -> None:
    if x.size == 0 or any(values.size == 0 for values in series.values()):
        raise EzPlotError("No finite rows remain after filtering NaN/Inf values.")
    sorted_x, sorted_series = prepare_line_series(x, series)
    fig, ax = plt.subplots(figsize=options.figsize)
    colors = plt.get_cmap("tab10").colors
    for index, (label, values) in enumerate(sorted_series.items()):
        ax.plot(
            sorted_x,
            values,
            marker="o",
            linewidth=linewidth,
            markersize=markersize,
            alpha=alpha,
            color=colors[index % len(colors)],
            label=label,
        )

    ax.set_xlabel(options.xlabel or "X")
    ax.set_ylabel(options.ylabel or ("Y" if len(series) > 1 else next(iter(series))))
    ax.set_xlim(*_resolve_limits(sorted_x, options.xlim))
    all_y = np.concatenate(list(sorted_series.values()))
    ax.set_ylim(*_resolve_limits(all_y, options.ylim))
    if len(sorted_series) > 1:
        ax.legend(frameon=False)
    _apply_axes_style(ax)
    _finalize_figure(fig, options)


def plot_hist(
    series: dict[str, np.ndarray],
    *,
    options: PlotOptions,
    bins: int,
) -> None:
    if not series or any(values.size == 0 for values in series.values()):
        raise EzPlotError("No finite rows remain after filtering NaN/Inf values.")
    fig, ax = plt.subplots(figsize=options.figsize)
    colors = plt.get_cmap("tab10").colors
    bin_edges = _compute_histogram_bin_edges(series, bins)
    for index, (label, values) in enumerate(series.items()):
        ax.hist(
            values,
            bins=bin_edges,
            alpha=0.45,
            color=colors[index % len(colors)],
            edgecolor=colors[index % len(colors)],
            linewidth=1.0,
            label=label,
        )

    if len(series) == 1:
        default_xlabel = next(iter(series))
    else:
        default_xlabel = "Value"
    ax.set_xlabel(options.xlabel or default_xlabel)
    ax.set_ylabel(options.ylabel or "Count")
    all_values = np.concatenate(list(series.values()))
    ax.set_xlim(*_resolve_limits(all_values, options.xlim))
    if options.ylim is not None:
        ax.set_ylim(*options.ylim)
    if len(series) > 1:
        ax.legend(frameon=False)
    _apply_axes_style(ax)
    _finalize_figure(fig, options)


def plot_heatmap(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    options: PlotOptions,
    vmin: float | None,
    vmax: float | None,
    bin_size: tuple[float, float] | None = None,
) -> None:
    if x.size == 0 or y.size == 0 or z.size == 0:
        raise EzPlotError("No finite rows remain after filtering NaN/Inf values.")
    if bin_size is None:
        grid = build_heatmap_grid(x, y, z)
    else:
        grid = build_binned_heatmap_grid(x, y, z, dx=bin_size[0], dy=bin_size[1])
    x_edges = _centers_to_edges(grid.x_values)
    y_edges = _centers_to_edges(grid.y_values)

    fig, ax = plt.subplots(figsize=options.figsize)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(grid.matrix),
        cmap="cividis",
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(mesh, ax=ax)
    ax.set_xlabel(options.xlabel or "X")
    ax.set_ylabel(options.ylabel or "Y")
    ax.set_xlim(*_resolve_limits(grid.x_values, options.xlim))
    ax.set_ylim(*_resolve_limits(grid.y_values, options.ylim))
    _apply_axes_style(ax)
    _finalize_figure(fig, options)


def plot_density_lines(
    series: dict[str, np.ndarray],
    *,
    options: PlotOptions,
    bin_size: float | None,
    bins: int,
) -> None:
    if not series or any(values.size == 0 for values in series.values()):
        raise EzPlotError("No finite rows remain after filtering NaN/Inf values.")

    centers, densities = build_density_lines(series, bin_size=bin_size, bins=bins)
    fig, ax = plt.subplots(figsize=options.figsize)
    colors = plt.get_cmap("tab10").colors
    for index, (label, values) in enumerate(densities.items()):
        ax.plot(centers, values, linewidth=1.4, color=colors[index % len(colors)], label=label)

    if len(series) == 1:
        default_xlabel = next(iter(series))
    else:
        default_xlabel = "Value"
    ax.set_xlabel(options.xlabel or default_xlabel)
    ax.set_ylabel(options.ylabel or "Density")
    ax.set_xlim(*_resolve_limits(centers, options.xlim))
    all_density_values = np.concatenate(list(densities.values()))
    ax.set_ylim(*_resolve_limits(all_density_values, options.ylim))
    if len(series) > 1:
        ax.legend(frameon=False)
    _apply_axes_style(ax)
    _finalize_figure(fig, options)


def plot_density_heatmap(
    x: np.ndarray,
    y: np.ndarray,
    *,
    options: PlotOptions,
    bin_size: tuple[float, float] | None,
    bins: int,
) -> None:
    if x.size == 0 or y.size == 0:
        raise EzPlotError("No finite rows remain after filtering NaN/Inf values.")

    grid = build_density_heatmap_grid(x, y, bin_size=bin_size, bins=bins)
    x_edges = _centers_to_edges(grid.x_values)
    y_edges = _centers_to_edges(grid.y_values)

    fig, ax = plt.subplots(figsize=options.figsize)
    mesh = ax.pcolormesh(x_edges, y_edges, grid.matrix, cmap="cividis", shading="auto")
    fig.colorbar(mesh, ax=ax)
    ax.set_xlabel(options.xlabel or "X")
    ax.set_ylabel(options.ylabel or "Y")
    ax.set_xlim(options.xlim if options.xlim is not None else (float(x_edges[0]), float(x_edges[-1])))
    ax.set_ylim(options.ylim if options.ylim is not None else (float(y_edges[0]), float(y_edges[-1])))
    _apply_axes_style(ax)
    _finalize_figure(fig, options)
