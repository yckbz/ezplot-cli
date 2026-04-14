# ezplot

Fast command-line plotting for research data on Linux servers. 

## Install

```bash
python -m pip install .
```

## Get Help

```bash
ezplot -h
ezplot scatter -h
ezplot line -h
ezplot hist -h
ezplot density -h
ezplot heatmap -h
```

If you pass only an input file, `ezplot` defaults to:

```bash
ezplot input.dat
```

which is equivalent to:

```bash
ezplot scatter -xi 0 -yi 1 input.dat
```

If that file does not exist, `ezplot` returns a readable file error instead of an `argparse` usage error or Python traceback.

## Core Syntax

```bash
ezplot <scatter|line|hist|density|heatmap> [options] <input_file>
```

- Output is always a PNG. Default output path is `graph.png`.
- Comment lines are skipped automatically. Default comment prefix is `#`.
- PLUMED-style comment headers such as `#! FIELDS time cv1 cv2` are recognized.
- Delimiters are auto-detected between CSV and whitespace-separated text.
- If axis labels are not given, `ezplot` uses header names when available, otherwise falls back to `X` and `Y`.

## Column Selection

- `-x`, `-y`, `-z` select columns by name.
- `-xi`, `-yi`, `-zi` select columns by zero-based index.
- Column names support exact match first, then unique prefix matching.
- The same axis cannot use both name and index forms at once.
- `scatter`, `line`, and `hist` allow multiple y columns through comma-separated `-y` or `-yi`.
- `density` uses `-x` / `-xi` for 1D, and `-x` plus `-y` for 2D.
- `heatmap` expects exactly one x column, one y column, and one z column.
- If you use column names but the file has no header, the command fails with an error.

## Common Options

- `-o, --output PATH`: output PNG path. Default `graph.png`.
- `--xlabel TEXT`: override x-axis label.
- `--ylabel TEXT`: override y-axis label.
- `--xlim MIN,MAX`: set x-axis range.
- `--ylim MIN,MAX`: set y-axis range.
- `--dpi INT`: output resolution. Default `160`.
- `--figsize W,H`: figure size in inches. Default `6,4`.
- `--comment PREFIX`: comment prefix to skip. Default `#`.

## Command-Specific Options

- `scatter`: `--size`, `--alpha`
- `line`: `--linewidth`, `--markersize`, `--alpha`
- `hist`: `--bins`
- `density`: `--bin WIDTH` for 1D, `--bin DX,DY` for 2D
- `heatmap`: `--bin DX,DY`, `--vmin`, `--vmax`

## Behavior Notes

- `line` sorts by x before drawing.
- `hist` overlays multiple y columns in one plot and uses shared bin edges across all series.
- `density` is histogram-based density, not KDE.
- `density -x col file.dat` draws a 1D density line. `density -x col1 -y col2 file.dat` draws a 2D density heatmap.
- `density` defaults to 100 bins per axis. If `--bin` is given, it is treated as bin width rather than bin count.
- `heatmap` accepts long-table `x,y,z` input, averages duplicate `(x,y)` points, and leaves missing grid cells blank.
- `heatmap --bin DX,DY` bins noisy x/y coordinates onto a zero-aligned `DX * DY` grid before averaging z values inside each bin.
- Existing output files are overwritten by default.

## Examples

```bash
ezplot scatter -x time -y value data.dat
ezplot scatter -xi 0 -yi 1,2 data.dat --xlim 0,100 --ylim -1,1
ezplot line -x z -y rho,cos_phi test1.dat -o line.png
ezplot hist -yi 1,2 --bins 80 data.csv
ezplot density -x value density_1d_demo.csv
ezplot density -x x -y y --bin 0.2,0.2 density_2d_demo.csv
ezplot heatmap -x x -y y -z value grid.csv --vmin 0 --vmax 10
ezplot heatmap -x x -y y -z fes --bin 0.01,0.01 fes.csv
```
