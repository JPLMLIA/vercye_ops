"""Multi-year LAI report PDF for one aggregation level.

Reads `agg_lai_timeseries_{level_name}_{study}_{year}_{timepoint}.csv` files
produced by `aggregate_lai_timeseries_per_level.py`, so the PDF, the per-level
CSVs, and the interactive map's per-level chart all come from the same
aggregation pass.
"""
import os
import re
from collections import defaultdict
from glob import glob

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def get_axis(axes, row, col, total_rows, n_cols):
    """Helper function to safely get the correct axis from the axes array"""
    if total_rows == 1 and n_cols == 1:
        return axes
    elif total_rows == 1:
        return axes[col]
    elif n_cols == 1:
        return axes[row]
    else:
        return axes[row, col]


def _continuous_doy(dates):
    """Convert dates to continuous DOY values, handling cross-year seasons.

    For cross-year seasons (e.g. Sept→Mar), early-year DOY values get
    shifted by +365 so the plot draws a continuous line instead of
    wrapping back from 365 to 1.
    """
    months = [d.month for d in dates]
    has_late = any(m >= 7 for m in months)
    has_early = any(m <= 6 for m in months)
    cross_year = has_late and has_early

    doys = []
    for d in dates:
        doy = d.timetuple().tm_yday
        if cross_year and d.month <= 6:
            doy += 365
        doys.append(doy)
    return doys, cross_year


def _month_ticks(cross_year):
    """Return (month_number, doy) pairs for x-axis tick labels."""
    if cross_year:
        return [
            (7, 182),
            (8, 213),
            (9, 244),
            (10, 274),
            (11, 305),
            (12, 335),
            (1, 366),
            (2, 397),
            (3, 425),
            (4, 456),
            (5, 486),
            (6, 517),
        ]
    else:
        from datetime import datetime

        return [(m, datetime(2000, m, 1).timetuple().tm_yday) for m in range(1, 13)]


def _discover_level_csvs(basedir, level_name):
    """Find every per-(year,timepoint) aggregated LAI CSV for this level.

    Returns dict {timepoint: {year: csv_path}}.
    """
    pattern = re.compile(
        rf"^agg_lai_timeseries_{re.escape(level_name)}_.+_(?P<year>[^_]+)_(?P<timepoint>[^_]+)\.csv$"
    )
    found = defaultdict(dict)
    for year in sorted(os.listdir(basedir)):
        year_path = os.path.join(basedir, year)
        if not os.path.isdir(year_path):
            continue
        for timepoint in sorted(os.listdir(year_path)):
            tp_path = os.path.join(year_path, timepoint)
            if not os.path.isdir(tp_path):
                continue
            for fname in os.listdir(tp_path):
                m = pattern.match(fname)
                if m and m.group("year") == year and m.group("timepoint") == timepoint:
                    found[timepoint][year] = os.path.join(tp_path, fname)
    return found


def _placeholder_pdf(out_path, level_name):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.text(
        0.5,
        0.5,
        f"No aggregated LAI timeseries CSV found for level '{level_name}'.\n\n"
        "Expected files: agg_lai_timeseries_{level}_{study}_{year}_{timepoint}.csv",
        ha="center",
        va="center",
        wrap=True,
    )
    ax.axis("off")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_agg_plots(basedir, out_path, level_name, lai_variants):
    """One PDF page section per timepoint, subplots per admin region.

    `lai_variants` is a list of (label, column_name) pairs. Each variant is
    drawn in its own subplot row directly below the previous one for the same
    admin unit (smoothed on top, unsmoothed below) so years can stay overlaid
    inside each subplot without crowding.
    """
    csvs_by_tp = _discover_level_csvs(basedir, level_name)
    if not csvs_by_tp:
        print(f"No aggregated LAI CSVs found for level '{level_name}' under {basedir}; writing placeholder.")
        _placeholder_pdf(out_path, level_name)
        return

    # Load every CSV, accumulate union of admin units across all (timepoint, year).
    # Drop variants that no CSV contains so the layout stays compact.
    loaded = {}  # (timepoint, year) -> DataFrame
    all_admin_names = set()
    all_years = set()
    columns_seen = set()
    for timepoint, year_to_path in csvs_by_tp.items():
        for year, path in year_to_path.items():
            df = pd.read_csv(path)
            if df.empty:
                continue
            df["Date_parsed"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
            df = df.dropna(subset=["Date_parsed"]).sort_values("Date_parsed")
            loaded[(timepoint, year)] = df
            all_admin_names.update(df["region"].unique())
            all_years.add(year)
            columns_seen.update(df.columns)

    lai_variants = [(lbl, col) for lbl, col in lai_variants if col in columns_seen]
    if not loaded or not lai_variants:
        print(f"No usable data in CSVs for level '{level_name}'; placeholder.")
        _placeholder_pdf(out_path, level_name)
        return

    all_timepoints = sorted(csvs_by_tp.keys())
    all_admin_names = sorted(all_admin_names)
    all_years = sorted(all_years)
    n_variants = len(lai_variants)

    print(
        f"Found {len(all_timepoints)} timepoints, {len(all_admin_names)} admin units, "
        f"{len(all_years)} years, {n_variants} LAI variant(s) to plot: {[v[0] for v in lai_variants]}"
    )

    n_admin = len(all_admin_names)
    n_cols = min(3, n_admin)
    admin_rows = (n_admin + n_cols - 1) // n_cols
    # Each "admin row" expands into `n_variants` actual subplot rows (smoothed
    # then unsmoothed) so the same admin unit's variants sit directly stacked.
    n_rows_per_timepoint = admin_rows * n_variants
    total_rows = len(all_timepoints) * n_rows_per_timepoint
    fig, axes = plt.subplots(total_rows, n_cols, figsize=(6 * n_cols, 3.5 * total_rows))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_years)))
    year_color_map = dict(zip(all_years, colors))

    for timepoint_idx, timepoint in enumerate(all_timepoints):
        section_start_row = timepoint_idx * n_rows_per_timepoint
        if n_cols > 1:
            fig.text(
                0.5,
                1 - (section_start_row + 0.5) / total_rows,
                f"Timepoint: {timepoint}",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                transform=fig.transFigure,
            )

        for admin_idx, admin_unit in enumerate(all_admin_names):
            admin_row = admin_idx // n_cols
            col = admin_idx % n_cols
            for variant_idx, (variant_label, variant_col) in enumerate(lai_variants):
                global_row = section_start_row + admin_row * n_variants + variant_idx
                ax = get_axis(axes, global_row, col, total_rows, n_cols)

                plotted_any = False
                for year in all_years:
                    df = loaded.get((timepoint, year))
                    if df is None or variant_col not in df.columns:
                        continue
                    sub = df[df["region"] == admin_unit]
                    if sub.empty:
                        continue
                    dates = list(sub["Date_parsed"])
                    values = sub[variant_col].to_numpy()
                    n_regions = int(sub["n_regions"].iloc[0]) if "n_regions" in sub.columns else None
                    day_of_year = [d.timetuple().tm_yday for d in dates]
                    label = f"{year} (n={n_regions})" if n_regions is not None else str(year)
                    ax.plot(day_of_year, values, label=label, color=year_color_map[year], linewidth=2, alpha=0.8)
                    plotted_any = True

                ax.set_title(f"{admin_unit} — {variant_label}", fontsize=11, fontweight="bold")
                ax.set_xlabel("Day of Year")
                ax.set_ylabel(variant_col)
                ax.grid(True, alpha=0.3)

                if plotted_any:
                    ax.legend(fontsize=8)
                else:
                    ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                            transform=ax.transAxes, fontsize=10, alpha=0.7)
                ax.set_ylim(0, None)

        # Hide unused trailing cells in this section (last admin row may be partial).
        used_admin_cells = n_admin * n_variants
        total_cells = n_rows_per_timepoint * n_cols
        for spare_idx in range(n_admin, admin_rows * n_cols):
            admin_row = spare_idx // n_cols
            col = spare_idx % n_cols
            for variant_idx in range(n_variants):
                global_row = section_start_row + admin_row * n_variants + variant_idx
                if global_row < total_rows:
                    ax = get_axis(axes, global_row, col, total_rows, n_cols)
                    ax.set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.5)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {out_path}")


@click.command()
@click.option("--base-dir", type=click.Path(exists=True), required=True,
              help="Yield Study base directory containing year/timepoint subdirs.")
@click.option("--out-path", type=click.Path(), required=True, help="Output PDF path.")
@click.option("--level-name", type=str, required=True,
              help="Aggregation level name (matches the aggregate_lai_timeseries CSV filename).")
@click.option("--lai-agg-type", type=click.Choice(["Mean", "Median"]),
              help="Column of the LAI traces to use - either Mean or Median.")
@click.option("--adjusted", is_flag=True, default=False, help="Use the adjusted column in the LAI data.")
@click.option("--include-unsmoothed/--no-include-unsmoothed", default=True,
              help="Also draw the unsmoothed companion column directly below each smoothed plot. "
                   "Auto-skipped when the unsmoothed column is missing from every CSV.")
def main(base_dir, out_path, level_name, lai_agg_type, adjusted, include_unsmoothed):
    base = "LAI Mean" if lai_agg_type == "Mean" else "LAI Median"
    if adjusted:
        base = base + " Adjusted"

    variants = [("Smoothed", base)]
    if include_unsmoothed:
        variants.append(("Unsmoothed", base + " Unsmoothed"))

    create_agg_plots(base_dir, out_path, level_name, variants)


if __name__ == "__main__":
    main()
