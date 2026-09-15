"""Aggregate per-region LAI timeseries to a single CSV per aggregation level.

Produces `agg_lai_timeseries_{level}_{study}_{year}_{timepoint}.csv` for one
(year, timepoint) under a study. The same data feeds the multiyear LAI report
PDF and the interactive map's per-level LAI charts, so the three artefacts
can never disagree.
"""
import logging
import os
from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import pandas as pd

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


LAI_STAT_BASES = ["Mean", "Median", "Stddev"]


def build_region_to_level_mapping(tp_path, level_shapefile, name_column):
    """Map each per-region directory under `tp_path` to a level polygon name.

    Forward pass: region centroid within level polygon (level coarser than region).
    Fallback: level-polygon centroid within region (level finer than region).
    """
    level_gdf = gpd.read_file(level_shapefile)

    region_rows = []
    crs = None
    for region in os.listdir(tp_path):
        geojson_file = os.path.join(tp_path, region, f"{region}.geojson")
        if not os.path.exists(geojson_file):
            continue
        gdf = gpd.read_file(geojson_file)
        if len(gdf) == 0:
            continue
        crs = gdf.crs
        region_rows.append({"_primary_region_": region, "geometry": gdf.geometry.iloc[0]})

    if not region_rows:
        return {}

    regions_gdf = gpd.GeoDataFrame(region_rows, crs=crs)
    if regions_gdf.crs != level_gdf.crs:
        regions_gdf = regions_gdf.to_crs(level_gdf.crs)

    level_sub = level_gdf[[name_column, "geometry"]].rename(columns={name_column: "_level_name_"})

    regions_centroid = regions_gdf.copy()
    regions_centroid["geometry"] = regions_centroid.geometry.centroid
    fwd = gpd.sjoin(regions_centroid, level_sub, how="left", predicate="within")

    mapping = {}
    for _, row in fwd.iterrows():
        if pd.notna(row.get("_level_name_")):
            mapping[row["_primary_region_"]] = str(row["_level_name_"])

    if mapping:
        return mapping

    level_centroid = level_sub.copy()
    level_centroid["geometry"] = level_centroid.geometry.centroid
    rev = gpd.sjoin(level_centroid, regions_gdf, how="left", predicate="within")
    for _, row in rev.iterrows():
        if pd.notna(row.get("_primary_region_")):
            mapping.setdefault(row["_primary_region_"], str(row["_level_name_"]))

    return mapping


def select_lai_columns(available_columns, adjusted, smoothed):
    """Return the LAI stat columns to aggregate, in stable order.

    `smoothed` controls whether the un-smoothed companion columns are also
    aggregated. The smoothed series itself is whatever the per-region pipeline
    wrote to `LAI Mean` / `LAI Median` / `LAI Stddev` (and Adjusted variants).
    """
    cols = []
    for base in LAI_STAT_BASES:
        cols.append(f"LAI {base}")
        if adjusted:
            cols.append(f"LAI {base} Adjusted")
    if smoothed:
        for base in LAI_STAT_BASES:
            cols.append(f"LAI {base} Unsmoothed")
            if adjusted:
                cols.append(f"LAI {base} Adjusted Unsmoothed")
    return [c for c in cols if c in available_columns]


def aggregate_for_level(tp_path, region_to_level, lai_columns):
    """For each level region name, read constituent LAI_STATS files and aggregate by Date."""
    rows_by_level = {}

    for region, level_name in region_to_level.items():
        stats_path = Path(tp_path) / region / f"{region}_LAI_STATS.csv"
        if not stats_path.exists():
            continue
        df = pd.read_csv(stats_path)
        if df.empty:
            continue
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
        df = df.dropna(subset=["Date"])
        rows_by_level.setdefault(level_name, []).append(df)

    out_frames = []
    for level_name, dfs in rows_by_level.items():
        combined = pd.concat(dfs, ignore_index=True)
        agg_dict = {col: "mean" for col in lai_columns if col in combined.columns}
        if "interpolated" in combined.columns:
            agg_dict["interpolated"] = "max"
        if "Cloud or Snow Percentage" in combined.columns:
            agg_dict["Cloud or Snow Percentage"] = "mean"

        grouped = combined.groupby("Date", as_index=False).agg(agg_dict)
        grouped["n_regions"] = combined.groupby("Date").size().reset_index(drop=True)
        grouped.insert(0, "region", level_name)

        if "Cloud or Snow Percentage" in grouped.columns:
            grouped = grouped.rename(columns={"Cloud or Snow Percentage": "mean_cloud_snow_pct"})

        out_frames.append(grouped)

    if not out_frames:
        cols = ["Date", "region"] + lai_columns + ["interpolated", "mean_cloud_snow_pct", "n_regions"]
        return pd.DataFrame(columns=cols)

    result = pd.concat(out_frames, ignore_index=True)
    result["Date"] = result["Date"].dt.strftime("%d/%m/%Y")

    ordered = ["Date", "region"] + [c for c in lai_columns if c in result.columns]
    for tail in ["interpolated", "mean_cloud_snow_pct", "n_regions"]:
        if tail in result.columns:
            ordered.append(tail)
    result = result[ordered].sort_values(["region", "Date"]).reset_index(drop=True)
    return result


@click.command()
@click.option("--basedir-path", required=True, type=click.Path(exists=True),
              help="Path to the timepoint directory (contains per-region subdirectories).")
@click.option("--level-shapefile", required=True, type=click.Path(exists=True),
              help="Path to the aggregation shapefile for this level.")
@click.option("--name-column", required=True, type=str,
              help="Column in the shapefile for region names.")
@click.option("--adjusted/--no-adjusted", default=True,
              help="Include crop-adjusted LAI columns if present in source files.")
@click.option("--smoothed/--no-smoothed", default=True,
              help="Include unsmoothed companion columns if present in source files.")
@click.option("--out-fpath", required=True, type=click.Path(),
              help="Output CSV path.")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
def cli(basedir_path, level_shapefile, name_column, adjusted, smoothed, out_fpath, verbose):
    """Aggregate per-region LAI timeseries to one row per (level region, date)."""
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    logger.info(f"Mapping regions to level polygons via {level_shapefile} (name_column={name_column})")
    region_to_level = build_region_to_level_mapping(basedir_path, level_shapefile, name_column)
    logger.info(f"Mapped {len(region_to_level)} regions")

    available_cols = set()
    for region in region_to_level:
        stats_path = Path(basedir_path) / region / f"{region}_LAI_STATS.csv"
        if stats_path.exists():
            available_cols.update(pd.read_csv(stats_path, nrows=0).columns)
            break
    lai_columns = select_lai_columns(available_cols, adjusted=adjusted, smoothed=smoothed)
    logger.info(f"Aggregating LAI columns: {lai_columns}")

    result = aggregate_for_level(basedir_path, region_to_level, lai_columns)

    Path(out_fpath).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_fpath, index=False)
    logger.info(f"Wrote {len(result)} rows ({result['region'].nunique() if not result.empty else 0} level regions) to {out_fpath}")


if __name__ == "__main__":
    cli()
