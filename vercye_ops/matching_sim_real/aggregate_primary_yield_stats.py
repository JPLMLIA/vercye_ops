import logging
from pathlib import Path

import click
import pandas as pd
import pyarrow.parquet as pq

from vercye_ops.reporting.zonal_aggregation import compute_zonal_yield_stats
from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


def collect_apsim_metadata(yield_dir: str) -> pd.DataFrame:
    """
    Collect APSIM-specific metadata from per-region conversion_factor.csv files.

    These are per-region scalar values that cannot be derived from the raster mosaic.
    They include APSIM simulation-based yield estimates and matching diagnostics.

    Parameters
    ----------
    yield_dir : str
        Path to the yield directory containing region subdirectories.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'region' column and APSIM metadata columns.
    """
    apsim_cols = [
        "max_rs_lai",
        "max_rs_lai_date",
        "apsim_max_matched_lai",
        "apsim_max_matched_lai_date",
        "apsim_max_all_lai",
        "apsim_max_all_lai_date",
        "apsim_mean_yield_estimate_kg_ha",
        "apsim_matched_std_yield_estimate_kg_ha",
        "apsim_all_std_yield_estimate_kg_ha",
        "apsim_matched_maxlai_std",
        "apsim_all_maxlai_std",
    ]

    rows = []
    for region_dir in Path(yield_dir).iterdir():
        if not region_dir.is_dir():
            continue
        region_name = region_dir.name
        conv_factor_csv_path = region_dir / f"{region_name}_conversion_factor.csv"

        if not conv_factor_csv_path.exists():
            logger.warning(f"No conversion_factor.csv for region {region_name}, skipping APSIM metadata.")
            continue

        conv_df = pd.read_csv(conv_factor_csv_path)
        row = {"region": region_name}
        for col in apsim_cols:
            if col in conv_df.columns:
                val = conv_df[col].iloc[0]
                row[col] = val
            else:
                row[col] = None

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["region"] + apsim_cols)

    df = pd.DataFrame(rows)
    df = df.fillna(-1)

    # Format numeric columns
    for col in [
        "max_rs_lai",
        "apsim_max_matched_lai",
        "apsim_max_all_lai",
        "apsim_matched_maxlai_std",
        "apsim_all_maxlai_std",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    for col in [
        "apsim_mean_yield_estimate_kg_ha",
        "apsim_matched_std_yield_estimate_kg_ha",
        "apsim_all_std_yield_estimate_kg_ha",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)

    return df


def collect_lai_metadata(yield_dir: str, chirps_path: str = None) -> pd.DataFrame:
    """
    Collect LAI and weather metadata from per-region files.

    Parameters
    ----------
    yield_dir : str
        Path to the yield directory containing region subdirectories.
    chirps_path : str, optional
        Path to CHIRPS parquet file to determine precipitation source per region.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'region' and LAI/weather metadata columns.
    """
    regions_using_chirps = []
    if chirps_path is not None:
        parquet_file = pq.ParquetFile(chirps_path)
        regions_using_chirps = parquet_file.schema.names

    rows = []
    for region_dir in Path(yield_dir).iterdir():
        if not region_dir.is_dir():
            continue
        region_name = region_dir.name
        lai_stats_csv_path = region_dir / f"{region_name}_LAI_STATS.csv"

        precipitation_src = "CHIRPS" if region_name in regions_using_chirps else "Met Source"

        n_days_with_rs_data_valid = None
        mean_cloud_snow_percentage = None

        if lai_stats_csv_path.exists():
            rs_df = pd.read_csv(lai_stats_csv_path)
            valid_mask = (rs_df["interpolated"] == 0) & (rs_df["Cloud or Snow Percentage"] < 100)
            n_days_with_rs_data_valid = valid_mask.sum()
            mean_cloud_snow_percentage = rs_df.loc[valid_mask, "Cloud or Snow Percentage"].mean()

        rows.append(
            {
                "region": region_name,
                "precipitation_src": precipitation_src,
                "n_days_with_rs_data_valid": n_days_with_rs_data_valid,
                "mean_cloud_snow_percentage": mean_cloud_snow_percentage,
            }
        )

    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(
            columns=["region", "precipitation_src", "n_days_with_rs_data_valid", "mean_cloud_snow_percentage"]
        )
    )


def aggregate_primary_yield_stats(
    yield_mosaic_tif: str,
    coverage_mask_tif: str,
    primary_shapefile: str,
    name_column: str,
    yield_dir: str,
    chirps_path: str = None,
) -> pd.DataFrame:
    """
    Aggregate yield statistics at the primary (simulation region) level.

    Uses compute_zonal_yield_stats() from the reprojected equal-area mosaic for
    yield/production/area numbers (consistent with all other levels), then joins
    APSIM-specific metadata from per-region CSVs.

    Parameters
    ----------
    yield_mosaic_tif : str
        Path to the reprojected equal-area yield mosaic.
    coverage_mask_tif : str
        Path to the coverage mask (same grid as mosaic).
    primary_shapefile : str
        Path to the primary regions shapefile.
    name_column : str
        Column in the primary shapefile for region names.
    yield_dir : str
        Path to directory with per-region subdirectories (for APSIM metadata).
    chirps_path : str, optional
        Path to CHIRPS parquet for precipitation source identification.

    Returns
    -------
    pd.DataFrame
        Aggregated primary yield statistics with APSIM metadata.
    """
    # Yield/production stats from mosaic (consistent with all levels)
    zonal_df = compute_zonal_yield_stats(
        yield_mosaic_tif=yield_mosaic_tif,
        coverage_mask_tif=coverage_mask_tif,
        shapefile_path=primary_shapefile,
        name_column=name_column,
    )

    # APSIM metadata from per-region CSVs
    apsim_df = collect_apsim_metadata(yield_dir)

    # LAI/weather metadata
    lai_df = collect_lai_metadata(yield_dir, chirps_path)

    # Merge all together
    result = zonal_df
    if not apsim_df.empty:
        result = result.merge(apsim_df, on="region", how="left")
    if not lai_df.empty:
        result = result.merge(lai_df, on="region", how="left")

    # Reference yield data (if available) is already included in zonal_df
    # via the reference_yield_column parameter to compute_zonal_yield_stats

    # Sort by region
    result = result.sort_values("region").reset_index(drop=True)

    return result


@click.command()
@click.option(
    "--yield-mosaic-tif",
    required=True,
    type=click.Path(exists=True),
    help="Path to the reprojected equal-area yield mosaic.",
)
@click.option(
    "--coverage-mask-tif",
    required=True,
    type=click.Path(exists=True),
    help="Path to the coverage mask (same grid as mosaic).",
)
@click.option(
    "--primary-shapefile",
    required=True,
    type=click.Path(exists=True),
    help="Path to the primary regions shapefile.",
)
@click.option(
    "--name-column",
    required=True,
    type=str,
    help="Column in shapefile for region names.",
)
@click.option(
    "--yield-dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to directory with per-region subdirectories (for APSIM metadata).",
)
@click.option(
    "--output-csv",
    required=True,
    type=click.Path(),
    help="Output CSV path for aggregated primary yield stats.",
)
@click.option(
    "--chirps-file",
    required=False,
    type=click.Path(exists=True),
    default=None,
    help="Path to CHIRPS parquet file for precipitation source identification.",
)
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
def cli(
    yield_mosaic_tif,
    coverage_mask_tif,
    primary_shapefile,
    name_column,
    yield_dir,
    output_csv,
    chirps_file,
    verbose,
):
    """Aggregate primary-level yield statistics from the reprojected mosaic."""
    logging_level = logging.INFO if verbose else logging.WARNING
    logger.setLevel(logging_level)

    result = aggregate_primary_yield_stats(
        yield_mosaic_tif=yield_mosaic_tif,
        coverage_mask_tif=coverage_mask_tif,
        primary_shapefile=primary_shapefile,
        name_column=name_column,
        yield_dir=yield_dir,
        chirps_path=chirps_file,
    )

    if not result.empty:
        result.to_csv(output_csv, index=False)
        logger.info(f"Primary yield stats saved to: {output_csv}")
    else:
        logger.error("No data to aggregate.")


if __name__ == "__main__":
    cli()
