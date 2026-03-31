import logging

import click
import pandas as pd

from vercye_ops.reporting.zonal_aggregation import compute_zonal_yield_stats, extract_reference_from_shapefile
from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


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
    "--level-shapefile",
    required=True,
    type=click.Path(exists=True),
    help="Path to the aggregation shapefile for this level.",
)
@click.option(
    "--name-column",
    required=True,
    type=str,
    help="Column in the shapefile for region names.",
)
@click.option(
    "--reference-yield-column",
    required=False,
    type=str,
    default=None,
    help="Optional column in shapefile with reference yield (kg/ha) for evaluation.",
)
@click.option(
    "--year-column",
    required=False,
    type=str,
    default=None,
    help="Column containing the year for each row. Required if reference_yield_column is set.",
)
@click.option(
    "--year",
    required=False,
    type=str,
    default=None,
    help="The year to filter reference data by (matched against year_column values).",
)
@click.option(
    "--out-fpath",
    required=True,
    type=click.Path(),
    help="Output CSV path for aggregated yield statistics.",
)
@click.option(
    "--out-refdata-fpath",
    required=False,
    type=click.Path(),
    default=None,
    help="Optional output path for extracted reference data CSV (for evaluate rule).",
)
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
def cli(
    yield_mosaic_tif,
    coverage_mask_tif,
    level_shapefile,
    name_column,
    reference_yield_column,
    year_column,
    year,
    out_fpath,
    out_refdata_fpath,
    verbose,
):
    """Compute zonal yield statistics at a given aggregation level."""
    logging_level = logging.INFO if verbose else logging.WARNING
    logger.setLevel(logging_level)

    logger.info(f"Computing yield stats for level shapefile: {level_shapefile}")

    # Zonal stats use unique geometries (deduplicated by name_column)
    # — the shapefile may have duplicate geometries if it has year-specific reference data
    result = compute_zonal_yield_stats(
        yield_mosaic_tif=yield_mosaic_tif,
        coverage_mask_tif=coverage_mask_tif,
        shapefile_path=level_shapefile,
        name_column=name_column,
    )

    # If reference yield column is set, extract year-filtered reference data
    # and merge with zonal stats
    ref_df = pd.DataFrame(columns=["region", "reported_mean_yield_kg_ha"])

    if reference_yield_column and year_column and year:
        logger.info(
            f"Extracting reference data for year {year} from column '{reference_yield_column}' (year column: '{year_column}')"
        )
        ref_df = extract_reference_from_shapefile(
            shapefile_path=level_shapefile,
            name_column=name_column,
            reference_yield_column=reference_yield_column,
            year_column=year_column,
            year=year,
        )

        if not ref_df.empty:
            result = result.merge(ref_df, on="region", how="left")
            logger.info(f"Merged reference data: {len(ref_df)} entries for year {year}")
        else:
            logger.warning(f"No reference data found for year {year}")

    elif reference_yield_column and not year_column:
        logger.warning(
            f"reference_yield_column '{reference_yield_column}' is set but year_column is not. "
            "Cannot filter reference data by year — skipping evaluation data."
        )

    result.to_csv(out_fpath, index=False)
    logger.info(f"Aggregated yield stats saved to: {out_fpath}")

    # Always write the reference CSV (even empty) so Snakemake output is satisfied
    if out_refdata_fpath:
        ref_df.to_csv(out_refdata_fpath, index=False)
        logger.info(f"Reference data written to: {out_refdata_fpath} ({len(ref_df)} rows)")


if __name__ == "__main__":
    cli()
