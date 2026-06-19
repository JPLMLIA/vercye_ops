import logging

import click
import numpy as np
import pandas as pd
import rasterio

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Mean-anchored LAI->yield conversion (Ukraine yield-improvement branch).
#
# ROOT-CAUSE FIX. The original generate_converted_lai_map.py applies
#     yield_pixel = conversion_factor * LAI_MAX_pixel,
#     conversion_factor = apsim_mean_yield / max_rs_lai
# where the numerator raster LAI_MAX_pixel is the per-pixel temporal MAX of the
# RAW (unsmoothed) LAI, while the denominator max_rs_lai is the max of the
# SMOOTHED regional-median LAI timeseries. These are inconsistent statistics:
# the per-pixel raw max is upward-biased by cloud/haze noise (each pixel's max
# catches a spurious spike), whereas the smoothed regional max suppresses it.
# The resulting regional mean predicted yield is
#     mean_pred = apsim_mean_yield * mean(LAI_MAX_pixel) / max_rs_lai
# and the ratio mean(LAI_MAX_pixel)/max_rs_lai >> 1, growing with LAI noise ->
# year-varying over-prediction (worst in cloudy years, e.g. Ukraine 2020/2021).
#
# This fix makes the conversion self-consistent: distribute the (well-calibrated,
# process-based) APSIM matched regional yield across pixels in proportion to each
# pixel's peak LAI, so the cropland-mean predicted yield EQUALS the APSIM matched
# yield, and LAI only sets the within-region spatial pattern:
#     factor = apsim_mean_yield / mean_cropland(LAI_MAX_pixel)
#     yield_pixel = factor * LAI_MAX_pixel
# => mean_cropland(yield_pixel) == apsim_mean_yield  (by construction).
#
# This is a mechanistically-grounded calibration, not an empirical debias: it
# removes a statistical inconsistency in the anchor and ties the prediction's
# magnitude to the physically meaningful APSIM yield.
# ---------------------------------------------------------------------------


def process_geotiff(tif_path, csv_path, output_tif_fpath):
    """Apply a mean-anchored conversion factor to the per-pixel max LAI image."""

    df = pd.read_csv(csv_path)

    if "apsim_mean_yield_estimate_kg_ha" not in df.columns:
        raise KeyError(
            "CSV file must contain 'apsim_mean_yield_estimate_kg_ha' for mean-anchored conversion."
        )
    apsim_mean_yield = float(df["apsim_mean_yield_estimate_kg_ha"].iloc[0])
    logger.info(f"APSIM mean matched yield (kg/ha): {apsim_mean_yield}")

    # Open the input geotiff (single band: adjustedLAImax per cropland pixel; nodata=nan)
    logger.info(f"Opening geotiff file {tif_path}")
    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()
        if src.count != 1:
            raise ValueError("Expecting a single band in the max LAI file.")
        data = src.read(1).astype("float64")

    # All LAI vals should be nonnegative; clip spurious negatives to 0.
    if np.any(data < 0):
        data = np.clip(data, 0, None)
        logger.error("Negative values found in the max LAI data. Clipping lower bound to 0.")

    # Cropland-mean of the per-pixel peak LAI (the self-consistent anchor denominator)
    valid = np.isfinite(data) & (data > 0)
    n_valid = int(valid.sum())
    if n_valid == 0:
        logger.error("No valid (finite, >0) LAI pixels found; writing zeros.")
        mean_peak_lai = 0.0
    else:
        mean_peak_lai = float(np.nanmean(data[valid]))
    logger.info(f"Cropland-mean peak LAI: {mean_peak_lai} over {n_valid} pixels")

    conversion_factor = 0.0 if mean_peak_lai == 0 else apsim_mean_yield / mean_peak_lai
    logger.info(f"Mean-anchored conversion factor: {conversion_factor}")

    # Apply. nan pixels stay nan (nan * factor = nan), preserving nodata.
    data_converted = data * conversion_factor

    # Save the output geotiff
    logger.info(f"Saving the converted yield map to {output_tif_fpath}")
    profile.update({"count": 1, "compress": "lzw"})
    with rasterio.open(output_tif_fpath, "w", **profile) as dst:
        dst.write(data_converted, 1)


@click.command()
@click.option(
    "--tif_fpath",
    required=True,
    type=click.Path(exists=True),
    help="Filepath to the input per-pixel max LAI geotiff file.",
)
@click.option(
    "--csv_fpath",
    required=True,
    type=click.Path(exists=True),
    help="Filepath to the CSV file containing apsim_mean_yield_estimate_kg_ha.",
)
@click.option(
    "--output_tif_fpath",
    required=True,
    type=click.Path(),
    help="Filepath where the output yield map geotiff will be saved.",
)
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
def cli(tif_fpath, csv_fpath, output_tif_fpath, verbose):
    """CLI for converting a per-pixel max LAI image to a yield map via mean-anchoring."""
    logging_level = logging.INFO if verbose else logging.WARNING
    logger.setLevel(logging_level)
    process_geotiff(tif_fpath, csv_fpath, output_tif_fpath)


if __name__ == "__main__":
    cli()
