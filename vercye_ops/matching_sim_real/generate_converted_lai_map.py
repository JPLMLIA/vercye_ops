import logging

import click
import numpy as np
import pandas as pd
import rasterio

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Mean-anchored LAI -> yield conversion.
#
# The APSIM matched regional yield is well calibrated and process-based; the
# remotely sensed peak-LAI raster is informative about WHERE the crop is doing
# better or worse, but its absolute level is not a reliable scale. So the APSIM
# yield sets the magnitude and LAI sets only the within-region pattern:
#
#     factor     = apsim_mean_yield / mean_cropland(LAI_MAX_pixel)
#     yield_px   = factor * LAI_MAX_pixel
#     => mean_cropland(yield_px) == apsim_mean_yield          (by construction)
#
# Why not scale by the regional LAI maximum: doing so divides by the max of the
# SMOOTHED regional-median LAI series while multiplying the per-pixel temporal max
# of the RAW LAI. Those are inconsistent statistics - the per-pixel raw max is
# biased upward by cloud/haze spikes (every pixel catches its own spurious peak)
# whereas the smoothed regional max suppresses them. The ratio
# mean(LAI_MAX_pixel)/max_rs_lai is then >> 1 and grows with LAI noise, producing
# year-varying over-prediction that is worst in cloudy seasons. Anchoring on the
# cropland mean removes that inconsistency; it is a mechanistic correction, not an
# empirical debias.
#
# Note: the conversion-factor CSV still carries a legacy `conversion_factor`
# column written by the matching script. It is deliberately not used here.
# ---------------------------------------------------------------------------

APSIM_YIELD_COLUMN = "apsim_mean_yield_estimate_kg_ha"


def process_geotiff(tif_path, csv_path, output_tif_fpath):
    """
    Convert a per-pixel max-LAI raster into a yield map by mean-anchoring it to the
    APSIM matched regional yield.

    Parameters
    ----------
    tif_path : str
        Filepath to the input per-pixel max LAI geotiff (single band, nodata=nan).
    csv_path : str
        Filepath to the per-region conversion-factor CSV; must contain
        ``apsim_mean_yield_estimate_kg_ha``.
    output_tif_fpath : str
        Filepath where the output yield map geotiff will be saved.
    """
    logger.info(f"Reading APSIM matched yield from {csv_path}")
    df = pd.read_csv(csv_path)

    if APSIM_YIELD_COLUMN not in df.columns:
        raise KeyError(f"CSV file must contain a '{APSIM_YIELD_COLUMN}' column for mean-anchored conversion.")
    apsim_mean_yield = float(df[APSIM_YIELD_COLUMN].iloc[0])
    logger.info(f"APSIM mean matched yield (kg/ha): {apsim_mean_yield}")

    logger.info(f"Opening geotiff file {tif_path}")
    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()
        if src.count != 1:
            raise ValueError("Expecting a single band in the max LAI file.")
        data = src.read(1).astype("float64")

    # All LAI vals should be nonnegative; clip spurious negatives before anchoring
    # so they cannot drag the denominator down.
    if np.any(data < 0):
        data = np.clip(data, 0, None)
        logger.error("Negative values found in the max LAI data. Clipping lower bound to 0.")

    # Cropland-mean of the per-pixel peak LAI - the self-consistent anchor denominator.
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
