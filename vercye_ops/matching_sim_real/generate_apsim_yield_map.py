import logging

import click
import numpy as np
import pandas as pd
import rasterio

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


APSIM_YIELD_COLUMN = "apsim_mean_yield_estimate_kg_ha"


def process_geotiff(reference_tif_path, csv_path, output_tif_fpath):
    """
    Write a per-pixel raster with the APSIM-matched yield scalar at every cropland pixel.

    The reference TIF is the per-region converted-LAI yield map; its valid (non-nodata)
    pixels define the cropland mask for the region. We broadcast the scalar
    ``apsim_mean_yield_estimate_kg_ha`` from the conversion-factor CSV across that mask,
    preserving nodata pixels.

    Parameters
    ----------
    reference_tif_path : str
        Path to the per-region converted-LAI yield map TIF (cropland mask source).
    csv_path : str
        Path to the per-region conversion_factor CSV.
    output_tif_fpath : str
        Output path for the APSIM yield map TIF.
    """
    logger.info(f"Reading APSIM yield scalar from {csv_path}")
    df = pd.read_csv(csv_path)

    if APSIM_YIELD_COLUMN not in df.columns:
        raise KeyError(f"CSV file must contain a '{APSIM_YIELD_COLUMN}' column.")
    apsim_yield = float(df[APSIM_YIELD_COLUMN].iloc[0])
    logger.info(f"APSIM mean yield estimate: {apsim_yield} kg/ha")

    logger.info(f"Opening reference geotiff {reference_tif_path}")
    with rasterio.open(reference_tif_path) as src:
        profile = src.profile.copy()
        if src.count != 1:
            raise ValueError("Expecting a single band in the reference yield map.")
        data = src.read(1)
        nodata = src.nodata

    # A pixel is cropland iff the reference yield value is not nodata. The reference
    # raster typically uses NaN as nodata for float data; fall back to explicit nodata
    # comparison for integer rasters.
    if np.issubdtype(data.dtype, np.floating):
        valid_mask = np.isfinite(data)
    else:
        valid_mask = np.ones_like(data, dtype=bool)
        if nodata is not None:
            valid_mask &= data != nodata

    out_dtype = np.float32
    out = np.full(data.shape, np.nan, dtype=out_dtype)
    out[valid_mask] = np.float32(apsim_yield)

    profile.update(
        {
            "count": 1,
            "dtype": out_dtype,
            "nodata": np.nan,
            "compress": "lzw",
        }
    )

    logger.info(f"Saving APSIM yield map to {output_tif_fpath}")
    with rasterio.open(output_tif_fpath, "w", **profile) as dst:
        dst.write(out, 1)


@click.command()
@click.option(
    "--reference_tif_fpath",
    required=True,
    type=click.Path(exists=True),
    help="Path to the per-region converted-LAI yield map (defines cropland mask).",
)
@click.option(
    "--csv_fpath",
    required=True,
    type=click.Path(exists=True),
    help="Path to the per-region conversion_factor CSV (contains apsim_mean_yield_estimate_kg_ha).",
)
@click.option(
    "--output_tif_fpath",
    required=True,
    type=click.Path(),
    help="Output path for the APSIM yield map TIF.",
)
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
def cli(reference_tif_fpath, csv_fpath, output_tif_fpath, verbose):
    """Generate a per-pixel APSIM yield map by broadcasting the APSIM scalar over the cropland mask."""
    logging_level = logging.INFO if verbose else logging.WARNING
    logger.setLevel(logging_level)

    process_geotiff(reference_tif_fpath, csv_fpath, output_tif_fpath)


if __name__ == "__main__":
    cli()
