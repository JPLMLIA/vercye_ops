import logging

import click
import numpy as np
import pandas as pd
import rasterio

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


def process_geotiff(tif_path, csv_path, output_tif_fpath, use_adjusted):
    """
    Apply a conversion factor from a CSV file to the adjusted LAI image.

    Parameters
    ----------
    tif_path : str
        Filepath to the input LAI geotiff file.
    csv_path : str
        Filepath to the CSV file containing the conversion factor.
    output_tif_fpath : str
        Filepath where the output geotiff will be saved.
    use_adjusted : bool
        Whether to use the adjusted LAI values (second band).
    """

    # Read the conversion factor from CSV
    logger.info(f"Reading conversion factor from {csv_path}")
    df = pd.read_csv(csv_path)

    if "conversion_factor" not in df.columns:
        raise KeyError("CSV file must contain a 'conversion_factor' column.")
    conversion_factor = df["conversion_factor"].iloc[0]
    logger.info(f"Conversion factor: {conversion_factor}")

    # Open the input geotiff
    logger.info(f"Opening geotiff file {tif_path} and applying conversion factor")
    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()  # Make a copy of the profile
        data = src.read(
            2 if use_adjusted else 1
        )  # Read the adjusted (second) or unadjusted (first) band

        # Check for negative values. All LAI vals should be nonnegative
        if np.any(data < 0):
            data = np.clip(data, 0, None)  # Clip lower bound to 0

            logger.error(
                "Negative values found in the raw yield image data. Clipping lower bound to 0."
            )

    # Apply conversion factor
    data_converted = data * conversion_factor

    # Save the output geotiff
    logger.info(f"Saving the converted LAI map to {output_tif_fpath}")
    profile.update(
        {
            "count": 1,  # Set number of bands to 1 (there was one for unadjusted and one for adjusted)
            "compress": "lzw",
        }
    )  # Add compression to reduce file size

    with rasterio.open(output_tif_fpath, "w", **profile) as dst:
        dst.write(data_converted, 1)


@click.command()
@click.option(
    "--tif_fpath",
    required=True,
    type=click.Path(exists=True),
    help="Filepath to the input geotiff file.",
)
@click.option(
    "--csv_fpath",
    required=True,
    type=click.Path(exists=True),
    help="Filepath to the CSV file containing the conversion factor.",
)
@click.option(
    "--output_tif_fpath",
    required=True,
    type=click.Path(),
    help="Filepath where the output geotiff will be saved.",
)
@click.option("--use_adjusted", is_flag=True, help="Use the adjusted LAI values (second band).")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
def cli(tif_fpath, csv_fpath, output_tif_fpath, use_adjusted, verbose):
    """CLI for processing a geotiff image with a conversion factor."""

    # Configure logging
    logging_level = logging.INFO if verbose else logging.WARNING
    logger.setLevel(logging_level)

    process_geotiff(tif_fpath, csv_fpath, output_tif_fpath, use_adjusted)


if __name__ == "__main__":
    cli()
