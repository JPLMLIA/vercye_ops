import logging
import click
import rasterio
import pandas as pd
import numpy as np

from vercye_ops.utils.init_logger import get_logger
from vercye_ops.matching_sim_real.utils import compute_pixel_area

logger = get_logger()


def estimate_yield(tif_path, output_yield_csv_fpath, target_epsg):
    """
    Estimate total yield from a converted LAI geotiff file.

    Parameters
    ----------
    tif_path : str
        Filepath to the input converted LAI geotiff file.
    output_yield_csv_fpath : str
        Filepath where the total yield will be saved as a CSV file.
    target_epsg : int
        EPSG code of the project target coordinate system.
    """
    logger.info(f"Opening geotiff file {tif_path}")
    with rasterio.open(tif_path) as src:
        data = src.read(1)  # Read the first (and only) band

        # Ensure the input CRS is in degrees (should be in EPSG:4326) before we do pixel area calculations
        if not src.crs.units_factor[0] == 'degree':
            raise ValueError("Input CRS is not in degrees. Cannot convert LAI to m2/pixel.")

        # Compute the pixel size using the center coordinates and projecting to the target CRS
        center_lon = (src.bounds.left + src.bounds.right) / 2
        center_lat = (src.bounds.bottom + src.bounds.top) / 2

        pixel_width_deg, pixel_height_deg = src.transform[0], np.abs(src.transform[4])

        pixel_area_m2 = compute_pixel_area(center_lon, center_lat, pixel_width_deg, pixel_height_deg, target_epsg)
        pixel_area_ha = pixel_area_m2 / 10000

        # For debugging purposes
        logger.info(f"Center coordinates: lon={center_lon:0.3f}, lat={center_lat:0.3f}")
        logger.info(f"Pixel area: {pixel_area_m2:0.4f} square meters, {pixel_area_ha:0.6f} hectares")

    # Estimate mean yield, total area, and total yield
    mean_yield = np.nanmean(data)  # Use nansum as there are likely nodata values in the input data
    total_area_ha = np.sum(~np.isnan(data)) * pixel_area_ha
    total_yield = np.nansum(data * pixel_area_ha)  # Use nansum as there are likely nodata values in the input data
    logger.info(f"Total yield: {total_yield} kg (average of {mean_yield:0.2f} kg/ha for {total_area_ha:0.2f} hectares)")

    # Save total yield to CSV
    logger.info(f"Saving total yield to {output_yield_csv_fpath}")
    with open(output_yield_csv_fpath, 'w') as f:
        pd.DataFrame({"mean_yield_kg_ha": [mean_yield],
                      "total_area_ha": [total_area_ha],
                      "total_yield_kg": [total_yield]}).to_csv(f, index=False)


@click.command()
@click.option('--converted_lai_tif_fpath', required=True, type=click.Path(exists=True), help='Filepath to the input converted LAI geotiff file.')
@click.option('--target_epsg', required=True, type=int, help='EPSG code of the project target coordinate system (used for pixel area calculation).')
@click.option('--output_yield_csv_fpath', required=True, type=click.Path(), help='Filepath where the total yield will be saved as a CSV file.')
@click.option('--verbose', is_flag=True, help='Enable verbose logging.')
def cli(converted_lai_tif_fpath, output_yield_csv_fpath, target_epsg, verbose):
    """CLI for estimating total yield from a converted LAI geotiff file."""

    # Configure logging
    logging_level = logging.INFO if verbose else logging.WARNING
    logger.setLevel(logging_level)

    estimate_yield(converted_lai_tif_fpath, output_yield_csv_fpath, target_epsg)

if __name__ == '__main__':
    cli()
