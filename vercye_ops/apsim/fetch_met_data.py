"""CLI tools for fetching/formatting meteorological data"""

import os.path as op
from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.transform import rowcol
import requests


from vercye_ops.utils.init_logger import get_logger

logger = get_logger()

# Valid climate variables for the NASA POWER API
VALID_CLIMATE_VARIABLES = ["ALLSKY_SFC_SW_DWN", "T2M_MAX", "T2M_MIN", "T2M", "PRECTOTCORR", "WS2M"]
DEFAULT_CLIMATE_VARIABLES = ["ALLSKY_SFC_SW_DWN", "T2M_MAX", "T2M_MIN", "T2M", "PRECTOTCORR", "WS2M"]


def error_checking_function(df):
    """
    Perform error checking and logging on a dataframe containing NASA POWER data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing weather data.
    
    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame.
    """

    # Constant limits
    # TODO: verify these with Harvest team
    T2M_MAX_LIMIT = 50
    T2M_MIN_LIMIT = -40
    WS2M_MIN_LIMIT = 0
    WS2M_MAX_LIMIT = 20
    PRECTOTCORR_MIN_LIMIT = 0
    PRECTOTCORR_MAX_LIMIT = 300
    ALLSKY_SFC_SW_DWN_MIN_LIMIT = 0
    ALLSKY_SFC_SW_DWN_MAX_LIMIT = 100

    # Check T2M_MAX
    if (df['T2M_MAX'] > T2M_MAX_LIMIT).any():
        logger.error(f"T2M_MAX exceeds {T2M_MAX_LIMIT}")

    # Check T2M_MIN
    if (df['T2M_MIN'] < T2M_MIN_LIMIT).any():
        logger.error(f"T2M_MIN is below {T2M_MIN_LIMIT}")

    # Check WS2M
    if ((df['WS2M'] < WS2M_MIN_LIMIT) | (df['WS2M'] > WS2M_MAX_LIMIT)).any():
        logger.error(f"WS2M is not within the range {WS2M_MIN_LIMIT} to {WS2M_MAX_LIMIT}")

    # Check PRECTOTCORR
    if ((df['PRECTOTCORR'] < PRECTOTCORR_MIN_LIMIT) | (df['PRECTOTCORR'] > PRECTOTCORR_MAX_LIMIT)).any():
        logger.error(f"PRECTOTCORR is not within the range {PRECTOTCORR_MIN_LIMIT} to {PRECTOTCORR_MAX_LIMIT}")

    # Check ALLSKY_SFC_SW_DWN
    if ((df['ALLSKY_SFC_SW_DWN'] < ALLSKY_SFC_SW_DWN_MIN_LIMIT) | (df['ALLSKY_SFC_SW_DWN'] > ALLSKY_SFC_SW_DWN_MAX_LIMIT)).any():
        logger.error(f"ALLSKY_SFC_SW_DWN is not within the range {ALLSKY_SFC_SW_DWN_MIN_LIMIT} to {ALLSKY_SFC_SW_DWN_MAX_LIMIT}")

    return df


def fetch_nasa_power_data(start_date, end_date, variables, lon, lat):
    """
    Fetches weather data from the NASA POWER API for a given latitude and longitude between start_date and end_date,
    for the desired variables.
    """

    logger.info("Fetching data from NASA POWER for %s to %s...", start_date.date(), end_date.date())

    # Format the variables list into a comma-separated string for the API
    variables_str = ','.join(variables)

    # API endpoint and parameters
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {'parameters': variables_str,
              'community': 'AG',
              'longitude': lon,
              'latitude': lat,
              'start': start_date.strftime('%Y%m%d'),
              'end': end_date.strftime('%Y%m%d'),
              'format': 'JSON'}

    # Fetch data
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()['properties']['parameter']

    logger.info("Data fetched successfully. Processing data.")

    # Convert to DataFrame
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)
    
    return df


def get_nasa_power_data(start_date, end_date, variables, lon, lat, output_fpath, overwrite):
    """
    Fetches weather data from the NASA POWER API for a given latitude and longitude between 
    start_date and end_date if not already present in the output_dir.
    """

    if Path(output_fpath).exists and not overwrite:
        logger.info("Weather data already exists locally. Skipping download for: \n%s", output_fpath)
        return pd.read_csv(output_fpath)
    
    return fetch_nasa_power_data(start_date, end_date, variables, lon, lat)


def validate_chirps_aggregation_inp(aggregation_method, lon, lat, geometry_path):
    """Validate the inputs based on the chosen aggregation method."""
    if aggregation_method == 'centroid' and (lon is None or lat is None):
        raise ValueError("Longitude and latitude must be provided for 'centroid' aggregation method.")

    if aggregation_method == 'mean' and not geometry_path:
        raise ValueError("A valid shapefile path must be provided for 'mean' aggregation method.")


def get_dates_range(start_date, end_date):
    """Generate a range of dates between the start and end dates."""
    return pd.date_range(start_date, end_date)


def read_chirps_file(chirps_dir, date):
    """Generate the file path for a given date and read the CHIRPS data file."""
    chirps_file_path = op.join(chirps_dir, f'chirps-v2.0.{date.strftime("%Y.%m.%d")}.cog')
    return rasterio.open(chirps_file_path)


def process_centroid_data(chirps_dir, dates, lon, lat):
    """Process CHIRPS data using the centroid aggregation method."""
    results = np.full(len(dates), np.nan)
    for idx, date in enumerate(dates):
        with read_chirps_file(chirps_dir, date) as src:
            row, col = rowcol(src.transform, lon, lat)
            try:
                value = src.read(1)[row, col]
            except IndexError:
                raise Exception(f"Coordinates out of bounds for date {date}.")
                # Handle cases where the coordinates are out of bounds by possibly falling back on nasa power
            results[idx] = value

    return results


def process_mean_data(chirps_dir, dates, geometry):
    """Process CHIRPS data using the mean aggregation method."""
    results = np.full(len(dates), np.nan)
    for idx, date in enumerate(dates):
        with read_chirps_file(chirps_dir, date) as src:
            chirps_data, _ = mask(src, geometry.geometry, crop=True, nodata=src.nodata)
            chirps_mean = np.nanmean(chirps_data)
            results[idx] = chirps_mean

    return results


def construct_chirps_data(dates, aggregation_method, geometry_path=None, lon=None, lat=None, chirps_dir=None):
    """
       Constructs (aggregates) CHIRPS precipitation data based on the provided parameters.

        Parameters
        ----------
        dates : pd.DatetimeIndex
            Date range for the CHIRPS data.
        aggregation_method : str
            Aggregation method to use ('centroid' or 'mean').
        geometry_path : str, optional
            Path to a shapefile to use for the 'mean' aggregation method.
        lon : float, optional
            Longitude for the 'centroid' aggregation method.
        lat : float, optional
            Latitude for the 'centroid' aggregation method.
        chirps_dir : str
            Directory containing CHIRPS .cog files.

        Returns
        -------
        pd.DataFrame
            DataFrame containing CHIRPS precipitation data.
        """
    validate_chirps_aggregation_inp(aggregation_method, lon, lat, geometry_path)

    if aggregation_method == 'centroid':
        results = process_centroid_data(chirps_dir, dates, lon, lat)
    elif aggregation_method == 'mean':
        geometry = gpd.read_file(geometry_path)

        # Chirps CRS is EPSG:4326
        if geometry.crs != rasterio.crs.CRS.from_epsg(4326):
            geometry = geometry.to_crs(rasterio.crs.CRS.from_epsg(4326))

        results = process_mean_data(chirps_dir, dates, geometry)
    else:
        raise ValueError("Invalid aggregation method. Choose 'centroid' or 'mean'.")

    return results


def all_chirps_data_exists(dates, chirps_dir):
    """
    Validate that the CHIRPS data files exist for the given date range.
    """
    for date in dates:
        chirps_file_path = op.join(chirps_dir, f'chirps-v2.0.{date.strftime("%Y.%m.%d")}.cog')
        if not op.exists(chirps_file_path):
            logger.error("CHIRPS data not found for date %s. This data should be present under: %s", date, chirps_file_path)
            return False
    
    return True


def bbox_within_bounds(bbox, bounds):
    """
    Validate that a given bounding box is within the bounds.
    """
    return (bbox[0] >= bounds[0] and bbox[1] <= bounds[1] and bbox[2] >= bounds[2] and bbox[3] <= bounds[3])


def coord_within_bounds(lon, lat, bounds):
    """
    Validate that the given coordinates are within the bounds.
    """
    return (lon >= bounds[0] and lon <= bounds[1] and lat >= bounds[2] and lat <= bounds[3])


def within_chirps_bounds(geometry_path, lon, lat):
    """
    Validate that the given coordinates are within the bounds of the CHIRPS data.
    """

    chirps_bounds = (-180, 180, -50, 50)

    if geometry_path:
        geometry = gpd.read_file(geometry_path)
        geometry_bounds = geometry.total_bounds

        return bbox_within_bounds(geometry_bounds, chirps_bounds)

    if lon and lat:
       return coord_within_bounds(lon, lat, chirps_bounds)

    raise Exception("No valid geometry or coordinates provided.")


def get_chirps_precipitation(start_date, end_date, aggregation_method, geometry_path, lon, lat, chirps_dir):
    """
    Fetches precipitation data from the CHIRPS API between start_date and end_date if not already present in the output_dir.
    """

    logger.info("Using CHIRPS precipitation data for the given date range.")

    required_dates = get_dates_range(start_date, end_date)

    # Download CHIRPS data for the given data range
    if not all_chirps_data_exists(required_dates, chirps_dir):
        raise FileNotFoundError(f"CHIRPS data incomplete. Please download the data first. You may use the download_chirps_data.py script.")
    
    # Validate that the coordinates are within the bounds of the CHIRPS data
    if not within_chirps_bounds(geometry_path, lon, lat):
        raise ValueError("Coordinates of ROI out of bounds for CHIRPS data.")
    
    # Process the CHIRPS data by the given spatial aggregation method
    # TODO Parallelize this
    logger.info("Processing CHIRPS data using the %s aggregation method...", aggregation_method)
    chirps_data = construct_chirps_data(required_dates, aggregation_method, geometry_path, lon, lat, chirps_dir)

    return chirps_data


def write_met_data_to_csv(df, output_fpath):
    """
    Write the meteorological data to a CSV file.
    """
    df.to_csv(output_fpath)

    logger.info("Data successfully written to %s", output_fpath)
    return output_fpath


@click.command()
@click.option('--start_date', type=click.DateTime(formats=["%Y-%m-%d"]), required=True, help="Start date for data collection in YYYY-MM-DD format.")
@click.option('--end_date', type=click.DateTime(formats=["%Y-%m-%d"]), required=True, help="End date for data collection in YYYY-MM-DD format.")
@click.option('--variables', type=click.Choice(VALID_CLIMATE_VARIABLES, case_sensitive=False), multiple=True, default=DEFAULT_CLIMATE_VARIABLES, show_default=True, help="Meteorological variables to fetch.")
@click.option('--lon', type=float, required=True, help="Longitude of the location.")
@click.option('--lat', type=float, required=True, help="Latitude of the location.")
@click.option('--precipitation_source', type=click.Choice(['chirps', 'nasa_power'], case_sensitive=False), default='nasa_power', show_default=True, help="Source of precipitation data.")
@click.option('--precipitation_aggregation_method', type=click.Choice(['centroid', 'mean'], case_sensitive=False), default='centroid', show_default=True, help="Method to spatially aggregate precipitation data.")
@click.option('--geometry_path', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True), default=None, help="Path to the shapefile for the mean aggregation method.")
@click.option('--chirps_dir', type=click.Path(file_okay=False, dir_okay=True, writable=True), default=None, help="Directory where the CHIRPS data is saved / will be saved.")
@click.option('--output_dir', type=click.Path(file_okay=False, dir_okay=True, writable=True), required=True, help="Directory where the .csv file will be saved.")
@click.option('--overwrite', is_flag=True, help="Enable file overwriting if weather data already exists.")
@click.option('--verbose', is_flag=True, help="Enable verbose logging.")
def cli(start_date, end_date, variables, lon, lat, precipitation_source, precipitation_aggregation_method, geometry_path, chirps_dir, output_dir, overwrite, verbose):
    """Wrapper to fetch_met_data"""
    if verbose:
        logger.setLevel('INFO')

    region = Path(output_dir).stem
    output_fpath = op.join(output_dir, f'{region}_nasapower.csv')

    if precipitation_source.lower() == 'nasa_power' and precipitation_aggregation_method != 'centroid':
        raise ValueError("NASA POWER currenlty only supports centroid aggregation method for precipitation data. Please choose 'centroid' as the aggregation method.")

    df = get_nasa_power_data(start_date, end_date, variables, lon, lat, output_dir, overwrite)

    if precipitation_source.lower() == 'chirps':
        chirps_data = get_chirps_precipitation(start_date, end_date, precipitation_aggregation_method, geometry_path, lon, lat, chirps_dir)

        # Sanity check
        if len(chirps_data) != len(df):
            raise ValueError("NasaPower and Chirps data do not have the same length.")
        
        df['NASA_POWER_PRECTOTCORR_UNUSED'] = df['PRECTOTCORR']
        df['PRECTOTCORR'] = chirps_data
    
    # Error checking function
    # TODO: Flesh out required checks
    df_cleaned = error_checking_function(df)

    write_met_data_to_csv(df_cleaned, output_fpath)

if __name__ == '__main__':
    cli()