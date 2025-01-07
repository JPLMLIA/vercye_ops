"""CLI tools for fetching/formatting meteorological data"""

import os
import os.path as op
from pathlib import Path

import click
import ftplib
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
    
    # Error checking function
    # TODO: Flesh out required checks
    df_cleaned = error_checking_function(df)

    return df_cleaned


def get_nasa_power_data(start_date, end_date, variables, lon, lat, output_fpath, overwrite):
    """
    Fetches weather data from the NASA POWER API for a given latitude and longitude between 
    start_date and end_date if not already present in the output_dir.
    """

    if Path(output_fpath).exists and not overwrite:
        logger.info("Weather data already exists locally. Skipping download for: \n%s", output_fpath)
        return pd.read_csv(output_fpath)
    
    return fetch_nasa_power_data(start_date, end_date, variables, lon, lat)


def download_file_ftp(file_name, output_fpath, ftp_connection):
    """
    Downloads a file form the cwd of a ftp connection
    """
    with open(output_fpath, 'wb') as local_file:
        ftp_connection.retrbinary(f"RETR {file_name}", local_file.write)


def fetch_chirps_daterange(start_date, end_date, output_dir):
    chirps_basedir = '/pub/org/chg/products/CHIRPS-2.0/global_daily/cogs/p05'
    ftp_connection = ftplib.FTP('ftp.chc.ucsb.edu')
    ftp_connection.login('anonymous', 'your_email_address')
    ftp_connection.cwd(chirps_basedir)

    cur_ftp_dir_year = None
    for date in pd.date_range(start_date, end_date):

        year = date.year
        if cur_ftp_dir_year != year:
            ftp_connection.cwd(op.join(chirps_basedir, str(year)))
            cur_ftp_dir_year = year

        chirps_file_name = f'chirps-v2.0.{date.strftime("%Y.%m.%d")}.cog'
        output_fpath = op.join(output_dir, chirps_file_name)
        if not op.exists(output_fpath):
            logger.info("Chirps precipitation data not existing locally for date {date}. Fetching and storing to: \n%s", output_fpath)
            download_file_ftp(chirps_file_name, output_fpath, ftp_connection)

    logger.info("Chirps data fetched successfully.")
    ftp_connection.quit()


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


def construct_chirps_data(start_date, end_date, aggregation_method, geometry_path=None, lon=None, lat=None, chirps_dir=None):
    """
    Constructs a DataFrame of CHIRPS precipitation data for the given date range and aggregation method.
    
    Parameters:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        aggregation_method (str): Aggregation method ('centroid' or 'mean').
        geometry_path (str, optional): Path to a shapefile for the 'mean' aggregation method.
        lon (float, optional): Longitude for the 'centroid' aggregation method.
        lat (float, optional): Latitude for the 'centroid' aggregation method.
        chirps_dir (str): Directory containing CHIRPS .cog files.

    Returns:
        pd.DataFrame: DataFrame with CHIRPS precipitation data.
    """
    validate_chirps_aggregation_inp(aggregation_method, lon, lat, geometry_path)

    dates = get_dates_range(start_date, end_date)

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


def get_chirps_precipitation(start_date, end_date, aggregation_method, geometry_path, lon, lat, chirps_dir):
    """
    Fetches precipitation data from the CHIRPS API between start_date and end_date if not already present in the output_dir.
    """

    # TODO Fallback to NASA POWER if CHIRPS data is not available fr certain regions (-50 to 50 lat)
    # TODO Parallelize this
    # TODO Discuss if we want to decouple the chirps data fetching from the pipeline, e.g by running cron-jobs.
    # TODO add better logging   
    # TODO validate mean and centroid values. Currently large discrepancies to nasa power. Check scale

    # Download CHIRPS data for the given data range
    fetch_chirps_daterange(start_date, end_date, chirps_dir)
    
    # Process the CHIRPS data by the given spatial aggregation method
    chirps_data = construct_chirps_data(start_date, end_date, aggregation_method, geometry_path, lon, lat, chirps_dir)

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

    df = get_nasa_power_data(start_date, end_date, variables, lon, lat, output_dir, overwrite)

    if precipitation_source == 'chirps':
        chirps_data = get_chirps_precipitation(start_date, end_date, precipitation_aggregation_method, geometry_path, lon, lat, chirps_dir)
        if len(chirps_data) != len(df):
            raise ValueError("NasaPower and Chirps data do not have the same length.")
        df['PRECTOTCORR'] = chirps_data

    write_met_data_to_csv(df, output_fpath)

if __name__ == '__main__':
    cli()