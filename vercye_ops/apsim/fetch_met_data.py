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
import time
import pyarrow.parquet as pq
import time


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
    cols_to_check = ['PRECTOTCORR', 'PRECTOTCORR_CHIRPS'] if 'PRECTOTCORR_CHIRPS' in df.columns else ['PRECTOTCORR']
    for col in cols_to_check:
        if ((df[col] < PRECTOTCORR_MIN_LIMIT) | (df[col] > PRECTOTCORR_MAX_LIMIT)).any():
            logger.error(f"PRECTOTCORR is not within the range {PRECTOTCORR_MIN_LIMIT} to {PRECTOTCORR_MAX_LIMIT}")

    # Check ALLSKY_SFC_SW_DWN
    if ((df['ALLSKY_SFC_SW_DWN'] < ALLSKY_SFC_SW_DWN_MIN_LIMIT) | (df['ALLSKY_SFC_SW_DWN'] > ALLSKY_SFC_SW_DWN_MAX_LIMIT)).any():
        logger.error(f"ALLSKY_SFC_SW_DWN is not within the range {ALLSKY_SFC_SW_DWN_MIN_LIMIT} to {ALLSKY_SFC_SW_DWN_MAX_LIMIT}")

    return df

def clean_nasa_power_data(df, nodata_value):
    """
    Cleans the NASA POWER data by replacing nodata values.
    """

    if nodata_value is None:
        # Data is loaded from a previously downloaded local file and should already be clean
        return df
    
    # Replace nodata values with NaN
    df_cleaned = df.replace(nodata_value, np.nan)

    # useing simple linear interpolation to fill missing values
    df_cleaned = df_cleaned.interpolate(method='linear', limit_direction='both')

    return df_cleaned


def fetch_nasa_power_data(start_date, end_date, variables, lon, lat):
    """
    Fetches weather data from the NASA POWER API for a given latitude and longitude between start_date and end_date,
    for the desired variables.
    """

    logger.info("Fetching data from NASA POWER for %s to %s...", start_date.date(), end_date.date())

    # Format the variables list into a comma-separated string for the API
    variables_str = ','.join(variables)

    # API endpoint and parameters
    time.sleep(60)
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
    nodata_val = response.json()['header']['fill_value']

    logger.info("Data fetched successfully. Processing data.")

    # Convert to DataFrame
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)
    
    return df, nodata_val


def get_nasa_power_data(start_date, end_date, variables, lon, lat, output_fpath, overwrite):
    """
    Fetches weather data from the NASA POWER API for a given latitude and longitude between 
    start_date and end_date if not already present in the output_dir.
    """

    if Path(output_fpath).exists and not overwrite:
        logger.info("Weather data already exists locally. Skipping download for: \n%s", output_fpath)
        return pd.read_csv(output_fpath), None
    
    return fetch_nasa_power_data(start_date, end_date, variables, lon, lat)


def get_dates_range(start_date, end_date):
    """Generate a range of dates between the start and end dates."""
    return pd.date_range(start_date, end_date)


def load_chirps_precipitation(start_date, end_date, chirps_file, column_name):
    """
    Loads the chirps precipitation data from the precomputed chirps file.
    """

    logger.info("Using CHIRPS precipitation data for the given date range.")
    required_dates = get_dates_range(start_date, end_date)

    parquet_file = pq.ParquetFile(chirps_file)
    column_names = parquet_file.schema.names
    if column_name not in column_names:
        raise KeyError(f"CHIRPS data incomplete. {column_name} not found in CHIRPS data.")

    chirps_data_unfiltered = pd.read_parquet(chirps_file, columns=[column_name])
    chirps_data = chirps_data_unfiltered.loc[required_dates.strftime('%Y-%m-%d')]
    chirps_data.index = pd.to_datetime(chirps_data.index)

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
@click.option('--chirps_column_name', default=None, help="Name of the region (ROI) must match a column in the CHIRPS file if used.")
@click.option('--fallback_nasapower', help="Fallback to NASA POWER data if CHIRPS data is not available.", default=False)
@click.option('--chirps_file', type=click.Path(file_okay=True, dir_okay=False), default=None, help="File where the CHIRPS extracted chirps-data is saved.")
@click.option('--output_dir', type=click.Path(file_okay=False, dir_okay=True, writable=True), required=True, help="Directory where the .csv file will be saved.")
@click.option('--overwrite', is_flag=True, help="Enable file overwriting if weather data already exists.")
@click.option('--verbose', is_flag=True, help="Enable verbose logging.")
def cli(start_date, end_date, variables, lon, lat, precipitation_source, chirps_column_name, fallback_nasapower, chirps_file, output_dir, overwrite, verbose):
    """Wrapper to fetch_met_data"""
    if verbose:
        logger.setLevel('INFO')
    region = Path(output_dir).stem
    output_fpath = op.join(output_dir, f'{region}_nasapower.csv')

    df, nodata_value = get_nasa_power_data(start_date, end_date, variables, lon, lat, output_dir, overwrite)
    df = clean_nasa_power_data(df, nodata_value)

    if precipitation_source.lower() == 'chirps':
        try:
            chirps_data = load_chirps_precipitation(start_date, end_date, chirps_file, chirps_column_name)

            # Sanity check
            if len(chirps_data) != len(df):
                raise ValueError("NasaPower and Chirps data do not have the same length.")
            
            df['PRECTOTCORR_CHIRPS'] = chirps_data[chirps_column_name]
        except KeyError as e:
            logger.error(e)
            if fallback_nasapower:
                logger.error("Falling back to NASA POWER centroid data.")
                # No action needed as PRECTOTCORR is already initalized with nasapower data.
            else:
                logger.error("You can set the --fallback_nasapower flag to use NASA POWER data as a fallback (Use with caution).")
                raise e
    
    # Error checking function
    # TODO: Flesh out required checks
    df_cleaned = error_checking_function(df)

    write_met_data_to_csv(df_cleaned, output_fpath)

if __name__ == '__main__':
    cli()
