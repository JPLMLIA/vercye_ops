"""CLI tools for fetching/formatting meteorological data"""

import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
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
    cols_to_check = ['PRECTOTCORR']
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

    # using simple linear interpolation to fill missing values
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

def get_consecutive_date_chunks(dates):
    """
    Split a DatetimeIndex of dates into consecutive chunks.
    """
    if dates.empty:
        return []
    
    # Ensure dates are sorted
    dates = dates.sort_values()

    chunks = []
    current_chunk_start = 0
    
    for i in range(1, len(dates)):
        # Check if there's a gap between consecutive dates (more than 1 day)
        if (dates[i] - dates[i-1]).days > 1:
            # End current chunk and start new one
            chunks.append(dates[current_chunk_start:i])
            current_chunk_start = i
    
    # Add the final chunk
    chunks.append(dates[current_chunk_start:])
    
    return chunks

def fetch_missing_nasa_power_data(output_fpath, start_date, end_date, variables, lon, lat):
    """
    Fetches missing weather data from the NASA POWER API for a given latitude and longitude between start_date and end_date,
    for the desired variables, based on an existing CSV file.
    """

    # Read existing data
    df_existing = pd.read_csv(output_fpath, index_col=0, parse_dates=True)

    # Determine the date range to fetch
    existing_dates = df_existing.index
    all_dates = get_dates_range(start_date, end_date)
    missing_dates = all_dates.difference(existing_dates)

    # Identify blocks of missing dates
    if missing_dates.empty:
        logger.info("No missing dates found. Using existing data.")
        return df_existing, None
    
    from_missing = missing_dates.min()
    to_missing = missing_dates.max()
    logger.info(f"Missing dates found: {from_missing.date()} to {to_missing.date()}")

    missing_data = []
    nodata_vals = []

    missing_date_chunks = get_consecutive_date_chunks(missing_dates)
    for i, chunk in enumerate(missing_date_chunks):
        logger.info("Fetching chunk %d with %d missing dates: %s to %s", 
                   i + 1, len(chunk), chunk[0].date(), chunk[-1].date())

        chunk_data, nodata_val = fetch_nasa_power_data(chunk[0], chunk[-1], variables, lon, lat)
        missing_data.append(chunk_data)
        nodata_vals.append(nodata_val)

    if len(set(nodata_vals)) > 1:
        raise ValueError("Nodata values are inconsistent across fetched data chunks - Not handling this case.")

    # Combine existing data with newly fetched data and bring in correct order
    df_combined = pd.concat([df_existing] + missing_data)
    df_combined.sort_index(inplace=True)

    return df_combined, nodata_vals[0]

def get_grid_aligned_coordinates(lat, lon):
    return lat, lon


def get_dates_range(start_date, end_date):
    """Generate a range of dates between the start and end dates."""
    return pd.date_range(start_date, end_date)


def write_met_data_to_csv(df, output_fpath):
    """
    Write the meteorological data to a CSV file.
    """
    df.to_csv(output_fpath)
    return output_fpath


def validate_aggregation_options(met_agg_method):
    """Helper to check that no unsupported combination is run"""

    if met_agg_method != 'centroid':
        raise Exception('NasaPower only supports centroid as the met_aggregation option.')


@click.command()
@click.option('--start_date', type=click.DateTime(formats=["%Y-%m-%d"]), required=True, help="Start date for data collection in YYYY-MM-DD format.")
@click.option('--end_date', type=click.DateTime(formats=["%Y-%m-%d"]), required=True, help="End date for data collection in YYYY-MM-DD format.")
@click.option('--variables', type=click.Choice(VALID_CLIMATE_VARIABLES, case_sensitive=False), multiple=True, default=DEFAULT_CLIMATE_VARIABLES, show_default=True, help="Meteorological variables to fetch for NasaPower. Unused if met source is ERA5")
@click.option('--lon', type=float, required=True, help="Longitude of the location.")
@click.option('--lat', type=float, required=True, help="Latitude of the location.")
@click.option('--met_agg_method', type=click.Choice(['mean', 'centroid'], case_sensitive=False), help="Method to aggregate meteorological data in a ROI.")
@click.option('--output_dir', type=click.Path(file_okay=False, dir_okay=True, writable=True), default=None, help="Directory where the .csv file will be saved.")
@click.option('--cache_dir', type=click.Path(file_okay=False, dir_okay=True, writable=True), default=None, help="Directory where the downloaded data will be cached to avoid rate limiting. If not provided, no caching will be done.")
@click.option('--overwrite-cache', is_flag=True, default=False, help="Enable file overwriting if weather data already exists in cache. Otherwise if a file exists, only missing dates will be appended to the existing file.")
@click.option('--verbose', is_flag=True, help="Enable verbose logging.")
def cli(start_date, end_date, variables, lon, lat, met_agg_method, output_dir, cache_dir, overwrite_cache, verbose):
    """Wrapper to fetch_met_data.
    This script is NOT intenden to be run in parallel for different dates for a specific location,
    as this would lead to race conditions."""

    if verbose:
        logger.setLevel('INFO')

    if output_dir is None and cache_dir is None:
        raise ValueError("At least one of output_dir or cache_dir must be specified.")
    
    if output_dir is None:
        logger.warning("No output_dir specified, data will only be saved to cache.")

    # !! Not implemented yet, currently just returns the same coordinates
    lat, lon = get_grid_aligned_coordinates(lat, lon)

    met_agg_method = met_agg_method.lower()
    validate_aggregation_options(met_agg_method)
    
    if cache_dir is not None:
        cache_region = f"{lon:.4f}_{lat:.4f}".replace('.', '_')
        cache_fpath = Path(cache_dir) / f'{cache_region}_{met_agg_method}_nasapower.csv'

    if cache_dir is not None and Path(cache_fpath).exists() and not overwrite_cache:
        logger.info("Weather data already exists locally. Will fetch and append only missing dates to if necessary: \n%s", cache_fpath)
        df, nodata_val = fetch_missing_nasa_power_data(cache_fpath, start_date, end_date, variables, lon, lat)
    else:
        logger.info("Fetching weather data from NASA POWER for %s to %s", start_date.date(), end_date.date())
        df, nodata_val = fetch_nasa_power_data(start_date, end_date, variables, lon, lat)
    
    df = clean_nasa_power_data(df, nodata_val)

    if cache_dir is not None:
        logger.info("Writing fetched data to cache file: %s", cache_fpath)
        os.makedirs(cache_dir, exist_ok=True)
        write_met_data_to_csv(df, cache_fpath)
    
    error_checking_function(df)
    if output_dir is not None:
        region = Path(output_dir).stem
        output_fpath = Path(output_dir) / f'{region}_met.csv'
        write_met_data_to_csv(df, output_fpath)
        logger.info("Data successfully written to %s", output_fpath)


if __name__ == '__main__':
    cli()
