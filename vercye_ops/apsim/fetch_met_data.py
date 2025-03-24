"""CLI tools for fetching/formatting meteorological data"""

import os.path as op
from pathlib import Path

import click
import requests
import pandas as pd
import ee
import numpy as np
from datetime import datetime, timedelta

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


def fetch_nasa_power_data(start_date, end_date, variables, lon, lat, output_dir, overwrite):
    """
    Fetches weather data from the NASA POWER API for a given latitude and longitude between start_date and end_date,
    for the desired variables, and writes it to a CSV after processing.
    """
    region = Path(output_dir).stem
    output_fpath = op.join(output_dir, f'{region}_nasapower.csv')
    if Path(output_fpath).exists and not overwrite:
        logger.info("Weather data already exists locally. Skipping: \n%s", output_fpath)
        return

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

    # Writing the cleaned DataFrame to the CSV file
    df_cleaned.to_csv(output_fpath)

    logger.info("Data successfully written to %s", output_fpath)


def get_era5_weather(lat, lon, start_date, end_date):
    point = ee.Geometry.Point([lon, lat])
    start_dt = datetime.strptime(start_date, '%d-%m-%Y')
    end_dt = datetime.strptime(end_date, '%d-%m-%Y')

    era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
        .filterDate(start_dt, end_dt) \
        .filterBounds(point) \
        .select([
            'total_precipitation_sum',
            'temperature_2m_min',
            'temperature_2m_max',
            'surface_solar_radiation_downwards_sum',
            'u_component_of_wind_10m',
            'v_component_of_wind_10m'
        ])

    def extract(image):
        date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
        values = image.reduceRegion(ee.Reducer.first(), point, 1000)
        return ee.Feature(None, values.set('date', date))

    try:
        features = era5.map(extract).getInfo()['features']
    except Exception as e:
        print(f"Failed to extract ERA5 data for lat={lat}, lon={lon}: {e}")
        return pd.DataFrame()

    records = [f['properties'] for f in features]
    df = pd.DataFrame(records)

    df['date'] = pd.to_datetime(df['date'])

    df = df.rename(columns={
        'total_precipitation_sum': 'rain',
        'temperature_2m_max': 'maxt',
        'temperature_2m_min': 'mint',
        'surface_solar_radiation_downwards_sum': 'radn',
        'u_component_of_wind_10m': 'u10',
        'v_component_of_wind_10m': 'v10'
    })

    # Convert temperatures from Kelvin to Celsius
    df['maxt'] = df['maxt'] - 273.15
    df['mint'] = df['mint'] - 273.15

    # Convert solar radiation from J/m² to MJ/m²
    df['radn'] = df['radn'] / 1_000_000

     # Convert rain from meters to millimeters
    df['rain'] = df['rain'] * 1000

    # Calculate wind speed from u and v components
    df['wind'] = np.sqrt(df['u10']**2 + df['v10']**2)

    # Add year and day-of-year columns
    df['year'] = df['date'].dt.year
    df['day'] = df['date'].dt.dayofyear

    # Keep only required columns for APSIM
    df = df[['date', 'year', 'day', 'radn', 'maxt', 'mint', 'rain', 'wind']]
    return df


@click.command()
@click.option('--start_date', type=click.DateTime(formats=["%Y-%m-%d"]), required=True, help="Start date for data collection in YYYY-MM-DD format.")
@click.option('--end_date', type=click.DateTime(formats=["%Y-%m-%d"]), required=True, help="End date for data collection in YYYY-MM-DD format.")
@click.option('--variables', type=click.Choice(VALID_CLIMATE_VARIABLES, case_sensitive=False), multiple=True, default=DEFAULT_CLIMATE_VARIABLES, show_default=True, help="Meteorological variables to fetch.")
@click.option('--lon', type=float, required=True, help="Longitude of the location.")
@click.option('--lat', type=float, required=True, help="Latitude of the location.")
@click.option('--output_dir', type=click.Path(file_okay=False, dir_okay=True, writable=True), required=True, help="Directory where the .csv file will be saved.")
@click.option('--overwrite', is_flag=True, help="Enable file overwriting if weather data already exists.")
@click.option('--verbose', is_flag=True, help="Enable verbose logging.")
def cli(start_date, end_date, variables, lon, lat, output_dir, overwrite, verbose):
    """Wrapper to fetch_met_data"""
    if verbose:
        logger.setLevel('INFO')

    fetch_nasa_power_data(start_date, end_date, variables, lon, lat, output_dir, overwrite)


if __name__ == '__main__':
    cli()