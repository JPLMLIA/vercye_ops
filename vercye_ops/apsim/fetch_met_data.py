"""CLI tools for fetching/formatting meteorological data"""

import os.path as op
import logging
from pathlib import Path

import click
import requests
import pandas as pd


# Set up logging
logging.basicConfig(level=logging.INFO)

# Valid climate variables for the NASA POWER API
VALID_CLIMATE_VARIABLES = ["ALLSKY_SFC_SW_DWN", "T2M_MAX", "T2M_MIN", "T2M", "PRECTOTCORR", "WS2M"]
DEFAULT_CLIMATE_VARIABLES = ["ALLSKY_SFC_SW_DWN", "T2M_MAX", "T2M_MIN", "T2M", "PRECTOTCORR", "WS2M"]


def error_checking_function(df):
    """
    Stub function to perform error checking and cleaning on the dataframe.
    Replace or extend this function with actual error checking as needed.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing weather data.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame.
    """
    # Placeholder for actual error checking logic
    return df


def fetch_met_data(start_date, end_date, variables, lon, lat, output_dir, overwrite, verbose):
    """
    Fetches weather data from the NASA POWER API for a given latitude and longitude between start_date and end_date,
    for the desired variables, and writes it to a CSV after processing.
    """
    region = Path(output_dir).stem
    output_fpath = op.join(output_dir, f'{region}_nasapower.csv')
    if Path(output_fpath).exists and not overwrite:
        logging.info("Weather data already exists locally. Skipping: \n%s", output_fpath)
        return

    if verbose:
        logging.info("Fetching data from NASA POWER for %s to %s...", start_date.date(), end_date.date())

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

    if verbose:
        logging.info("Data fetched successfully. Processing data.")

    # Convert to DataFrame
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)
    
    # Error checking function
    # TODO: Flesh out required checks
    df_cleaned = error_checking_function(df)

    # Writing the cleaned DataFrame to the CSV file
    df_cleaned.to_csv(output_fpath)

    if verbose:
        logging.info("Data successfully written to %s", output_fpath)


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

    fetch_met_data(start_date, end_date, variables, lon, lat, output_dir, overwrite, verbose)


if __name__ == '__main__':
    cli()