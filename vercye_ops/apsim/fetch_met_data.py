"""CLI tools for fetching/formatting meteorological data"""

import os.path as op
from pathlib import Path

import click
import numpy as np
import pandas as pd
import geopandas as gpd
import ee
import numpy as np
import requests
import pyarrow.parquet as pq
from dateutil.relativedelta import relativedelta

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

    if Path(output_fpath).exists() and not overwrite:
        logger.info("Weather data already exists locally. Skipping download for: \n%s", output_fpath)
        return pd.read_csv(output_fpath)
    
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


def get_era5_data(start_date, end_date, lon, lat, polygon_path, ee_project, output_fpath, overwrite):
    if Path(output_fpath).exists() and not overwrite:
        logger.info("Weather data already exists locally. Skipping download for: \n%s", output_fpath)
        return pd.read_csv(output_fpath)
    
    return fetch_era5_data(start_date, end_date, ee_project, lon, lat, polygon_path)


def fetch_era5_data(start_date, end_date, ee_project, lon=None, lat=None, polygon_path=None) :
    """
    Fetch meteorological data from ECMWF ERA5. Adjust outputs to align with NasaPower feature names.
    """
    logger.info('Fetching meteorological data from ERA5 trough google earth engine.')
    logger.info('Initializing google earth engine.')
    ee.Initialize(project=ee_project)

    logger.info('Querying data.')

    if polygon_path is None:
        if lat is None or lon is None:
            raise ValueError("Must provide either lat/lon or a polygon.")
        geometry = ee.Geometry.Point([lon, lat])
        geo_type = 'point'
    else:
        gdf = gpd.read_file(polygon_path)
        gdf = gdf.to_crs(epsg=4326)

        # Ensure only a single geometry is present in the file
        if len(gdf.geometry) != 1:
            raise Exception("Polygon File must contain a single geometry.")

        polygon_geom = gdf.geometry.iloc[0]
        geojson_dict = polygon_geom.__geo_interface__
        geometry = ee.Geometry(geojson_dict)
        geo_type = 'polygon'

    all_records = []
    
    def split_date_range(start_date, end_date, chunk_years=10):
        start = start_date
        end = end_date + relativedelta(days=1)
        while start < end:
            next_start = min(start + relativedelta(years=chunk_years), end)
            yield start.strftime('%Y-%m-%d'), next_start.strftime('%Y-%m-%d')
            start = next_start

    def extract(image):
        date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
        reducer = ee.Reducer.first() if geo_type == 'point' else ee.Reducer.mean()
        values = image.reduceRegion(reducer, geometry, 1000)
        return ee.Feature(None, values.set('date', date))
    
    for chunk_start, chunk_end in split_date_range(start_date, end_date, chunk_years=5):
        logger.info(f'Fetching data from {chunk_start} to {chunk_end}')
        try:
            era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
                .filterDate(chunk_start, chunk_end) \
                .filterBounds(geometry) \
                .select([
                    'total_precipitation_sum',
                    'temperature_2m_min',
                    'temperature_2m_max',
                    'surface_solar_radiation_downwards_sum',
                    'u_component_of_wind_10m',
                    'v_component_of_wind_10m'
                ])

            features = era5.map(extract)
            feature_collection = ee.FeatureCollection(features)
            result = feature_collection.getInfo()
            records = [f['properties'] for f in result['features']]
            all_records.extend(records)

        except Exception as e:
            logger.warning(f"Failed to fetch data from {chunk_start} to {chunk_end}: {e}")
            raise e

    df = pd.DataFrame(all_records)
    
    logger.info('Processing ERA5 data to required format.')

    df['date'] = pd.to_datetime(df['date'])

    df = df.rename(columns={
        'total_precipitation_sum': 'PRECTOTCORR',
        'temperature_2m_max': 'T2M_MAX',
        'temperature_2m_min': 'T2M_MIN',
        'surface_solar_radiation_downwards_sum': 'ALLSKY_SFC_SW_DWN',
        'u_component_of_wind_10m': 'u10',
        'v_component_of_wind_10m': 'v10'
    })

    # Convert temperatures from Kelvin to Celsius
    df['T2M_MAX'] = df['T2M_MAX'] - 273.15
    df['T2M_MIN'] = df['T2M_MIN'] - 273.15

    # Convert solar radiation from J/m² to MJ/m²
    df['ALLSKY_SFC_SW_DWN'] = df['ALLSKY_SFC_SW_DWN'] / 1_000_000

    # Convert rain from meters to millimeters
    df['PRECTOTCORR'] = df['PRECTOTCORR'] * 1000

    # Calculate wind speed from u and v components
    df['u10'] = df['u10'].astype(float)
    df['v10'] = df['v10'].astype(float)
    df['WS2M'] = np.sqrt(df['u10']**2 + df['v10']**2)

    # Compute mean temperate
    df['T2M'] = (df['T2M_MAX'] + df['T2M_MIN']) / 2

    # Keep only required columns for APSIM
    df = df[['date', 'ALLSKY_SFC_SW_DWN', 'T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'WS2M']]
    df.fillna({'ALLSKY_SFC_SW_DWN': 0, 'T2M': 0, 'T2M_MAX': 0, 'T2M_MIN': 0, 'PRECTOTCORR': 0, 'WS2M': 0}, inplace=True)

    # Ensure that we have continous data for every day from start_date to end_date
    expected_dates = pd.date_range(df['date'].min(), end_date, freq='D')
    missing_dates = expected_dates.difference(df['date'])

    if not missing_dates.empty:
        logger.warning(f"Missing dates in the data: {len(missing_dates)}")

        # sort missing dates ascending
        missing_dates = missing_dates.sort_values()
        # check if it is the newest dates that are missing
        if missing_dates[0] > df['date'].max():
            logger.warning(f"Missing dates are at the end of the data. Not filling them.")
        else:
            raise Exception('Missing dates - Not yet handled, this shouldnt occur.')

    # set date as index
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index).date
    df = df.sort_index()

    # check for duplicates
    if df.index.duplicated().any():
        raise ValueError("Duplicate dates found in the data.")

    return df


def validate_aggregation_options(precipitation_source, met_source, precipitation_agg_method, met_agg_method):
    """Helper to check that no unsupported combination is run"""

    if precipitation_source.lower() != met_source.lower()  and not precipitation_source.lower() == 'chirps':
        raise Exception('Currently precipitation_source and met_source must be the same if not using CHIPRS.')

    if precipitation_source.lower() == met_source.lower() and not precipitation_agg_method.lower() == met_agg_method.lower():
        raise Exception('Precipitation and Meteorological Aggregation Method must be the same if not using CHIRPS for precipitation data.')

    if met_source.lower() == 'nasa_power' and met_agg_method.lower() == 'mean':
        raise Exception('NasaPower only supports centroid as the met_aggregation option.')
    
    if  precipitation_source.lower() == 'nasa_power' and precipitation_agg_method.lower() != 'centroid':
        raise Exception('Currently only centroid aggregation method is supported for NasaPower.')

    
def clean_era5(df):
    # Clip negative precipitation values, if any
    neg_precip = df['PRECTOTCORR'] < 0
    if neg_precip.any():
        logger.warning(f'Clipping {neg_precip.sum()} negative precipitation values to 0.')
        df.loc[neg_precip, 'PRECTOTCORR'] = 0
    return df


@click.command()
@click.option('--start_date', type=click.DateTime(formats=["%Y-%m-%d"]), required=True, help="Start date for data collection in YYYY-MM-DD format.")
@click.option('--end_date', type=click.DateTime(formats=["%Y-%m-%d"]), required=True, help="End date for data collection in YYYY-MM-DD format.")
@click.option('--variables', type=click.Choice(VALID_CLIMATE_VARIABLES, case_sensitive=False), multiple=True, default=DEFAULT_CLIMATE_VARIABLES, show_default=True, help="Meteorological variables to fetch for NasaPower. Unused if met source is ERA5")
@click.option('--lon', type=float, required=True, help="Longitude of the location.")
@click.option('--lat', type=float, required=True, help="Latitude of the location.")
@click.option('--polygon_path', type=click.Path(file_okay=True, dir_okay=False), required=False, help="Path to the regions polygon if using mean aggregation method.", default=None)
@click.option('--met_source', type=click.Choice(['era5', 'nasa_power'], case_sensitive=False), default='nasa_power', show_default=True, help="Source of meteorological data.")
@click.option('--met_agg_method', type=click.Choice(['mean', 'centroid'], case_sensitive=False), help="Method to aggregate meteorological data in a ROI.")
@click.option('--precipitation_source', type=click.Choice(['chirps', 'nasa_power', 'era5'], case_sensitive=False), default='nasa_power', show_default=True, help="Source of precipitation data.")
@click.option('--chirps_column_name', default=None, help="Name of the region (ROI) must match a column in the CHIRPS file if used.")
@click.option('--fallback_precipitation', type=bool, help="Fallback to the original NasaPower or ERA5 precipitation data if CHIRPS data is not available.", default=False)
@click.option('--precipitation_agg_method', type=click.Choice(['mean', 'centroid'], case_sensitive=False), help="Method to aggregate precipitation data in a ROI.")
@click.option('--chirps_file', type=click.Path(file_okay=True, dir_okay=False), default=None, help="File where the CHIRPS extracted chirps-data is saved.")
@click.option('--ee_project', type=str, required=False, help='Name of the Earth Engine Project in which to run the ERA5 processing. Only required when using --met_source era5.')
@click.option('--output_dir', type=click.Path(file_okay=False, dir_okay=True, writable=True), required=True, help="Directory where the .csv file will be saved.")
@click.option('--overwrite', is_flag=True, help="Enable file overwriting if weather data already exists.")
@click.option('--verbose', is_flag=True, help="Enable verbose logging.")
def cli(start_date, end_date, variables, lon, lat, polygon_path, met_source, met_agg_method, precipitation_source, chirps_column_name, fallback_precipitation, precipitation_agg_method, chirps_file, ee_project, output_dir, overwrite, verbose):
    """Wrapper to fetch_met_data"""
    if verbose:
        logger.setLevel('INFO')
    region = Path(output_dir).stem
    output_fpath = Path(output_dir) / f'{region}_met.csv'

    validate_aggregation_options(precipitation_source, met_source, precipitation_agg_method, met_agg_method)

    if met_source.lower() == 'nasa_power':
        df, nodata_val = get_nasa_power_data(start_date, end_date, variables, lon, lat, output_fpath, overwrite)
        df = clean_nasa_power_data(df, nodata_val)
    elif met_source.lower() == 'era5':
        if ee_project is None:
            raise Exception('Setting --ee_project required when using ERA5 as the meteorological data source.')

        # Get mean or centroid data
        if met_agg_method.lower() == 'mean':
            df = get_era5_data(start_date, end_date, None, None, polygon_path, ee_project, output_fpath, overwrite)
        else:
            df = get_era5_data(start_date, end_date, lon, lat, None, ee_project, output_fpath, overwrite)

        
        error_checking_function(df)
        df = clean_era5(df)

    if precipitation_source.lower() == 'chirps':
        try:
            chirps_data = load_chirps_precipitation(start_date, end_date, chirps_file, chirps_column_name)

            # Sanity check
            if len(chirps_data) != len(df):
                raise ValueError("Unexpected Chirps datalength.")
            
            df['PRECTOTCORR_CHIRPS'] = chirps_data[chirps_column_name]
        except KeyError as e:
            logger.error(e)
            if fallback_precipitation:
                logger.warning(f"Falling back to {met_source} centroid data.")
                # No action needed as PRECTOTCORR is already initalized with NASAPOWER/ERA5 data.
            else:
                logger.error("You can set the --fallback_precipitation flag to use NASA POWER data as a fallback (Use with caution).")
                raise e
    
    error_checking_function(df)
    write_met_data_to_csv(df, output_fpath)

if __name__ == '__main__':
    cli()
