"""Functions to generate a MET file from a CSV of meteorological data"""

import os.path as op
from pathlib import Path
from datetime import datetime

import numpy as np
import click
import pandas as pd
import pyarrow.parquet as pq

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


# Mapping constants for conversion of NASA POWER data to APSIM compatible names
POWER_TO_APSIM = {'YEAR': 'year',
                  'DOY': 'day',
                  'WS2M':'wind',
                  'T2M_MAX':'maxt',
                  'T2M_MIN':'mint',
                  'T2M':'meant',
                  'ALLSKY_SFC_SW_DWN':'radn',
                  'PRECTOTCORR':'rain',
                  'data_type': 'data_type'}

# Units for APSIM weather data
APSIM_UNITS = {'year':'()',
               'day': '()',
               'mint': '(oC)',
               'maxt': '(oC)',
               'meant': '(oC)',
               'rain':'(mm)',
               'wind':'(m/s)',
               'radn':'(MJ/m^2)',
               'code': '()',
               'data_type': '()'}

# Define how many decimal places to keep for projected data
APSIM_DECIMALS = {'WS2M': 2,
                  'T2M_MAX': 2,
                  'T2M_MIN': 2, 
                  'T2M': 2, 
                  'ALLSKY_SFC_SW_DWN': 2,
                  'PRECTOTCORR': 2}


def parse_weather_filename(filename):
    """
    Parse a filename with a specific format to extract latitude, longitude, and datetime range.
    
    Parameters
    ----------
    filename : str
        The filename to parse, expected to be in the format:
        "lat_{latitude}_lon_{longitude}_start_{start_date}_end_{end_date}.csv"
    
    Returns
    -------
    dict
        A dictionary containing the lat, lon, start and end dates:
    """
    # Remove the file extension
    basename = filename[:-4] if filename.endswith('.csv') else filename
    
    # Split the filename into parts based on underscores
    parts = basename.split('_')
    
    if parts[1] != 'lon' or parts[3] !='lat' or parts[5] != 'start' or parts[7] != 'end':
        raise RuntimeError('Error in the format of the input weather csv file')
    
    # Extract the values based on their positions
    lon = float(parts[2])  # Latitude is after 'lat'
    lat = float(parts[4])  # Longitude is after 'lon'
    start = datetime.strptime(parts[6], '%Y-%m-%d')  # Start date is after 'start'
    end = datetime.strptime(parts[8], '%Y-%m-%d')  # End date is after 'end'
    
    return lat, lon, start, end


def load_chirps_precipitation(start_date, end_date, chirps_file, column_name):
    """
    Loads the chirps precipitation data from the precomputed chirps file.
    """

    logger.info("Using CHIRPS precipitation data for the given date range.")
    required_dates = pd.date_range(start_date, end_date)

    parquet_file = pq.ParquetFile(chirps_file)
    column_names = parquet_file.schema.names
    if column_name not in column_names:
        raise KeyError(f"CHIRPS data incomplete. {column_name} not found in CHIRPS data.")

    chirps_data_unfiltered = pd.read_parquet(chirps_file, columns=[column_name])
    chirps_data = chirps_data_unfiltered.loc[required_dates.strftime('%Y-%m-%d')]
    chirps_data.index = pd.to_datetime(chirps_data.index)

    # Check for missing dates
    missing_dates = required_dates.difference(chirps_data.index)
    if not missing_dates.empty:
        raise ValueError(f"CHIRPS data is missing for the following dates: {missing_dates.tolist()}")

    return chirps_data[column_name]


def load_prep_project_data(weather_data_fpath, sim_end_date, precipitation_src, fallback_precipitation, region, chirps_file):
    """
    Load CSV data and prepare it by adding day of year and measurement type. Add
    weather projections if necessary

    Parameters
    ----------
    weather_data_fpath : str
        File path to the CSV file containing weather data.
    sim_end_date : datetime
        The end date for the simulation.

    Returns
    -------
    pandas.DataFrame
        The prepared DataFrame with additional columns for day of year and data type.
    """
    # Load CSV parsing dates in the first column and using that as the index
    df = pd.read_csv(weather_data_fpath, parse_dates=[0], index_col=0)

    # Using CHIRPS data if specified, by simply replacing the PRECTOTCORR column
    if precipitation_src.lower() == 'chirps':
        chirps_data = None
        try:
            start_date = df.index.min()
            end_date = df.index.max()

            chirps_data = load_chirps_precipitation(
                start_date=start_date,
                end_date=end_date,
                chirps_file=chirps_file,
                column_name=region
            )
        except KeyError as e:
            if not fallback_precipitation:
                raise KeyError('CHIRPS region data not found and fallback precipitation is not enabled.')
        except ValueError as e:
            if not fallback_precipitation:
                raise KeyError('CHIRPS region data incomplete.')

        if chirps_data is not None :
            logger.info('Using CHIRPS data for precipitation.')
            if len(chirps_data) != len(df):
                raise ValueError("Unexpected Chirps datalength.")
                
            df['PRECTOTCORR'] = chirps_data
        elif chirps_data is None and fallback_precipitation:
            logger.warning('CHIRPS data not found, falling back to original precipitation data.')
        else:
            raise KeyError('CHIRPS data not found and fallback precipitation is not enabled.')

    # Add year and day of year columns to prep for .met export
    df.insert(0, 'YEAR', df.index.year)
    df.insert(1, 'DOY', df.index.dayofyear)
    
    ###################################
    # Generate weather projections if needed
    last_date = df.index[-1]
    
    # We have measured data beyond the desired end of simulation
    if last_date >= sim_end_date:
        df['data_type'] = 'measured'
        return df
        
    if sim_end_date.year != last_date.year:
        logger.warning('Attempting to project weather data beyond the current year. You likely want data and simulation end date to be in the same year.')
    
    # Generate average weather data from past years, tack on year and day of year
    past_years_inds = df['YEAR'] < sim_end_date.year
    yearly_avg_data = df.loc[past_years_inds, :].groupby('DOY').mean().drop(columns='YEAR')
    
    # Handle leap year edge case, and detect too little data for projection
    if len(yearly_avg_data) == 365:
        yearly_avg_data.loc[366, :] = yearly_avg_data.loc[365, :]
    elif len(yearly_avg_data) < 365:
        logger.error('Projection is needed, but there\'s less than a full year of data.')
        raise RuntimeError()

    # Get projected days, years, and dt (for index)
    proj_days = [d.dayofyear for d in pd.date_range(last_date, sim_end_date, inclusive='right')]
    proj_years = [d.year for d in pd.date_range(last_date, sim_end_date, inclusive='right')]
    proj_dt = [datetime.strptime(f'{y}-{doy:03}', '%Y-%j') for y, doy in zip(proj_years, proj_days)]
    
    # Extract the weather day for all DOYs. Construct projected_df
    proj_dicts = [yearly_avg_data.loc[pd].to_dict() for pd in proj_days]  # index into projection data, now indexed by DOY
    projected_df = pd.DataFrame(proj_dicts, index=proj_dt)
    projected_df.insert(0, 'YEAR', proj_years)
    projected_df.insert(1, 'DOY', proj_days)
    
    # Round data to desired number of decimals
    round_dict = {key: val for key, val in APSIM_DECIMALS.items() 
                  if key in projected_df.columns}
    projected_df = projected_df.round(round_dict)
    
    # Add tag for type of data and return concatenated data
    df['data_type'] = 'measured'
    projected_df['data_type'] = 'projected'

    return pd.concat([df, projected_df])


def get_tav_amp(df):
    """Helper to compute average temp and temp amplitude

    Parameters
    ----------
    df : pandas.DataFrame
        Weather data containing the `data_type`, `T2M` columns

    Returns
    ----------
    tav : float
        Average temp calculated by averaging the 12 mean monthly temps
    amp : float
        Temp amplitude calculated as max - min of 12 mean monthly temps
    
    Notes
    -----
    Follows APSIM calculations described here: https://www.apsim.info/wp-content/uploads/2019/10/tav_amp-1.pdf
    
    >Amp is obtained by averaging the mean daily temperature of each month over 
    the entire data period resulting in twelve mean temperatures, and then 
    subtracting the minimum of these values from the maximum. Tav is obtained 
    by averaging the twelve mean monthly temperatures.
    """

    # Find inds to measured data
    relevant_data = df.loc[df['data_type'] == 'measured'].copy()
    
    # Add month of year column, then get average of T2M for each month
    relevant_data['month'] = relevant_data.index.month.to_list()
    monthly_means = relevant_data.loc[:, ('T2M', 'month')].groupby('month').mean().to_numpy() 

    # Compute average temp and temp amplitude according to APSIM docs:
    tav = np.mean(monthly_means)
    amp = np.max(monthly_means) - np.min(monthly_means)
    
    return tav, amp


def create_met_file(weather_data_fpath, lon, lat, sim_end_date, fallback_precipitation, output_dir, precipitation_src, chirps_file):
    """
    Processes weather data for APSIM simulations, integrating measured and forecasted data, and writing to a .met file.
    """

    ###################################
    # Extract basic info and run error checking

    region = Path(output_dir).name
    output_fpath = Path(op.join(output_dir, f'{region}_weather.met'))

    ###################################
    # Load, prep, project data and calc tav/amp

    df = load_prep_project_data(weather_data_fpath, sim_end_date, precipitation_src, fallback_precipitation, region, chirps_file)
    tav, amp = get_tav_amp(df)

    logger.info('Loaded data from %s', weather_data_fpath)

    ###################################
    # Write out to .met file

    with open(output_fpath, 'w') as file:
        file.write("[weather.met.weather]\n")
        file.write(f"latitude = {lat:0.2f}  (DECIMAL DEGREES)\n")
        file.write(f"longitude = {lon:0.2f}  (DECIMAL DEGREES)\n")
        file.write(f"tav = {tav:5.2f} (oC)    ! annual average ambient temperature\n")
        file.write(f"amp = {amp:5.2f} (oC)    ! annual amplitude (i.e., range) in mean monthly temperature\n")

        # Write header
        header = " ".join([f"{POWER_TO_APSIM[key]:<8}" for key in df.columns if key in POWER_TO_APSIM])
        header_units = " ".join([f"{APSIM_UNITS[POWER_TO_APSIM[key]]:<8}" for key in df.columns if key in POWER_TO_APSIM])
        file.write(f"{header}\n")
        file.write(f"{header_units}\n")

        # Add data rows to met file
        def format_row(row):
            """Helper to format each row into a string that matches the APSIM .met file format."""
            line = " ".join([f'{str(row[col]):<8}' if col in POWER_TO_APSIM else '' for col in df.columns])
            return f"{line}\n"

        formatted_data_lines = []
        for row in df.itertuples(index=False):  # Iter over tuples to preserve the integer datatypes
            formatted_data_lines.append(format_row(row._asdict()))

        # Write all lines at once
        file.writelines(formatted_data_lines)

    logger.info('Wrote .met file (containing %i data records) to\n%s', len(df), output_fpath)


@click.command()
@click.option('--weather_data_fpath', type=click.Path(exists=True), required=True, help="Path to the CSV file containing weather data.")
@click.option('--lon', type=float, required=True, help="Longitude of the location.")
@click.option('--lat', type=float, required=True, help="Latitude of the location.")
@click.option('--sim_end_date', type=click.DateTime(formats=["%Y-%m-%d"]), required=True, help="End date of the simulation period.")
@click.option('--chirps_file', type=click.Path(exists=True), required=False, help="Path to the precomputed CHIRPS data file.")
@click.option('--precipitation_source', type=click.Choice(['chirps', 'ERA5', 'nasa_power'], case_sensitive=False), default='nasa_power', show_default=True, help="Source of precipitation data.")
@click.option('--fallback_precipitation', help="Fallback to original precipitation data if CHIRPS data is not available.", default=False)
@click.option('--output_dir', type=click.Path(file_okay=False, dir_okay=True, writable=True), required=True, help="File path for the .met output file.")
@click.option('--verbose', is_flag=True, help="Enable verbose logging.")
def cli(weather_data_fpath, lon, lat, sim_end_date, chirps_file, precipitation_source, fallback_precipitation, output_dir, verbose):
    """Wrapper to processess weather data"""
    
    if verbose:
       logger.setLevel('INFO')

    create_met_file(weather_data_fpath, lon, lat, sim_end_date, fallback_precipitation, output_dir, precipitation_source, chirps_file)
    
    
if __name__ == '__main__':
    cli()