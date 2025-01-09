import os.path as op
import ftplib

import click
import pandas as pd

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()

CHIRPS_URL = 'ftp.chc.ucsb.edu'
CHIRPS_BASEDIR = '/pub/org/chg/products/CHIRPS-2.0/global_daily/cogs/p05'
CHIRPS_USER = 'anonymous'
CHIRPS_PASS = 'your_email_address'


def download_file_ftp(file_name, output_fpath, ftp_connection):
    """
    Downloads a file form the cwd of a ftp connection
    """
    with open(output_fpath, 'wb') as local_file:
        try:
            ftp_connection.retrbinary(f"RETR {file_name}", local_file.write)
        except ftplib.all_errors as e:
            raise Exception("Error downloading file %s: %s", file_name, e)


def fetch_chirps_daterange(start_date, end_date, output_dir):
    try:
        ftp_connection = ftplib.FTP(CHIRPS_URL)
        ftp_connection.login(CHIRPS_USER, CHIRPS_PASS)
        ftp_connection.cwd(CHIRPS_BASEDIR)
    except ftplib.all_errors as e:
        raise Exception("Error connecting to ftp server: %s", e)

    cur_ftp_dir_year = None
    for date in pd.date_range(start_date, end_date):

        year = date.year
        if cur_ftp_dir_year != year:
            ftp_connection.cwd(op.join(CHIRPS_BASEDIR, str(year)))
            cur_ftp_dir_year = year

        chirps_file_name = f'chirps-v2.0.{date.strftime("%Y.%m.%d")}.cog'
        output_fpath = op.join(output_dir, chirps_file_name)
        if not op.exists(output_fpath):
            logger.info("Chirps precipitation data not existing locally for date {date}. Fetching and storing to: \n%s", output_fpath)
            download_file_ftp(chirps_file_name, output_fpath, ftp_connection)

    logger.info("Chirps data fetched successfully.")
    ftp_connection.quit()

@click.command()
@click.option('--start_date', type=click.DateTime(formats=["%Y-%m-%d"]), required=True, help="Start date for chirps data collection in YYYY-MM-DD format.")
@click.option('--end_date', type=click.DateTime(formats=["%Y-%m-%d"]), required=True, help="End date for chirps data collection in YYYY-MM-DD format.")
@click.option('--output_dir', required=True, help="Output directory to store the chirps data.")
def cli(start_date, end_date, output_dir):
    """
    CLI wrapper to fetch CHIRPS precipitation data for a given date range.
    """

    # TODO Parallelize download

    fetch_chirps_daterange(start_date, end_date, output_dir)