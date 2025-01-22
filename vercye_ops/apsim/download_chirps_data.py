import os.path as op
import ftplib
from concurrent.futures import ThreadPoolExecutor, as_completed
import click
import pandas as pd
from tqdm import tqdm
from vercye_ops.utils.init_logger import get_logger
import multiprocessing
import time
import random
import queue
from threading import Lock


# Initialize logger
logger = get_logger()

# Constant to adapt depending on the CHIRPS data version
CHIRPS_BASEDIR = '/pub/org/chg/products/CHIRPS-2.0/global_daily/cogs/p05'
CHIRPS_FILE_FORMAT = 'chirps-v2.0.{date}.cog'

# Constants
CHIRPS_URL = 'ftp.chc.ucsb.edu'
CHIRPS_USER = 'anonymous'
CHIRPS_PASS = 'your_email_address'
NUM_RETRIES = 5
RETRY_WAIT_TIME = 5  # Seconds to wait between retries
PROGRESS_UPDATE_INTERVAL = 10


class FTPConnectionPool:
    def __init__(self, max_connections, url, user, password):
        self.url = url
        self.user = user
        self.password = password
        self.pool = queue.Queue(max_connections)
        self.lock = Lock()
        for _ in range(max_connections):
            self.pool.put(self._create_connection())

    def _create_connection(self):
        try:
            ftp_connection = ftplib.FTP(self.url)
            ftp_connection.login(self.user, self.password)
            return ftp_connection
        except ftplib.all_errors as e:
            raise ConnectionError(f"Error connecting to FTP server: {e}")

    def get_connection(self):
        return self.pool.get()

    def release_connection(self, connection):
        self.pool.put(connection)

    def close_all_connections(self):
        while not self.pool.empty():
            connection = self.pool.get()
            connection.quit()


def download_file_ftp(file_name, output_fpath, ftp_connection):
    """
    Downloads a file from the current directory of an FTP connection.

    Parameters
    ----------
    file_name : str
        Name of the file to download.
    output_fpath : str
        Path to save the downloaded file.
    ftp_connection : ftplib.FTP
        Active FTP connection.
    """
    with open(output_fpath, 'wb') as local_file:
        try:
            ftp_connection.retrbinary(f"RETR {file_name}", local_file.write)
        except ftplib.all_errors as e:
            raise IOError(f"Error downloading file {file_name}: {e}")


def fetch_chirps_files(daterange, output_dir, connection_pool):
    """
    Fetch CHIRPS files for a specific daterange.

    Parameters
    ----------
    daterange : list of pandas.Timestamp
        List of dates to fetch files for.
    output_dir : str
        Directory to store the downloaded files.

    Returns
    -------
    list of pandas.Timestamp
        Dates for which downloads failed.
    """
    failed_downloads = []
    cur_ftp_dir_year = None
    ftp_connection = connection_pool.get_connection()

    for date in daterange:
        year = date.year
        if cur_ftp_dir_year != year:
            try:
                ftp_connection.cwd(op.join(CHIRPS_BASEDIR, str(year)))
                cur_ftp_dir_year = year
            except ftplib.all_errors as e:
                logger.error(f"Error changing directory to year {year}: {e}")
                ftp_connection.quit()
                remaining_dates = daterange[daterange.index(date):]
                failed_downloads.extend(remaining_dates)
                return failed_downloads

        chirps_file_name = CHIRPS_FILE_FORMAT.format(date=date.strftime("%Y.%m.%d"))
        output_fpath = op.join(output_dir, chirps_file_name)
        if not op.exists(output_fpath):
            logger.info("Chirps precipitation data not existing locally for date {date}. Fetching and storing to: \n%s", output_fpath)
            try:
                download_file_ftp(chirps_file_name, output_fpath, ftp_connection)
            except Exception as e:
                logger.error(f"Error downloading file {chirps_file_name}: {e}")
                failed_downloads.append(date)

    connection_pool.release_connection(ftp_connection)
    return failed_downloads




def fetch_chirps_daterange_parallel(start_date, end_date, output_dir, cpu_fraction):
    """
    Fetch CHIRPS data for a specified date range using parallel downloads.

    Parameters
    ----------
    start_date : datetime.datetime
        Start date for the data fetch.
    end_date : datetime.datetime
        End date for the data fetch.
    output_dir : str
        Directory to save downloaded files.
    cpu_fraction : float
        Fraction of available CPU cores to use for downloads.
    """
    logger.info("Fetching CHIRPS data for range: %s to %s", start_date, end_date)

    all_dates = pd.date_range(start_date, end_date)
    total_files = len(all_dates)
    chunk_size = max(1, total_files // int(multiprocessing.cpu_count() * cpu_fraction))
    chunk_size = min(chunk_size, PROGRESS_UPDATE_INTERVAL)
    daterange_chunks = [all_dates[i:i + chunk_size] for i in range(0, total_files, chunk_size)]

    max_workers = max(1, int(multiprocessing.cpu_count() * cpu_fraction))
    retries = NUM_RETRIES

    logger.info("Initializing %d workers for parallel downloads.", max_workers)
    ftp_connection_pool = FTPConnectionPool(max_workers, CHIRPS_URL, CHIRPS_USER, CHIRPS_PASS)
    
    with tqdm(total=total_files, desc="Total Progress") as progress_bar:
        for attempt in range(retries):
            logger.info("Starting download attempt %d", attempt + 1)
            failed_downloads = []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(fetch_chirps_files, daterange, output_dir, ftp_connection_pool): daterange
                    for daterange in daterange_chunks
                }

                for future in as_completed(futures):
                    try:
                        failed = future.result()

                        progress_bar.update(len(futures[future]) - len(failed))
                        if failed:
                            failed_downloads.extend(failed)
                    except Exception as e:
                        logger.error("Error while downloading daterange: %s", e)

            if not failed_downloads:
                logger.info("All downloads completed successfully.")
                return

            logger.warning("Retrying failed downloads (%d remaining)...", len(failed_downloads))

            # Reduces amount of directory switches
            failed_downloads.sort()

            daterange_chunks = [
                failed_downloads[i:i + chunk_size] 
                for i in range(0, len(failed_downloads), chunk_size)
            ]

            # shuffle the chunks to avoid downloading the same files on the same workers
            random.shuffle(daterange_chunks)
            time.sleep(RETRY_WAIT_TIME)  # Wait before retrying

        logger.error("Maximum retries reached. Some files could not be downloaded.")

def chirps_file_exists(date, output_dir):
    """
    Check if a CHIRPS file exists locally for a given date.

    Parameters
    ----------
    date : datetime.datetime
        Date to check for.
    output_dir : str
        Directory to check for the file.

    Returns
    -------
    bool
        True if the file exists, False otherwise.
    """
    chirps_file_name = CHIRPS_FILE_FORMAT.format(date=date.strftime("%Y.%m.%d"))
    output_fpath = op.join(output_dir, chirps_file_name)
    return op.exists(output_fpath)


def validate_chirps_files(start_date, end_date, output_dir):
    """
    Validate the downloaded CHIRPS files for existence and corruption.

    Parameters
    ----------
    start_date : datetime.datetime
        Start date for the data fetch.
    end_date : datetime.datetime
        End date for the data fetch.
    output_dir : str
        Directory to save downloaded files.
    """
    logger.info("Validating downloaded CHIRPS files for range: %s to %s", start_date, end_date)
    all_dates = pd.date_range(start_date, end_date)

    # Validate existence of files
    for date in all_dates:
        if not chirps_file_exists(date, output_dir):
            logger.error("CHIRPS file not found for date: %s", date)
            continue

    # Validate file integrity
    #TODO: Implement file integrity validation
    logger.info("Validation completed. Check for errors above.")


@click.command()
@click.option('--start_date', type=click.DateTime(formats=["%Y-%m-%d"]), required=True, help="Start date for CHIRPS data collection in YYYY-MM-DD format.")
@click.option('--end_date', type=click.DateTime(formats=["%Y-%m-%d"]), required=True, help="End date for CHIRPS data collection in YYYY-MM-DD format.")
@click.option('--output_dir', required=True, help="Output directory to store the CHIRPS data.")
@click.option('--cpu_fraction', type=float, default=0.5, show_default=True, help="Fraction of available CPU cores to use for downloads.")
def cli(start_date, end_date, output_dir, cpu_fraction):
    """
    CLI wrapper to fetch CHIRPS precipitation data for a given date range.

    Parameters
    ----------
    start_date : datetime.datetime
        Start date for data collection.
    end_date : datetime.datetime
        End date for data collection.
    output_dir : str
        Directory to store the CHIRPS data.
    cpu_fraction : float
        Fraction of available CPU cores to use.

    """
    logger.setLevel('INFO')

    if cpu_fraction <= 0 or cpu_fraction > 1:
        raise ValueError("CPU fraction must be between 0 and 1.")
    
    if not op.exists(output_dir):
        raise FileNotFoundError(f"Output directory {output_dir} does not exist.")

    # Download CHIRPS data for the specified date range if not already present in outputdir
    fetch_chirps_daterange_parallel(start_date, end_date, output_dir, cpu_fraction)

    # Validate the downloaded files for existence and integrity
    validate_chirps_files(start_date, end_date, output_dir)

if __name__ == '__main__':
    cli()
