import gzip
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
import tempfile
import os
import shutil


# Initialize logger
logger = get_logger()

# Constant to adapt depending on the CHIRPS data version
CHIRPS_BASEDIR = '/pub/org/chc/products/CHIRPS-2.0/global_daily/cogs/p05'
CHIRPS_FILE_FORMAT = 'chirps-v2.0.{date}.cog'

CHIRPS_PRELIM_BASEDIR = '/pub/org/chc/products/CHIRPS-2.0/prelim/global_daily/tifs/p05'
CHIRPS_PRELIM_FILE_FORMAT = 'chirps-v2.0.{date}.tif.gz'

# Constants
CHIRPS_URL = 'ftp.chc.ucsb.edu'
CHIRPS_USER = 'anonymous'
CHIRPS_PASS = 'your_email_address'
NUM_RETRIES = 5
RETRY_WAIT_TIME = 5  # Seconds to wait between retries
PROGRESS_UPDATE_INTERVAL = 10
MAX_FTP_CONNECTIONS = 9  # Maximum number of FTP connections to open at once


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
    temp_fpath = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as temp_file:
            temp_fpath = temp_file.name
            ftp_connection.retrbinary(f"RETR {file_name}", temp_file.write)
            temp_file.flush()  # Ensure buffer is flushed
            os.fsync(temp_file.fileno())  # Ensure data is written to disk

        # Now the file is closed, so you can safely check the file size
        fsize = ftp_connection.size(file_name)
        if fsize != os.path.getsize(temp_fpath):
            os.remove(temp_fpath)
            raise IOError("Downloaded file is incomplete.")

        shutil.move(temp_fpath, output_fpath)
    except ftplib.all_errors as e:
        if temp_fpath and os.path.exists(temp_fpath):
            os.remove(temp_fpath)  # Ensure temp file is deleted on error
        raise IOError(f"Error downloading file {file_name}: {e}")
    
def file_exists_ftp(file_name, ftp_connection):
    """
    Check if a file exists on the FTP server.

    Parameters
    ----------
    file_name : str
        Name of the file to check.
    ftp_connection : ftplib.FTP
        Active FTP connection.

    Returns
    -------
    bool
        True if the file exists, False otherwise.
    """
    try:
        ftp_connection.size(file_name)
        return True
    except ftplib.error_perm as e:
        if str(e).startswith('550'):
            return False
        else:
            raise e


def unzip_file(file_path, output_path):
    """
    Unzips a gzipped file.

    Parameters
    ----------
    file_path : str
        Path to the gzipped file.

    Returns
    -------
    str
        Path to the unzipped file.
    """
    try:
        # open both in a single with-statement
        with gzip.open(file_path, 'rb') as in_f, open(output_path, 'wb') as out_f:
            shutil.copyfileobj(in_f, out_f)
    except Exception as e:
        # if anything went wrong, remove the partial output
        logger.warning('REMOVING')
        if op.exists(output_path):
            os.remove(output_path)
    
        raise e
    
    # success: now delete the source .gz
    os.remove(file_path)

    return output_path



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
    unavailable_dates = []
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
            logger.info(f"Chirps precipitation data not existing locally for date {date}. Fetching and storing to: \n%s", output_fpath)

            if not file_exists_ftp(chirps_file_name, ftp_connection):
                logger.warning("Final CHIRPS product not available for date: %s", date)
                unavailable_dates.append(date)
                continue

            try:
                download_file_ftp(chirps_file_name, output_fpath, ftp_connection)
                if output_fpath.endswith('.gz'):
                    output_fpath = unzip_file(output_fpath)
            except Exception as e:
                logger.error(f"Error downloading file {chirps_file_name}: {e}")
                failed_downloads.append(date)

            # Check if prelim file exists locally
            chirps_prelim_file_name = CHIRPS_PRELIM_FILE_FORMAT.format(date=date.strftime("%Y.%m.%d"))
            chirps_prelim_file_name = chirps_prelim_file_name.replace('.tif.gz', '_prelim.tif')
            chirps_prelim_fpath = op.join(output_dir, chirps_prelim_file_name)                                                                          

            # Remove the prelim file if it exists as it is replaced by the final file now
            if op.exists(chirps_prelim_fpath):
                logger.warning("REMOVING A")
                os.remove(chirps_prelim_fpath)
    
    # Try to download the preliminary files for the unavailable dates
    ftp_connection.cwd(CHIRPS_PRELIM_BASEDIR)
    cur_ftp_dir_year = None
    for date in unavailable_dates:
        chirps_prelim_file_name = CHIRPS_PRELIM_FILE_FORMAT.format(date=date.strftime("%Y.%m.%d"))
        output_fpath = op.join(output_dir, chirps_prelim_file_name)
        year = date.year
        if cur_ftp_dir_year != year:
            try:
                ftp_connection.cwd(op.join(CHIRPS_PRELIM_BASEDIR, str(year)))
                cur_ftp_dir_year = year
            except ftplib.all_errors as e:
                logger.error(f"Error changing directory to year {year}: {e}")
                ftp_connection.quit()
                remaining_dates = daterange[daterange.index(date):]
                failed_downloads.extend(remaining_dates)
                return failed_downloads
        
        unzipped_prelim_filepath = output_fpath.replace('.tif.gz', '_prelim.tif')
        if not op.exists(unzipped_prelim_filepath):
            logger.info(f"Chirps preliminary data not existing locally for date {date}. Fetching and storing to: \n%s", output_fpath)

            if not file_exists_ftp(chirps_prelim_file_name, ftp_connection):
                logger.error(f"Neither Final nor Preliminary CHIRPS product available for date: %s", date)
                continue

            try:
                download_file_ftp(chirps_prelim_file_name, output_fpath, ftp_connection)
                if output_fpath.endswith('.gz'):
                    output_fpath = unzip_file(output_fpath, unzipped_prelim_filepath)
            except Exception as e:
                logger.error(f"Error downloading file {chirps_prelim_file_name}: {e}")
                failed_downloads.append(date)

    connection_pool.release_connection(ftp_connection)
    return failed_downloads

def fetch_chirps_daterange_parallel(start_date, end_date, output_dir, num_workers):
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
    num_workers : int
        Number of parallel processes to use for downloading.
    """
    logger.info("Fetching CHIRPS data for range: %s to %s", start_date, end_date)

    num_workers = min(num_workers, MAX_FTP_CONNECTIONS)  # Cap due to server limitations

    all_dates = pd.date_range(start_date, end_date)
    total_files = len(all_dates)
    chunk_size = max(1, total_files // num_workers)
    chunk_size = min(chunk_size, PROGRESS_UPDATE_INTERVAL)
    daterange_chunks = [all_dates[i:i + chunk_size] for i in range(0, total_files, chunk_size)]

    retries = NUM_RETRIES

    logger.info("Initializing %d workers for parallel downloads.", num_workers)
    ftp_connection_pool = FTPConnectionPool(num_workers, CHIRPS_URL, CHIRPS_USER, CHIRPS_PASS)
    
    with tqdm(total=total_files, desc="Total Progress") as progress_bar:
        for attempt in range(retries):
            logger.info("Starting download attempt %d", attempt + 1)
            failed_downloads = []

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
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

def chirps_prelim_file_exists(date, output_dir):
    """
    Check if a CHIRPS preliminary file exists locally for a given date.

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
    chirps_prelim_file_name = CHIRPS_PRELIM_FILE_FORMAT.format(date=date.strftime("%Y.%m.%d"))
    chirps_prelim_file_name = chirps_prelim_file_name.replace('.tif.gz', '_prelim.tif')
    output_fpath = op.join(output_dir, chirps_prelim_file_name)
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
            if not chirps_prelim_file_exists(date, output_dir):
                logger.error(f"No CHIRPS product could be downloaded for date: %s", date)
            else:
                logger.warning(f"Final CHIRPS product not available for date: %s. Using preliminary data instead.", date)

    logger.info("Validation completed. Check for errors above.")


@click.command()
@click.option('--start-date', type=click.DateTime(formats=["%Y-%m-%d"]), required=True, help="Start date for CHIRPS data collection in YYYY-MM-DD format.")
@click.option('--end-date', type=click.DateTime(formats=["%Y-%m-%d"]), required=True, help="End date for CHIRPS data collection in YYYY-MM-DD format.")
@click.option('--output-dir', required=True, help="Output directory to store the CHIRPS data.")
@click.option('--num-workers', type=int, default=5, show_default=True, help="Number of parallel processes. Capped at 9 due to server limitations.")
@click.option('--verbose', is_flag=True, help="Enable verbose logging.", default=False)
def cli(start_date, end_date, output_dir, num_workers, verbose):
    """
    CLI wrapper to fetch CHIRPS precipitation data for a given date range.

    Parameters
    ----------
    start-date : datetime.datetime
        Start date for data collection.
    end-date : datetime.datetime
        End date for data collection.
    output-dir : str
        Directory to store the CHIRPS data.
    num-workersn : int
        Number of parallel processes. Capped at 9 due to server limitations.
    verbose : bool
        Enable verbose logging.

    """
    if verbose:
        logger.setLevel('INFO')
    else:
        logger.setLevel('WARNING')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Download CHIRPS data for the specified date range if not already present in outputdir
    fetch_chirps_daterange_parallel(start_date, end_date, output_dir, num_workers)

    # Validate the downloaded files for existence
    validate_chirps_files(start_date, end_date, output_dir)

if __name__ == '__main__':
    cli()
