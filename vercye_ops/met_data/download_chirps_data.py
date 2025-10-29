import ftplib
import gzip
import os
import os.path as op
import queue
import random
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as Date
from enum import Enum, IntEnum
from threading import Lock

import click
import pandas as pd
from tqdm import tqdm

from vercye_ops.utils.init_logger import get_logger


class ProductType(str, Enum):
    PRELIM = "prelim"
    FINAL = "final"


class ProductVersion(IntEnum):
    V2 = 2
    V3 = 3


# Constants
CHIRPS_URL = "ftp.chc.ucsb.edu"
CHIRPS_USER = "anonymous"
CHIRPS_PASS = "your_email_address"
NUM_RETRIES = 5
RETRY_WAIT_TIME = 5  # Seconds to wait between retries
PROGRESS_UPDATE_INTERVAL = 10
MAX_FTP_CONNECTIONS = 5  # Maximum number of FTP connections to open at once. Max 5 to avoid blacklist.

FILE_TEMPLATES = {
    ProductVersion.V2: {
        ProductType.FINAL: "chirps-v2.0.{date}.cog",
        ProductType.PRELIM: "chirps-v2.0.{date}_prelim.tif",
    },
    ProductVersion.V3: {
        ProductType.FINAL: "chirp-v3.0.{date}.tif",
        # No Prelim available
    },
}

FTP_BASEDIRS = {
    ProductVersion.V2: {
        # Using COGs dir for final as they are better compressed
        ProductType.FINAL: "/pub/org/chc/products/CHIRPS-2.0/global_daily/cogs/p05",
        ProductType.PRELIM: "/pub/org/chc/products/CHIRPS-2.0/prelim/global_daily/tifs/p05",
    },
    ProductVersion.V3: {
        ProductType.FINAL: "/pub/org/chc/products/CHIRP-v3.0/daily/global/tifs",
        # No Prelim available
    },
}

# Initialize logger
logger = get_logger()


def get_chirps_file_name(date: Date, version: ProductVersion, product: ProductType) -> str:
    d = date.strftime("%Y.%m.%d")
    pmap = FILE_TEMPLATES.get(version)
    if not pmap:
        raise ValueError(f"Unsupported CHIRPS version: {version}")
    f_name = pmap.get(product)
    if not f_name:
        raise ValueError(f"{version.name} does not support product type '{product.value}'.")
    return f_name.format(date=d)


def get_chirps_file_path(dir: str, date: Date, version: ProductVersion, product: ProductType) -> str:
    return os.path.join(dir, get_chirps_file_name(date, version, product))


def get_chirps_ftp_basedir(version: ProductVersion, product: ProductType) -> str:
    pmap = FTP_BASEDIRS.get(version)
    if not pmap:
        raise ValueError(f"Invalid Product Version {version} specified.")
    basedir = pmap.get(product)
    if not basedir:
        raise ValueError(f"Product {product.value} is not available for CHIRPS version {version}.")
    return basedir


def version_supports_type(version: ProductVersion, product: ProductType) -> bool:
    return product in FTP_BASEDIRS[version]


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
        with tempfile.NamedTemporaryFile(suffix=".tif") as temp_file:
            temp_fpath = temp_file.name
            ftp_connection.retrbinary(f"RETR {file_name}", temp_file.write)
            temp_file.flush()  # Ensure buffer is flushed
            os.fsync(temp_file.fileno())  # Ensure data is written to disk

            # Now the file is closed, so can safely check the file size
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
        if str(e).startswith("550"):
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
        with gzip.open(file_path, "rb") as in_f, open(output_path, "wb") as out_f:
            shutil.copyfileobj(in_f, out_f)
    except Exception as e:
        # if anything went wrong, remove the partial output
        logger.warning("REMOVING")
        if op.exists(output_path):
            os.remove(output_path)

        raise e

    # success: now delete the source .gz
    os.remove(file_path)

    return output_path


def fetch_chirps_files(daterange, output_dir, connection_pool, version):
    """
    Fetch CHIRPS files for a specific daterange.

    Parameters
    ----------
    daterange : list of pandas.Timestamp
        List of dates to fetch files for.
    output_dir : str
        Directory to store the downloaded files.
    connection_pool: FTPConnectionPool
        Instance of ftp connection pool.
    version:
        Chirps data vesion to use (2 | 3)

    Returns
    -------
    list of pandas.Timestamp
        Dates for which downloads failed.
    """
    failed_downloads = []
    unavailable_dates = []
    cur_ftp_dir_year = None
    ftp_connection = connection_pool.get_connection()

    chirps_basedir = get_chirps_ftp_basedir(version, ProductType.FINAL)
    chirps_prelim_basedir = (
        get_chirps_ftp_basedir(version, ProductType.PRELIM)
        if version_supports_type(version, ProductType.PRELIM)
        else None
    )

    for date in daterange:
        year = date.year
        if cur_ftp_dir_year != year:
            try:
                ftp_connection.cwd(op.join(chirps_basedir, str(year)))
                cur_ftp_dir_year = year
            except ftplib.all_errors as e:
                logger.error(f"Error changing directory to year {year}: {e}")
                ftp_connection.quit()
                remaining_dates = list(daterange)[list(daterange).index(date) :]
                failed_downloads.extend(remaining_dates)
                return failed_downloads

        chirps_file_name = get_chirps_file_name(date, version, ProductType.FINAL)
        output_fpath = op.join(output_dir, chirps_file_name)

        if not op.exists(output_fpath):
            logger.info(
                f"Chirps precipitation data not existing locally for date {date}. Fetching and storing to: \n%s",
                output_fpath,
            )

            if not file_exists_ftp(chirps_file_name, ftp_connection):
                logger.warning("Final CHIRPS product not available for date: %s", date)
                unavailable_dates.append(date)
                continue

            try:
                download_file_ftp(chirps_file_name, output_fpath, ftp_connection)
                if output_fpath.endswith(".gz"):
                    output_fpath = unzip_file(output_fpath)
            except Exception as e:
                logger.error(f"Error downloading file {chirps_file_name}: {e}")
                failed_downloads.append(date)

            # Check if prelim file exists locally (only for V2 currtly as not avail in V3)
            if chirps_prelim_basedir:
                chirps_prelim_file_name = get_chirps_file_name(date, version, ProductType.PRELIM)
                chirps_prelim_file_name = chirps_prelim_file_name.replace(".tif.gz", "_prelim.tif")
                chirps_prelim_fpath = op.join(output_dir, chirps_prelim_file_name)

                # Remove the prelim file if it exists as it is replaced by the final file now
                if op.exists(chirps_prelim_fpath):
                    logger.warning("REMOVING Preliminary file and replaced")
                    os.remove(chirps_prelim_fpath)

    # Try to download the preliminary files for the unavailable dates
    if chirps_prelim_basedir:
        ftp_connection.cwd(chirps_prelim_basedir)
        cur_ftp_dir_year = None
        for date in unavailable_dates:
            chirps_prelim_file_name = get_chirps_file_name(date, version, ProductType.PRELIM)
            output_fpath = op.join(output_dir, chirps_prelim_file_name)
            year = date.year
            if cur_ftp_dir_year != year:
                try:
                    ftp_connection.cwd(op.join(chirps_prelim_basedir, str(year)))
                    cur_ftp_dir_year = year
                except ftplib.all_errors as e:
                    logger.error(f"Error changing directory to year {year}: {e}")
                    ftp_connection.quit()
                    remaining_dates = list(daterange)[list(daterange).index(date) :]
                    failed_downloads.extend(remaining_dates)
                    return failed_downloads

            unzipped_prelim_filepath = output_fpath.replace(".tif.gz", "_prelim.tif")
            if not op.exists(unzipped_prelim_filepath):
                logger.info(
                    "Chirps preliminary data not existing locally for date {date}. Fetching and storing to: \n%s",
                    output_fpath,
                )

                if not file_exists_ftp(chirps_prelim_file_name, ftp_connection):
                    logger.error("Neither Final nor Preliminary CHIRPS product available for date: %s", date)
                    continue

                try:
                    download_file_ftp(chirps_prelim_file_name, output_fpath, ftp_connection)
                    if output_fpath.endswith(".gz"):
                        output_fpath = unzip_file(output_fpath, unzipped_prelim_filepath)
                except Exception as e:
                    logger.error(f"Error downloading file {chirps_prelim_file_name}: {e}")
                    failed_downloads.append(date)

    connection_pool.release_connection(ftp_connection)
    return failed_downloads


def fetch_chirps_daterange_parallel(start_date, end_date, output_dir, num_workers, version):
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
    version: int
        Must be 2 or 3 for Chirps data version v2 or v3.
    """
    logger.info("Fetching CHIRPS data for range: %s to %s", start_date, end_date)

    num_workers = min(num_workers, MAX_FTP_CONNECTIONS)  # Cap due to server limitations

    all_dates = pd.date_range(start_date, end_date)
    total_files = len(all_dates)
    chunk_size = max(1, total_files // num_workers)
    chunk_size = min(chunk_size, PROGRESS_UPDATE_INTERVAL)
    daterange_chunks = [all_dates[i : i + chunk_size] for i in range(0, total_files, chunk_size)]

    retries = NUM_RETRIES

    logger.info("Initializing %d workers for parallel downloads.", num_workers)
    ftp_connection_pool = FTPConnectionPool(
        num_workers,
        CHIRPS_URL,
        CHIRPS_USER,
        CHIRPS_PASS,
    )

    with tqdm(total=total_files, desc="Total Progress") as progress_bar:
        for attempt in range(retries):
            logger.info("Starting download attempt %d", attempt + 1)
            failed_downloads = []

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(fetch_chirps_files, daterange, output_dir, ftp_connection_pool, version): daterange
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
                failed_downloads[i : i + chunk_size] for i in range(0, len(failed_downloads), chunk_size)
            ]

            # shuffle the chunks to avoid downloading the same files on the same workers
            random.shuffle(daterange_chunks)
            time.sleep(RETRY_WAIT_TIME)  # Wait before retrying

        logger.error("Maximum retries reached. Some files could not be downloaded.")


def validate_chirps_files(start_date: Date, end_date: Date, output_dir: str, version: ProductVersion):
    """
    Validate the downloaded CHIRPS files for existence

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
        if not os.path.exists(get_chirps_file_path(output_dir, date, version, ProductType.FINAL)):
            if version_supports_type(version, ProductType.PRELIM):
                if os.path.exists(get_chirps_file_path(output_dir, date, version, ProductType.PRELIM)):
                    logger.warning(
                        "Final CHIRPS product not available for date: %s. Using preliminary data instead.",
                        date,
                    )
                else:
                    logger.error("No CHIRPS product could be downloaded for date: %s", date)

    logger.info("Validation completed. Check for errors above.")


def is_daterange_complete(start_date, end_date, output_dir):
    required_dates = pd.date_range(start_date, end_date)
    required_dates = [d.strftime("%Y.%m.%d") for d in required_dates]
    available_files = os.listdir(output_dir)
    available_dates = [".".join(f.split(".")[2:5]) for f in available_files]
    missing_dates = set(required_dates) - set(available_dates)
    return len(missing_dates) == 0


def run_chirps_download(start_date: Date, end_date: Date, output_dir: str, num_workers: int, version: ProductVersion):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Shortcut to avoid setting up ftp if all data already present
    if is_daterange_complete(start_date, end_date, output_dir):
        return

    # Download CHIRPS data for the specified date range if not already present in outputdir
    fetch_chirps_daterange_parallel(start_date, end_date, output_dir, num_workers, version)

    # Validate the downloaded files for existence
    validate_chirps_files(start_date, end_date, output_dir, version)


@click.command()
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    required=True,
    help="Start date for CHIRPS data collection in YYYY-MM-DD format.",
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    required=True,
    help="End date for CHIRPS data collection in YYYY-MM-DD format.",
)
@click.option("--output-dir", required=True, help="Output directory to store the CHIRPS data.")
@click.option(
    "--num-workers",
    type=int,
    default=5,
    show_default=True,
    help="Number of parallel processes. Capped at 9 due to server limitations.",
)
@click.option(
    "--version",
    type=click.Choice([2, 3]),
    default=3,
    show_default=True,
    help="CHIRPS data version to use. Default is 3. Can be from v2 or from v3.",
)
@click.option("--verbose", is_flag=True, help="Enable verbose logging.", default=False)
def cli(start_date, end_date, output_dir, num_workers, version, verbose):
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
    num-workers : int
        Number of parallel processes. Capped at 9 due to server limitations.
    version: [2,3]
        Chiprs data version to use: v2 or v3. V3 is supposed to be the improved version with more coverage.
    verbose : bool
        Enable verbose logging.

    """
    if verbose:
        logger.setLevel("INFO")
    else:
        logger.setLevel("WARNING")

    run_chirps_download(
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        num_workers=num_workers,
        version=ProductVersion(version),
    )


if __name__ == "__main__":
    cli()
