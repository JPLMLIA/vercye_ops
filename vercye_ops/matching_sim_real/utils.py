import sqlite3

import pandas as pd
from pyproj import Transformer

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


# Function to load simulation data from the SQLite database
def load_simulation_data(db_path, crop_name):
    """Helper to load APSIM Databases"""

    crop_name = crop_name.lower()
    crop_name = crop_name.capitalize()

    # Set up a connection, and extract the SQLite DB to a pandas DF
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT SimulationID, `Clock.Today`, `Clock.Today.DayOfYear`, `{crop_name}.Leaf.LAI`, `Yield`
    FROM Report
    """
    df = pd.read_sql_query(query, conn, parse_dates="Clock.Today")
    conn.close()

    # check for duplicates and drop only those where all columns are the same
    # Note: In theory this should not happen, but in practice it does, need to investigate why!
    if df.duplicated(subset=["SimulationID", "Clock.Today"]).any():
        logger.warning(
            f"Duplicate entries found for {crop_name} in the simulation data. Dropping duplicates."
        )
        df = df.drop_duplicates(subset=df.columns, keep="first")

    if df.duplicated(subset=["SimulationID", "Clock.Today"]).any():
        logger.error(
            f"Duplicate entries for the same date in simulation still found after dropping duplicates."
        )
        raise ValueError("Duplicate entries still found in the simulation data.")

    # Cleanup date and set as index
    df["Date"] = df["Clock.Today"].dt.floor("D")
    df.drop(columns="Clock.Today", inplace=True)
    df.set_index("Date", inplace=True)

    return df


def load_simulation_units(db_path):
    """Helper to load APSIM Database Units"""

    # Set up a connection, and extract the SQLite DB to a pandas DF
    conn = sqlite3.connect(db_path)
    query = """
    SELECT * FROM _Units
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def compute_pixel_area(
    lon, lat, pixel_width_deg, pixel_height_deg, output_crs, input_crs="EPSG:4326"
):
    """Compute the area of a pixel in square meters given a latitude and pixel size in degrees"""

    # Create a transformer object
    transformer = Transformer.from_crs(input_crs, output_crs, always_xy=True)

    # Convert the corners of the pixel to Web Mercator
    x1, y1 = transformer.transform(lon, lat)
    x2, y2 = transformer.transform(lon + pixel_width_deg, lat + pixel_height_deg)

    # Calculate width and height in meters
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    # Calculate and return area in square meters
    return width * height
