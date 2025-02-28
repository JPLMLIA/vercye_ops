import pandas as pd
import sqlite3
from pyproj import Transformer

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


# Function to load simulation data from the SQLite database
def load_simulation_data(db_path):
    """Helper to load APSIM Databases"""

    # Set up a connection, and extract the SQLite DB to a pandas DF
    conn = sqlite3.connect(db_path)
    query = """
    SELECT SimulationID, `Clock.Today`, `Clock.Today.DayOfYear`, `Maize.Leaf.LAI`, `Yield`
    FROM Report
    """
    df = pd.read_sql_query(query, conn, parse_dates='Clock.Today')
    conn.close()

    # Cleanup date and set as index
    df['Date'] = df['Clock.Today'].dt.floor('D')
    df.drop(columns='Clock.Today', inplace=True)
    df.set_index('Date', inplace=True)

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


def compute_pixel_area(lon, lat, pixel_width_deg, pixel_height_deg, output_epsg, input_epsg=4326):
    """Compute the area of a pixel in square meters given a latitude and pixel size in degrees"""

    # Create a transformer object
    transformer = Transformer.from_crs(input_epsg, output_epsg, always_xy=True)
    
    # Convert the corners of the pixel to Web Mercator
    x1, y1 = transformer.transform(lon, lat)
    x2, y2 = transformer.transform(lon + pixel_width_deg, lat + pixel_height_deg)
    
    # Calculate width and height in meters
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    
    # Calculate and return area in square meters
    return width * height