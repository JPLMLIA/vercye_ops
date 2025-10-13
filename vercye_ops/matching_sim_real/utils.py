import sqlite3

import pandas as pd
from pyproj import Transformer

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


def build_default_query(crop_name):
    query = f"""
        SELECT SimulationID, `Clock.Today`, `Clock.Today.DayOfYear`, `{crop_name}.Leaf.LAI`, `Yield`
        FROM Report
        """
    return query


# Function to load simulation data from the SQLite database
def load_simulation_data(db_path, crop_name=None, query=None):
    """Helper to load APSIM Databases
    Provide either crop_name to load default most important data
    or a custom query.
    """

    if not query and not crop_name:
        raise ValueError(("Either 'crop_name' or 'query' must be provided."))

    if query and crop_name:
        print("query and crop_name provided. Will only use query.")

    if crop_name:
        crop_name = crop_name.lower()
        crop_name = crop_name.capitalize()

    if not query:
        query = build_default_query(crop_name)

    # Set up a connection, and extract the SQLite DB to a pandas DF
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn, parse_dates="Clock.Today")
    conn.close()

    keys = ["SimulationID", "Clock.Today"]
    wt_col = "Wheat.Total.Wt"

    # check for duplicates and drop only those where all columns are the same
    # Note: In theory this should not happen, but in practice it does, need to investigate why!
    if df.duplicated(subset=keys).any():
        logger.warning(f"Duplicate entries found for {crop_name} in the simulation data. Dropping duplicates.")
        df = df.drop_duplicates(subset=df.columns, keep="first")

    # Work only on groups that still have duplicate keys
    dupe_mask = df.duplicated(subset=keys, keep=False)
    if dupe_mask.any():
        dups = df.loc[dupe_mask].copy()

        # Columns that must be identical across the group (everything except keys + wt_col)
        non_wt_cols = [c for c in df.columns if c not in set(keys + [wt_col])]

        # For each (SimulationID, Clock.Today) group, check if all non-WT columns are identical
        same_non_wt = dups.groupby(keys)[non_wt_cols].nunique(dropna=False).le(1).all(axis=1)
        rounding_only_groups = same_non_wt[same_non_wt].index

        # In rounding-only groups: keep the row with the largest Wheat.total.Wt, drop the others
        if len(rounding_only_groups) > 0:
            good_dups = dups.loc[dups.set_index(keys).index.isin(rounding_only_groups)].copy()
            # make sure it's numeric; NaNs will be sorted to the bottom
            good_dups[wt_col] = pd.to_numeric(good_dups[wt_col], errors="coerce")
            keep_idx = good_dups.groupby(keys)[wt_col].idxmax()
            drop_idx = good_dups.index.difference(keep_idx)
            if len(drop_idx) > 0:
                logger.info(f"Dropping {len(drop_idx)} rounding-only duplicate rows (kept the higher {wt_col}).")
                df = df.drop(index=drop_idx)

        # If any groups still have duplicates, they differ in other columns which we cant handle
        remaining = df[df.duplicated(subset=keys, keep=False)]
        if not remaining.empty:
            logger.error("Duplicate entries found for the same date in simulation (differences beyond Wheat.total.Wt):")
            logger.error(f"\n{remaining}")
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


def compute_pixel_area(lon, lat, pixel_width_deg, pixel_height_deg, output_crs, input_crs="EPSG:4326"):
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
