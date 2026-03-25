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
    ndvi_col = "NDVI" if "NDVI" in df.columns else None

    # check for duplicates and drop only those where all columns are the same
    # Note: In theory this should not happen, but in practice it does, need to investigate why!
    before = len(df)
    df = df.drop_duplicates(keep="first")
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} exact duplicate rows.")

    dupe_mask = df.duplicated(subset=keys, keep=False)
    if dupe_mask.any():
        dups = df.loc[dupe_mask].copy()

        # Columns that are allowed to differ within a duplicate group
        allowed_vary = []
        if wt_col in df.columns:
            allowed_vary.append(wt_col)
        if ndvi_col:
            allowed_vary.append(ndvi_col)

        # If nothing is allowed to differ, we cannot resolve duplicates automatically
        if not allowed_vary:
            remaining = dups
            if not remaining.empty:
                for group_key, g in remaining.groupby(keys):
                    diff_cols = [c for c in g.columns if c not in keys and g[c].nunique(dropna=False) > 1]

                    logger.error(f"Group {group_key} has differing columns: {diff_cols}")

                    cols_to_show = keys + diff_cols
                    logger.error(f"\n{g[cols_to_show]}")
                logger.error(
                    "Duplicate entries found for the same date in simulation and no resolvable column to break ties."
                )
                logger.error(f"\n{remaining}")
                raise ValueError("Duplicate entries still found in the simulation data and no column to resolve them.")

        # Columns that must be identical across the group (everything except keys + allowed_vary)
        non_vary_cols = [c for c in df.columns if c not in set(keys + allowed_vary)]

        # For each (SimulationID, Clock.Today) group, check if all non-vary columns are identical
        same_non_vary = dups.groupby(keys)[non_vary_cols].nunique(dropna=False).le(1).all(axis=1)
        resolvable_keys = same_non_vary.index[same_non_vary]

        # In resolvable groups: keep one row according to:
        #   - Prefer lowest NDVI if NDVI exists
        #   - If multiple rows share that NDVI, prefer highest Wheat.Total.Wt if it exists
        #   - Otherwise keep the first
        if len(resolvable_keys) > 0:
            good_dups = dups[dups.set_index(keys).index.isin(resolvable_keys)].copy()

            if wt_col in good_dups.columns:
                good_dups[wt_col] = pd.to_numeric(good_dups[wt_col], errors="coerce")
            if ndvi_col and ndvi_col in good_dups.columns:
                good_dups[ndvi_col] = pd.to_numeric(good_dups[ndvi_col], errors="coerce")

            keep_idx = []
            for _, g in good_dups.groupby(keys):
                g_sel = g

                # Step 1: prefer lowest NDVI if available
                if ndvi_col and ndvi_col in g_sel.columns:
                    min_ndvi = g_sel[ndvi_col].min()
                    g_sel = g_sel[g_sel[ndvi_col] == min_ndvi]

                # Step 2: if Wheat.Total.Wt exists, prefer highest Wt among remaining
                if wt_col in g_sel.columns:
                    idx = g_sel[wt_col].idxmax()
                else:
                    idx = g_sel.index[0]

                keep_idx.append(idx)

            drop_idx = good_dups.index.difference(keep_idx)

            if len(drop_idx) > 0:
                msg = f"Dropping {len(drop_idx)} duplicate rows"
                if ndvi_col and wt_col in df.columns:
                    msg += f" (kept lowest {ndvi_col} and, if tied, highest {wt_col})."
                elif ndvi_col:
                    msg += f" (kept lowest {ndvi_col})."
                elif wt_col in df.columns:
                    msg += f" (kept highest {wt_col})."
                logger.info(msg)
                df = df.drop(index=drop_idx)

        # If any groups still have duplicates, they differ in other columns which cannot be resolved automatically
        remaining = df[df.duplicated(subset=keys, keep=False)]
        if not remaining.empty:
            logger.error(
                "Duplicate entries found for the same date in simulation (differences beyond allowed varying columns):"
            )

            for group_key, g in remaining.groupby(keys):
                diff_cols = [c for c in g.columns if c not in keys and g[c].nunique(dropna=False) > 1]

                logger.error(f"Group {group_key} has differing columns: {diff_cols}")

                cols_to_show = keys + diff_cols
                logger.error(f"\n{g[cols_to_show]}")

            raise ValueError("Duplicate entries still found in the simulation data.")

    # Cleanup date and set as index
    df["Date"] = df["Clock.Today"].dt.floor("D")
    df.drop(columns="Clock.Today", inplace=True)
    df.set_index("Date", inplace=True)

    return df

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
