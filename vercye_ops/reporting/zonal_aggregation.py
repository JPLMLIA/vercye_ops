import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from exactextract import exact_extract

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


def compute_zonal_yield_stats(
    yield_mosaic_tif: str,
    coverage_mask_tif: str,
    shapefile_path: str,
    name_column: str,
) -> pd.DataFrame:
    """
    Compute zonal yield statistics for each polygon in the shapefile using exactextract.

    All statistics are computed from the reprojected equal-area mosaic, ensuring
    uniform pixel area and consistent numbers across all aggregation levels.

    Parameters
    ----------
    yield_mosaic_tif : str
        Path to the reprojected equal-area yield mosaic (kg/ha per pixel).
    coverage_mask_tif : str
        Path to the coverage mask (uint8: 1=covered by primary region, 0=not, 255=nodata).
        Must be in the same CRS and grid as the yield mosaic.
    shapefile_path : str
        Path to the shapefile defining regions for aggregation.
    name_column : str
        Column in the shapefile used for region names in output.
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - region: region name from name_column
        - mean_yield_kg_ha: mean yield of valid pixels
        - median_yield_kg_ha: median yield of valid pixels
        - total_cropland_area_ha: area of valid (non-nodata) yield pixels
        - total_production_kg: sum(yield_kg_ha * pixel_area_ha)
        - total_production_ton: total_production_kg / 1000
        - coverage_pct: percentage (0-100) of polygon area covered by primary regions
        - covered_area_ha: area with primary region coverage
        - total_polygon_area_ha: full polygon area in equal-area CRS
    """
    logger.info(f"Computing zonal yield stats for {shapefile_path} using column '{name_column}'")

    # Load shapefile and reproject to match mosaic CRS
    gdf = gpd.read_file(shapefile_path)

    if name_column not in gdf.columns:
        raise ValueError(
            f"Column '{name_column}' not found in shapefile {shapefile_path}. "
            f"Available columns: {list(gdf.columns)}"
        )

    # Deduplicate: shapefile may have multiple rows per region (one per year for reference data).
    # For zonal stats we only need unique geometries.
    n_before = len(gdf)
    gdf = gdf.drop_duplicates(subset=[name_column]).reset_index(drop=True)
    if len(gdf) < n_before:
        logger.info(f"Deduplicated shapefile from {n_before} to {len(gdf)} unique regions by '{name_column}'")

    with rasterio.open(yield_mosaic_tif) as src:
        mosaic_crs = src.crs
        pixel_res_x = abs(src.transform[0])
        pixel_res_y = abs(src.transform[4])

    # Pixel area in hectares (equal-area CRS, so uniform)
    pixel_area_m2 = pixel_res_x * pixel_res_y
    pixel_area_ha = pixel_area_m2 / 10_000

    logger.info(f"Pixel area: {pixel_area_m2:.2f} m2 ({pixel_area_ha:.6f} ha)")

    # Reproject shapefile to mosaic CRS if needed
    if gdf.crs is not None and gdf.crs != mosaic_crs:
        logger.info(f"Reprojecting shapefile from {gdf.crs} to {mosaic_crs}")
        gdf = gdf.to_crs(mosaic_crs)
    elif gdf.crs is None:
        logger.warning("Shapefile has no CRS set. Assuming it matches the mosaic CRS.")

    # Run exactextract on the yield mosaic
    # We request: mean, median, count (valid pixels), sum (for total production)
    logger.info("Running exactextract on yield mosaic...")
    yield_stats = exact_extract(
        yield_mosaic_tif,
        gdf,
        ops=["mean", "median", "count", "sum"],
        include_cols=[name_column],
        output="pandas",
    )

    # Run exactextract on the coverage mask
    # count where value == 1 (covered), and total count of all pixels in polygon
    logger.info("Running exactextract on coverage mask...")
    coverage_stats = exact_extract(
        coverage_mask_tif,
        gdf,
        ops=["sum", "count"],
        include_cols=[name_column],
        output="pandas",
    )

    # Compute polygon areas in the equal-area CRS
    polygon_areas_m2 = gdf.geometry.area
    polygon_areas_ha = polygon_areas_m2 / 10_000

    # Build result DataFrame
    results = pd.DataFrame()
    results["region"] = yield_stats[name_column].astype(str)
    results["mean_yield_kg_ha"] = yield_stats["mean"].round(0).astype(int)
    results["median_yield_kg_ha"] = yield_stats["median"].round(0).astype(int)

    # Total cropland area = count of valid yield pixels * pixel area
    valid_pixel_count = yield_stats["count"]
    results["total_cropland_area_ha"] = (valid_pixel_count * pixel_area_ha).round(2)

    # Total production = sum of (yield_kg_ha * pixel_area) for valid pixels
    # exactextract "sum" gives sum of pixel values; multiply by pixel_area_ha for production
    results["total_production_kg"] = (yield_stats["sum"] * pixel_area_ha).round(0).astype(int)
    results["total_production_ton"] = (results["total_production_kg"] / 1000).round(3)

    # Coverage: sum of coverage mask (=number of covered pixels), total count of pixels in polygon
    covered_pixels = coverage_stats["sum"]
    total_pixels_in_polygon = coverage_stats["count"]
    results["coverage_pct"] = np.where(
        total_pixels_in_polygon > 0,
        (covered_pixels / total_pixels_in_polygon * 100).round(1),
        0.0,
    )
    results["covered_area_ha"] = (covered_pixels * pixel_area_ha).round(2)
    results["total_polygon_area_ha"] = polygon_areas_ha.round(2).values

    # Sort by region name
    results = results.sort_values("region").reset_index(drop=True)

    # Log warnings for low coverage
    low_coverage = results[results["coverage_pct"] < 50]
    if len(low_coverage) > 0:
        logger.warning(
            f"{len(low_coverage)} regions have <50% coverage by primary regions: "
            f"{list(low_coverage['region'].values)}"
        )

    zero_coverage = results[results["coverage_pct"] == 0]
    if len(zero_coverage) > 0:
        logger.warning(
            f"{len(zero_coverage)} regions have 0% coverage (no primary region data): "
            f"{list(zero_coverage['region'].values)}"
        )

    logger.info(f"Computed stats for {len(results)} regions")
    return results


def extract_reference_from_shapefile(
    shapefile_path: str,
    name_column: str,
    reference_yield_column: str,
    year_column: str = None,
    year: str = None,
) -> pd.DataFrame:
    """
    Extract reference yield data from a shapefile's attribute table, optionally filtered by year.

    The shapefile may have multiple rows per region (one per year) when year_column is set.
    This function filters to the specified year and returns one row per region.

    Parameters
    ----------
    shapefile_path : str
        Path to the shapefile or GeoJSON.
    name_column : str
        Column for region names.
    reference_yield_column : str
        Column containing reference yield values (kg/ha).
    year_column : str, optional
        Column containing the year for each row.
    year : str, optional
        The year to filter by. Required if year_column is set.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'region' and 'reported_mean_yield_kg_ha' columns.
    """
    gdf = gpd.read_file(shapefile_path)

    if name_column not in gdf.columns:
        raise ValueError(f"Column '{name_column}' not found in shapefile.")
    if reference_yield_column not in gdf.columns:
        raise ValueError(f"Column '{reference_yield_column}' not found in shapefile.")

    # Filter by year if year_column is provided
    if year_column and year:
        if year_column not in gdf.columns:
            raise ValueError(f"Year column '{year_column}' not found in shapefile.")

        # Normalize both sides to handle int, float, and string year values
        # e.g., shapefile might have 2019 (int), 2019.0 (float), or "2019" (str)
        def _normalize_year(val):
            try:
                return str(int(float(val)))
            except (ValueError, TypeError):
                return str(val)

        normalized_col = gdf[year_column].apply(_normalize_year)
        normalized_target = _normalize_year(year)
        gdf = gdf[normalized_col == normalized_target]
        if gdf.empty:
            logger.warning(f"No rows found for year {year} in column '{year_column}'")
            return pd.DataFrame(columns=["region", "reported_mean_yield_kg_ha"])

    ref_df = pd.DataFrame(
        {
            "region": gdf[name_column].astype(str),
            "reported_mean_yield_kg_ha": gdf[reference_yield_column],
        }
    )

    return ref_df.dropna(subset=["reported_mean_yield_kg_ha"]).reset_index(drop=True)
