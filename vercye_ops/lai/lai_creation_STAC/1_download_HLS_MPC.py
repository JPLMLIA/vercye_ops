import os
import time
from typing import Dict, List

import click
import geopandas as gpd
import numpy as np
import planetary_computer
import rasterio as rio
from pystac.item import Item as pyStacItem
from shapely.validation import make_valid
from stac_downloader.raster_processing import ResamplingMethod
from stac_downloader.stac_downloader import STACDownloader

from vercye_ops.utils.init_logger import get_logger

RESAMPLING_METHOD = ResamplingMethod.NEAREST  # Resampling method for raster assets
SCL_KEEP_CLASSES = [4, 5]

# All others should be int16 (default)
RASTER_BAND_DTYPES = {"VZA": np.uint16, "SZA": np.uint16, "SAA": np.uint16, "VAA": np.uint16}

logger = get_logger()


def build_snow_clouds_mask(maskbands):
    """Create a mask for pixels that are clouds, cloud shadow, adjacent to cloud(-shadow) or snow/ice.

    See Appendix A: https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf
    """
    fmask_meta, fmask = maskbands["Fmask"]
    mask = np.ones_like(fmask)  # Start with all valid (1)

    is_cloud = ((fmask >> 1) & 1).astype(bool)  # bit 1
    is_cloud_shadow = ((fmask >> 3) & 1).astype(bool)  # bit 3
    is_adjacent = ((fmask >> 2) & 1).astype(bool)  # bit 2
    is_snow = ((fmask >> 4) & 1).astype(bool)  # bit 4

    should_be_masked = is_cloud | is_adjacent | is_snow | is_cloud_shadow
    mask = np.where(should_be_masked, 0, mask)

    return {}, mask


def create_geometry_bands(
    item: pyStacItem,
    band_paths: Dict[str, str],
    band_names: List[str],
    mask: np.ndarray,
    file_asset_paths: Dict[str, str],
    resolution: float,
    output_folder: str,
):
    """Create bands containing cosVZA, cosSZA, cosRAA from present bands in HLS.
    Based on Leaf Toolbox in GEE (toolsL8 script).
    """

    # Exporting to int16 instead of uint16 as in GEE! Shouldnt be a problem since max range should be 10000
    # due to angle = cos(x) * 10000
    def save_cos_band(cos_arr: np.ndarray, out_path: str, profile: dict):

        # use 11000 as nodata value and replace all initial nodata values
        orig_nodata = profile.get("nodata")
        new_nodata = -9999
        if orig_nodata is None:
            raise Exception("Not yet handling cases with no set nodata value in angles band.")
        cos_arr = np.where(cos_arr == orig_nodata, new_nodata, cos_arr)

        profile.update(dtype=np.int16, count=1, compress="lzw", nodata=new_nodata)

        with rio.open(out_path, "w", **profile) as dst:
            dst.write(np.squeeze(cos_arr), 1)

    with rio.open(band_paths["VZA"]) as src:
        vza = src.read()
        cos_vza = np.cos(np.deg2rad(vza / 100)) * 10000
        profile_vza = src.profile

    cos_vza_path = os.path.join(output_folder, f"{item.id}_cos_vza_{str(resolution)}m.tif")
    save_cos_band(cos_vza, cos_vza_path, profile_vza)

    with rio.open(band_paths["SZA"]) as src:
        sza = src.read()
        cos_sza = np.cos(np.deg2rad(sza / 100)) * 10000
        profile_sza = src.profile

    cos_sza_path = os.path.join(output_folder, f"{item.id}_cos_sza_{str(resolution)}m.tif")
    save_cos_band(cos_sza, cos_sza_path, profile_sza)

    with rio.open(band_paths["VAA"]) as src:
        vaa = src.read()

    with rio.open(band_paths["SAA"]) as src:
        saa = src.read()
        profile_saa = src.profile

    raa = vaa - saa
    cos_raa = np.cos(np.deg2rad(raa / 100)) * 10000
    cos_raa_path = os.path.join(output_folder, f"{item.id}_cos_raa_{str(resolution)}m.tif")
    save_cos_band(cos_raa, cos_raa_path, profile_saa)

    # Ensure correct bandorder for output
    band_paths["cos_vza"] = cos_vza_path
    band_paths["cos_sza"] = cos_sza_path
    band_paths["cos_raa"] = cos_raa_path

    # Drop original VZA SZA VAA SAA bands and instead add new bands
    band_names = ["cos_vza", "cos_sza", "cos_raa"] + band_names[4:]

    return band_paths, band_names


def band_scaling_hook(raster: np.ndarray, raster_profile: dict, item: pyStacItem):
    """Float 32 bands should be saved as int 16 and need therefore be scaled properly"""


@click.command()
@click.option(
    "--satellite",
    type=click.Choice(["HLS_S30", "HLS_L30"], case_sensitive=False),
    default="HLS_S30",
    help="Satellite source to use. Currently supported 'HLS_L30' (landsat version of HLS) or HLS_S30 (Sentinel Version of HLS)",
)
@click.option("--start-date", type=str, required=True, help="Start date in YYYY-MM-DD format")
@click.option("--end-date", type=str, required=True, help="End date in YYYY-MM-DD format")
@click.option("--resolution", type=int, help="Target resolution in meters.")
@click.option(
    "--geojson-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the GeoJSON/shapefile defining the area of interest",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    required=True,
    help="Folder where output files will be saved",
)
@click.option(
    "--max-cloud-cover",
    type=int,
    default=70,
    help="Maximum cloud cover percentage (0-100) of a tile",
)
@click.option(
    "--num-workers",
    type=int,
    default=None,
    help="Number of parallel workers (defaults to CPU count - 1)",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite already downloaded files.",
)
def main(
    satellite,
    start_date,
    end_date,
    resolution,
    geojson_path,
    output_dir,
    max_cloud_cover,
    num_workers,
    overwrite,
):
    modifier = planetary_computer.sign  # Required from MPC
    stac_downloader = STACDownloader(
        catalog_url="https://planetarycomputer.microsoft.com/api/stac/v1",
        logger=logger,
        stac_item_modifier=modifier,
    )

    mask_bands = ["Fmask"]
    metadata_asset_names = []

    satellite = satellite.lower()
    if satellite == "hls_s30":
        stac_collection_name = "hls2-s30"
        band_assets = ["VZA", "SZA", "VAA", "SAA", "B03", "B04", "B8A", "B11", "B12"]
    elif satellite == "hls_l30":
        stac_collection_name = "hls2-l30"
        band_assets = ["VZA", "SZA", "VAA", "SAA", "B03", "B04", "B05", "B06", "B07"]
    else:
        raise Exception("Currently only supporting HLS_S30 or HLS_L30 download.")

    # Register masking hook based Fmask for clouds/shadows/snow/adjacent pixels
    stac_downloader.register_masking_hook(build_snow_clouds_mask)
    stac_downloader.register_postdownload_hook(create_geometry_bands)

    logger.info(f"Searching for items from {start_date} to {end_date}...")
    t0 = time.time()

    gdf = gpd.read_file(geojson_path)
    gdf = gdf.to_crs(epsg=4326)
    gdf["geometry"] = gdf["geometry"].apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)

    if gdf.empty:
        raise ValueError("Empty shapefile provided.")

    # Query Stac catalog
    # First try using all exact geometries. However, sometimes this is a too large request and might throw an error
    try:
        geometry = gdf.geometry.union_all()
        items = stac_downloader.query_catalog(
            collection_name=stac_collection_name,
            start_date=start_date,
            end_date=end_date,
            geometry=geometry,
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        )
    except Exception:
        # If it throws an error, we simplify each geometry to a bounding box and try again.
        logger.warning("Not able to query STAC for true intersection. Trying bounding box.")
        # Create bounding box (envelope) of each geometry
        envelopes = gdf.geometry.envelope

        # Combine all bounding boxes into a single geometry
        unified_geometry = envelopes.union_all()
        geometry = unified_geometry

        items = stac_downloader.query_catalog(
            collection_name=stac_collection_name,
            start_date=start_date,
            end_date=end_date,
            geometry=geometry,
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        )

    print(f"Found {len(items)} items")
    logger.info(f"Found {len(items)} items")
    logger.info(f"Search took {time.time() - t0:.2f} seconds")

    # Download the items
    logger.info("Starting download of items...")
    os.makedirs(output_dir, exist_ok=True)
    stac_downloader.download_items(
        items=items,
        raster_assets=band_assets,
        file_assets=metadata_asset_names,
        mask_assets=mask_bands,
        output_folder=output_dir,
        overwrite=overwrite,
        resolution=resolution,
        resampling_spec=RESAMPLING_METHOD,
        num_workers=num_workers,
        raster_asset_target_dtypes=RASTER_BAND_DTYPES,
    )

    logger.info(f"Downloads saved under {output_dir}.")


if __name__ == "__main__":
    main()
