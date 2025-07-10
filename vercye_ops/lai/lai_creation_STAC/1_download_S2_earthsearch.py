import os
import time

import click
import geopandas as gpd
import numpy as np

from vercye_ops.utils.init_logger import get_logger

from s2_download_hooks import add_geometry_bands, build_s2_masking_hook, s2_harmonization_processor
from stac_downloader.raster_processing import ResamplingMethod
from stac_downloader.stac_downloader import STACDownloader

from shapely.validation import make_valid

RESAMPLING_METHOD = ResamplingMethod.NEAREST  # Resampling method for raster assets
SCL_KEEP_CLASSES = [4, 5]

logger = get_logger()
logger.setLevel('INFO')


@click.command()
@click.option(
    "--satellite",
    type=click.Choice(["S2"], case_sensitive=False),
    default="S2",
    help="Satellite source to use. Currently supported 'S2' (Sentinel-2)",
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
    "--cloudprob-thresh",
    type=int,
    default=25,
    help="Threshold for pixels to be considered clouds (S2Cloudless).",
)
@click.option(
    "--snowprob-thresh",
    type=int,
    default=15,
    help="Threshold for pixel to be considered as snow.",
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
    cloudprob_thresh,
    snowprob_thresh,
    num_workers,
    overwrite
):
    stac_downloader = STACDownloader(catalog_url="https://earth-search.aws.element84.com/v1", logger=logger)

    satellite = satellite.lower()
    if satellite == "s2":
        stac_collection_name = "sentinel-2-c1-l2a" 
        mask_bands = ["scl", "cloud", "snow"]
        
        metadata_asset_names = ["granule_metadata",]
        # 'cosVZA', 'cosSZA', 'cosRAA' will be prepended by the custom band processor
        if resolution == 10:
            # ['B2', 'B3', 'B4', 'B8']
            band_assets = [
                "blue",
                "green",
                "red",
                "nir"
            ]
        else:
            # ['B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']
            band_assets = [
                "green",
                "red",
                "rededge1",
                "rededge2",
                "rededge3",
                "nir08",
                "swir16",
                "swir22",
            ]
    else:
        raise Exception("Currently only supporting S2 download.")

    # Register masking hook based on SCL & Cloud Probs with a threshold for cloud and snow
    s2_masking_hook = build_s2_masking_hook(
        cloud_thresh=cloudprob_thresh,
        snowprob_thresh=snowprob_thresh,
        scl_keep_classes=SCL_KEEP_CLASSES,
    )
    stac_downloader.register_masking_hook(s2_masking_hook)

    # Register hook to harmonize the sentinel-2 data to match baseline < 4.0 (-1000 for newer)
    stac_downloader.register_bandprocessing_hook(s2_harmonization_processor, band_assets=band_assets)

    # Register geometry bands hook to create bands adding cosines of angles from metadata
    stac_downloader.register_postdownload_hook(add_geometry_bands)

    # Query stac catalog
    logger.info(f"Searching for items from {start_date} to {end_date}...")
    t0 = time.time()
    gdf = gpd.read_file(geojson_path)
    gdf = gdf.to_crs(epsg=4326)
    gdf["geometry"] = gdf["geometry"].apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)

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
    except Exception as e:
        # If it throws an error, we simplify each geometry to a bounding box and try again.
        logger.warning('Not able to query STAC for true intersection. Trying bounding box.')
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

    logger.info(f"Found {len(items)} items")
    logger.info(f"Search took {time.time() - t0:.2f} seconds")

    # Download the items
    logger.info("Starting download of items...")
    os.makedirs(output_dir, exist_ok=True)
    downloaded_item_paths = stac_downloader.download_items(
        items=items,
        raster_assets=band_assets,
        file_assets=metadata_asset_names,
        mask_assets=mask_bands,
        output_folder=output_dir,
        overwrite=overwrite,
        resolution=resolution,
        resampling_method=RESAMPLING_METHOD,
        num_workers=num_workers,
    )

    logger.info(f"Downloads saved under {output_dir}.")


if __name__ == "__main__":
    main()
