import concurrent.futures
import multiprocessing
import os
import signal
import subprocess
import sys
import time
from tqdm import tqdm


import click
import geopandas as gpd
import numpy as np
import pystac_client
import rasterio as rio
import requests
from pystac.extensions.eo import EOExtension
from rasterio.warp import Resampling, reproject, calculate_default_transform

# Set GDAL environment variables to enable multi-threaded downloads
os.environ["GDAL_ENABLE_CURL_MULTI"] = "YES"
os.environ["GDAL_HTTP_MULTIRANGE"] = "YES"


def save_band(metadata, data, item, band_name, resolution, output_folder):
    output_path = os.path.join(output_folder, f"{item.id}_{band_name}_{resolution}m.tif")

    metadata.update(
        {
            "driver": "GTiff",
            "comress": "LZW",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
    )

    with rio.open(
        output_path,
        "w",
        **metadata,
    ) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, band_name)

    return output_path


def load_resample(tile_path, resolution, mask=None):
    with rio.open(tile_path) as src:
        crs = src.crs
        bounds = src.bounds

        if src.count > 1:
            raise ValueError(
                "The input tile has more than one band. Currently only handling single band rasters."
            )
        
        # TODO: Might have to use np.allclose for floating point precision in CRS
        # Might instead want to only allow resampling to the dimensions (resolution) of another band of the tile

        # Resample if not already at target resolution
        if src.res != (resolution, resolution):
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs, src.crs, src.width, src.height, *src.bounds, resolution=resolution
            )

            resampled = np.empty((dst_height, dst_width), dtype=src.dtypes[0])
            reproject(
                source=rio.band(src, 1),
                destination=resampled,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=crs,
                resampling=Resampling.nearest,
                num_threads=4,
            )
            data = resampled
            transform = dst_transform
        else:
            data = src.read(1)
            transform = src.transform

        # Apply mask if provided. Using original nodata value for less confusion
        # Expects a binary mask!
        if mask is not None:
            data = mask * data

        # Save the band
        profile = src.profile
        profile.update(
            {
                "height": data.shape[0],
                "width": data.shape[1],
                "transform": transform,
                "dtype": src.dtypes[0],
                "nodata": src.nodata,
                "crs": crs,
            }
        )

        return profile, data


def download_resample(tile_path, resolution, item, band_name, mask=None, output_folder=None):
    """
    Download and resample the tile to the desired resolution entirely in memory.
    If a mask is provided it will be applied to the data.
    """

    profile, data = load_resample(tile_path, resolution, mask)

    out_path = save_band(profile, data, item, band_name, resolution, output_folder)
    return band_name, out_path


def combine_bands(output_file_path, band_paths, band_names, create_gtiff=False, blocksize=256):
    # Use order of band_names to create the VRT
    band_paths_ordered = [band_paths[band_name] for band_name in band_names]

    try:
        # TODO use blocksize also for vrt if gdal is newer than (not sure which one need to check)
        vrt_build_command = [
            "gdalbuildvrt",
            "-separate",
            "-overwrite",
            output_file_path,
            *band_paths_ordered,
        ]
        subprocess.run(vrt_build_command, shell=False, check=True)

        if create_gtiff:
            # Create a GTiff from the VRT for easier exporting.
            output_file_path_gtiff = output_file_path.replace(".vrt", ".tif")
            geotiff_build_command = [
                "gdal_translate",
                "-of",
                "GTiff",
                "-co",
                "TILED=YES",
                "-co",
                "compress=LZW",
                "-co",
                f"blocksize={blocksize}",
                output_file_path,
                output_file_path_gtiff,
            ]
            subprocess.run(geotiff_build_command, shell=False, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Subprocess error: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
        if os.path.exists(output_file_path):
            try:
                os.remove(output_file_path)
            except:
                pass
        raise

    return output_file_path


def download_file(url, output_path):
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except (requests.RequestException, IOError) as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def process_scene(
    item,
    band_names,
    resolution,
    metadata_asset_names,
    mask_bands,
    mask_band_processor,
    save_mask,
    custom_band_processor,
    output_folder,
):
    # Step 1: Download metadata if requested
    metadata_file_paths = {}
    if metadata_asset_names:
        for metadata_asset_name in metadata_asset_names:
            if metadata_asset_name not in item.assets:
                raise ValueError(f"Metadata asset '{metadata_asset_name}' not found in item.")

            metadata_url = item.assets[metadata_asset_name].href
            metadata_ext = os.path.splitext(metadata_url)[1]
            metedata_out_file_name = f'{item.id}_{metadata_asset_name}{metadata_ext}'
            metadata_out_path = os.path.join(output_folder, metedata_out_file_name)
            download_file(metadata_url, metadata_out_path)
            metadata_file_paths[metadata_asset_name] = metadata_out_path

    # Step 2: Prepare mask by resampling mask bands to target resolution
    # If a custom mask band processing function is provided, use that
    mask = None
    if mask_bands:
        if len(mask_bands) > 1 and not mask_band_processor:
            raise ValueError("Maskband processing func required for multiple mask bands")

        print("Downloading mask bands and resampling to target resolution...")
        maskbands = {}
        for mask_band_name in mask_bands:
            mask_band_path = item.assets[mask_band_name].href
            meta_and_mask = load_resample(mask_band_path, resolution)
            maskbands[mask_band_name] = meta_and_mask

        # The band contains 1 for pixels to keep and 0 for pixels to mask
        if not mask_band_processor and len(mask_bands) == 1:
            mask_metadata, mask = maskbands[mask_bands.keys()[0]]
        else:
            print('Processing mask bands to binary mask.')
            mask_metadata, mask = mask_band_processor(
                maskbands, resolution, item, output_folder
            )
        
        # Validate that the mask is binary
        if not np.array_equal(np.unique(mask), [0, 1]):
            raise ValueError(f"Mask must be binary (0 and 1 values only). Values found: {np.unique(mask)}")

    # Step 3: Download and resample bands of interest
    print("Downloading band data and resampling...")
    band_paths = {}
    for band_name in band_names:
        print(f"Downloading and resampling band {band_name}...")
        band_path = item.assets[band_name].href
        band_name, out_path = download_resample(
            band_path, resolution, item, band_name, mask, output_folder=output_folder
        )
        band_paths[band_name] = out_path

    if save_mask:
        mask_band_path = save_band(mask_metadata, mask, item, "mask", resolution, output_folder)
        band_paths["mask"] = mask_band_path
        band_names.append("mask")

    # Step 4: If custom band processing is provided, apply it
    # To bandorder will be specified by the names provided in band_names
    if custom_band_processor:
        print("Applying custom band processing...")
        band_paths, band_names = custom_band_processor(
            item, band_paths, band_names, resolution, mask, metadata_file_paths, output_folder
        )

    # Step 5: Combine band into a vrt (or GTiff if requested)
    print(f"Combining bands into single file for tile {item.id}...")
    vrt_path = os.path.join(
        output_folder, f"{item.id}_{resolution}m_{item.datetime.strftime('%Y-%m-%d')}.vrt"
    )
    combine_bands(vrt_path, band_paths, band_names)

    return vrt_path

def execution_wrapper(args):
    try:
        return process_scene(*args)
    except Exception as e:
        print(f"Error during processing: {e}")
        return None

def download_process_parallel(
    items,
    bands,
    resolution,
    metadata_asset_names,
    mask_bands,
    mask_band_processor,
    save_mask,
    custom_band_processor,
    output_folder,
    num_workers,
):
    output_paths = []

    task_args = [
        (
            item,
            bands,
            resolution,
            metadata_asset_names,
            mask_bands,
            mask_band_processor,
            save_mask,
            custom_band_processor,
            output_folder,
        )
        for item in items
    ]

    output_paths = []
    pool = multiprocessing.Pool(processes=num_workers)

    try:
        for result in tqdm(pool.imap_unordered(execution_wrapper, task_args), total=len(task_args), desc="Processing scenes"):
            if result:
                output_paths.append(result)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print("Interrupted by user. Terminating pool...")
        pool.terminate()
        pool.join()

    return output_paths


def run_pipeline(
    start_date,
    end_date,
    resolution,
    roi_gdf,
    bands,
    output_folder,
    stack_catalog_url,
    stac_collection_name,
    max_cloud_cover,
    num_workers,
    metadata_asset_names,
    mask_bands,
    save_mask,
    mask_band_processor=None,
    custom_band_processor=None,
):
    # Open the STAC catalog
    catalog = pystac_client.Client.open(stack_catalog_url)

    # Build query for cloud cover if 'eo' extension is available
    query = {}
    if max_cloud_cover is not None:
        collection = catalog.get_collection(stac_collection_name)
        if not EOExtension.has_extension(collection):
            raise ValueError(
                f"The collection '{stac_collection_name}' does not support the EO extension required for 'eo:cloud_cover' filtering."
            )

        query = {"eo:cloud_cover": {"lt": max_cloud_cover}}

    # Search for STAC-items in the STAC catalog
    print(f"Searching for items from {start_date} to {end_date}...")
    t0 = time.time()
    search = catalog.search(
        collections=[stac_collection_name],
        intersects=roi_gdf.geometry.values[0],
        datetime=f"{start_date}/{end_date}",
        query=query,
    )

    items = search.item_collection()
    print(f"Found {len(items)} items")
    print(f"Search took {time.time() - t0:.2f} seconds")

    # Download all items
    print("Downloading and processing items...")
    t1 = time.time()
    output_paths = download_process_parallel(
        items,
        bands,
        resolution,
        metadata_asset_names,
        mask_bands,
        mask_band_processor,
        save_mask,
        custom_band_processor,
        output_folder,
        num_workers,
    )

    print(f"\n Download data and created {len(output_paths)} files in {output_folder}")
    print(f"\nTotal runtime: {time.time() - t1:.2f} seconds")


@click.command()
@click.option("--start-date", type=str, required=True, help="Start date in YYYY-MM-DD format")
@click.option("--end-date", type=str, required=True, help="End date in YYYY-MM-DD format")
@click.option(
    "--resolution", required=True, type=int, help="Target resolution in original units (e.g meters for Sentinel-2)"
)
@click.option(
    "--geojson-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the GeoJSON/shapefile defining the area of interest",
)
@click.option(
    "--bands",
    type=str,
    default="green,red,rededge1,rededge2,rededge3,nir08,swir16,swir22,scl,cloud,snow",
    help="Comma-separated list of bands to download and resample. If you want to download mask bands without applying them, specify them here.",
)
@click.option(
    "--output-folder",
    type=click.Path(),
    required=True,
    help="Folder where output files will be saved",
)
@click.option(
    "--stack-catalog-url",
    type=str,
    default="https://earth-search.aws.element84.com/v1",
    help="URL of the STAC catalog to use for searching",
)
@click.option(
    "--stac-collection-name",
    type=str,
    default="sentinel-2-c1-l2a",
    help="STAC collection to use for searching",
)
@click.option(
    "--max-cloud-cover",
    type=int,
    default=70,
    help="Maximum cloud cover percentage (0-100) of a tile. \
        The collection must support the EO extension for this to work.",
)
@click.option(
    "--num-workers",
    type=int,
    default=None,
    help="Number of parallel workers (defaults to CPU count - 1)",
)
@click.option(
    "--metadata_asset_names",
    type=str,
    default="granule_metadata",
    help="Comma-separated list of metadata asset names to download",
)
@click.option(
    "--mask-bands",
    type=str,
    help="Comma-separated list of mask bands to process.\
        If a single band is provided, it will be used as a mask. This band then has to be binary. \
        If multiple bands are provided, a custom band processor function must be provided. \
        If you simply want to download mask bands without applying them, specify them in the bands option.",
    default=None
)
@click.option(
    "--save-mask",
    is_flag=True,
    help="Save the mask as an own band. Will be the last band.",
)
def main(
    start_date,
    end_date,
    resolution,
    geojson_path,
    bands,
    output_folder,
    stack_catalog_url,
    stac_collection_name,
    max_cloud_cover,
    num_workers,
    metadata_asset_names,
    mask_bands,
    save_mask,
):
    print("Starting satellite data download pipeline.")

    # TODOs
    # - Only fetch data within geometry if COGs are available using rasterio windowed read
    # - Add standard mask processors for some collections we know
    # - Also allow only downloading without resampling
    # - Handle download failures, retries

    bands = bands.split(",")
    metadata_asset_names = metadata_asset_names.split(",") if metadata_asset_names else None
    mask_bands = mask_bands.split(",") if mask_bands else None

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Set up parallel processing
    if num_workers is None:
        num_workers = max(1, int(multiprocessing.cpu_count() - 1))
    multiprocessing.set_start_method("spawn")
    print(f"Using {num_workers} workers out of {multiprocessing.cpu_count()} available cores")

    # Read area of interest
    print("Reading GeoDataFrame")
    gdf = gpd.read_file(geojson_path)
    gdf = gdf.to_crs(epsg=4326)

    run_pipeline(
        start_date=start_date,
        end_date=end_date,
        resolution=resolution,
        roi_gdf=gdf,
        bands=bands,
        output_folder=output_folder,
        stack_catalog_url=stack_catalog_url,
        stac_collection_name=stac_collection_name,
        max_cloud_cover=max_cloud_cover,
        num_workers=num_workers,
        metadata_asset_names=metadata_asset_names,
        mask_bands=mask_bands,
        save_mask=save_mask,
    )
    print("Download Pipeline completed successfully.")


if __name__ == "__main__":
    main()
